"""
SearchRescueAgent — LLM-driven rescue agent powered by MARBLE / LiteLLM.

Extends LLMAgentBase which handles all infrastructure (navigation, carry
retry, rendezvous, LLM polling, action validation, task injection).

This class is responsible only for:
    - Choosing LLM model and reasoning strategy
    - Building the per-tick reasoning prompt
    - Submitting the LLM call
"""

from typing import Dict, Optional, Tuple

from matrx.agents.agent_utils.state import State

from agents1.capabilities import filter_tools_for_capabilities, get_capability_prompt, get_game_rules
from agents1.llm_agent_base import LLMAgentBase
from agents1.modules.reasoning_module import ReasoningIO
from agents1.tool_registry import REASONING_STRATEGIES, build_tool_schemas
from memory.shared_memory import SharedMemory

logger_name = 'SearchRescueAgent'


class SearchRescueAgent(LLMAgentBase):
    """Lightweight MARBLE-powered rescue agent.

    Args:
        slowdown:       Tick slow-down factor (passed to ArtificialBrain).
        condition:      World condition ('normal' | 'strong' | 'weak').
        name:           Cooperative partner name.
        folder:         Working folder path.
        llm_model:      LiteLLM model string, e.g. ``'ollama/qwen3:8b'``.
        strategy:       Reasoning strategy: 'cot', 'react', or 'reflexion'.
        include_human:  Whether to include the human in the observation filter.
        shared_memory:  Optional SharedMemory for cross-agent state sharing.
        planning_mode:  ``'simple'`` (flat list) or ``'dag'`` (task graph).
    """

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'ollama/llama3',
        strategy: str = 'react',
        include_human: bool = True,
        shared_memory: Optional[SharedMemory] = None,
        planning_mode: str = 'simple',
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        comm_strategy: str = 'always_respond',
    ) -> None:
        super().__init__(
            slowdown=slowdown,
            condition=condition,
            name=name,
            folder=folder,
            llm_model=llm_model,
            include_human=include_human,
            shared_memory=shared_memory,
            planning_mode=planning_mode,
            api_base=api_base,
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            comm_strategy=comm_strategy,
        )
        self._strategy = strategy if strategy in REASONING_STRATEGIES else 'react'
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        # Filter tools based on capabilities
        if self._capabilities:
            self.tools_by_name, self.tool_schemas = filter_tools_for_capabilities(
                self.tool_schemas, self.tools_by_name, self._capabilities
            )

        self.reasoning = ReasoningIO('EMPTY')

        print(
            f'[SearchRescueAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'planning={planning_mode}, caps={capabilities})'
        )

    # ── Main decision loop ────────────────────────────────────────────────

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        self._tick_setup(filtered_state)

        if not self._current_task:
            return self._idle()

        # Infrastructure: carry retry, navigation, rendezvous, LLM poll
        result = self._run_preamble(filtered_state)
        if result is not None:
            return result

        # Agent reasoning: build prompt and submit LLM call
        if self._reasoning_step:
            no_tasks = (
                not self.planner.has_remaining_tasks()
                if self.planner.mode == 'dag'
                else self.task_num <= 0
            )
            if no_tasks:
                return self._idle()

            # Merge local observation with globally known objects
            observation = dict(self.WORLD_STATE)
            global_state = self.WORLD_STATE_GLOBAL
            if any(global_state.get(k) for k in ('victims', 'obstacles', 'doors')):
                observation['known'] = {
                    k: v for k, v in global_state.items()
                    if k != 'teammate_positions' and v
                }

            prompt = self.reasoning.get_reasoning_prompt({
                'task_decomposition': self.planner.get_tasks_for_reasoning(self.task_num),
                'observation': observation,
                'feedback': self._action_feedback,
                'memory': self.memory.retrieve_all()[-15:],
                'messages': self.comm.get_messages_for_prompt(limit=10),
            })

            # Inject capability info and tailored game rules into system prompt
            if self._capabilities and self._capability_knowledge == 'informed':
                cap_text = get_capability_prompt(self._capabilities)
                rules_text = get_game_rules(self._capabilities)
                extra = f"\n\n{cap_text}\n\n{rules_text}"
                prompt[0]['content'] = prompt[0]['content'] + extra
            else:
                rules_text = get_game_rules()
                prompt[0]['content'] = prompt[0]['content'] + f"\n\n{rules_text}"

            print(f'[{self.agent_id}] Submitting LLM call')
            self._submit_llm(prompt, tools=self.tool_schemas)

            if self.planner.mode == 'simple':
                self.task_num -= 1
            self._action_feedback = ''

        return self._idle()
