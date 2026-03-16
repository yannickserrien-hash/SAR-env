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
        comm_strategy: str = 'priority',
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
        )
        self._strategy = strategy if strategy in REASONING_STRATEGIES else 'react'
        self._comm_strategy = comm_strategy
        self.tools_by_name, self.tool_schemas = build_tool_schemas()
        self.reasoning = ReasoningIO('EMPTY')

        print(
            f'[SearchRescueAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'planning={planning_mode}, comm={comm_strategy})'
        )

    # ── Main decision loop ────────────────────────────────────────────────

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        self.process_observations(filtered_state)

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

            # Parse new messages and build communication context
            self._msg_handler.parse_new_messages(self.received_messages)
            comm_context = self._msg_handler.get_context_for_prompt()

            prompt = self.reasoning.get_reasoning_prompt({
                'task_decomposition': self.planner.get_tasks_for_reasoning(),
                'observation': observation,
                'all_observations': global_state,
                'memory': self.memory.retrieve_all()[-15:],
                'communication': comm_context,
            })
            print(f'[{self.agent_id}] Submitting LLM call')

            self._submit_llm(prompt, tools=self.tool_schemas)

            if self.planner.mode == 'simple':
                self.task_num -= 1

        return self._idle()
