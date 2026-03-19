"""
SearchRescueAgent — LLM-driven rescue agent with MindForge-style cognitive loop.

Extends LLMAgentBase which handles all infrastructure (navigation, carry
retry, rendezvous, action validation, task injection).

This class implements a multi-stage async pipeline:
    CRITIC → PLANNING → REASONING → EXECUTE

Each LLM stage is non-blocking: submit a call, return Idle, poll next tick.
Stage outputs flow forward via _pipeline_context.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from matrx.agents.agent_utils.state import State
from matrx.messages.message import Message

from agents1.async_model_prompting import get_llm_result
from agents1.capabilities import filter_tools_for_capabilities, get_capability_prompt, get_game_rules
from agents1.llm_agent_base import LLMAgentBase, SM_TASK_ASSIGNMENTS
from agents1.modules.area_tracker import AreaExplorationTracker
from agents1.modules.execution_module import execute_action
from agents1.modules.reasoning_module import ReasoningIO
from agents1.modules.task_critic_module import CriticBase
from agents1.tool_registry import REASONING_STRATEGIES, build_tool_schemas
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation

logger = logging.getLogger('SearchRescueAgent')


class PipelineStage(Enum):
    IDLE = 'idle'
    CRITIC = 'critic'
    PLANNING = 'planning'
    REASONING = 'reasoning'
    EXECUTE = 'execute'


class SearchRescueAgent(LLMAgentBase):
    """MARBLE-powered rescue agent with multi-stage cognitive pipeline.

    Pipeline per cycle:
        [CRITIC] → PLANNING → REASONING → EXECUTE

    Critic runs on all cycles except the first (no previous action to evaluate)
    and when the last action was Idle/None.
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
        env_info: Optional[EnvironmentInformation] = None,
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
            env_info=env_info,
        )
        self._strategy = strategy if strategy in REASONING_STRATEGIES else 'react'
        self.area_tracker = AreaExplorationTracker(self.env_info.get_area_cells())
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        if self._capabilities:
            self.tools_by_name, self.tool_schemas = filter_tools_for_capabilities(
                self.tool_schemas, self.tools_by_name, self._capabilities
            )

        self.reasoning = ReasoningIO('EMPTY')
        self.critic = CriticBase('EMPTY')

        # Pipeline state
        self._pipeline_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_context: Dict[str, Any] = {}
        self._is_first_cycle: bool = True
        self._last_action: Dict[str, Any] = {}

        print(
            f'[SearchRescueAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'planning={planning_mode}, caps={capabilities})'
        )

    # ── Task injection override ─────────────────────────────────────────

    def set_current_task(self, task: str) -> None:
        super().set_current_task(task)
        self._pipeline_stage = PipelineStage.IDLE
        self._pipeline_context = {}
        self._is_first_cycle = True
        self._last_action = {}

    # ── Perception ──────────────────────────────────────────────────────

    def update_knowledge(self, filtered_state: State) -> None:
        super().update_knowledge(filtered_state)
        agent_loc = filtered_state[self.agent_id]['location']
        vision = self._capabilities.get('vision', 1) if self._capabilities else 1
        self.area_tracker.update(agent_loc, vision_radius=vision)

    # ── Main decision loop ──────────────────────────────────────────────

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        self.update_knowledge(filtered_state)

        if not self._current_task:
            return self._idle()

        # Infrastructure: carry retry, navigation, rendezvous
        action = self._run_infra(filtered_state)
        if action is not None:
            return action

        # Poll pending LLM future
        if self._pending_future is not None:
            try:
                result = get_llm_result(self._pending_future)
            except Exception as exc:
                logger.warning('[%s] LLM future raised: %s', self.agent_id, exc)
                self._pending_future = None
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle()
            if result is None:
                return self._idle()
            self._pending_future = None
            return self._on_llm_result(result)

        # Advance pipeline
        return self._advance_pipeline(filtered_state)

    # ── Pipeline router ─────────────────────────────────────────────────

    def _advance_pipeline(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        if self._pipeline_stage == PipelineStage.IDLE:
            if self._is_first_cycle:
                self._is_first_cycle = False
                self._pipeline_stage = PipelineStage.PLANNING
            else:
                self._pipeline_stage = PipelineStage.CRITIC
            self._pipeline_context = {}

        if self._pipeline_stage == PipelineStage.CRITIC:
            return self._submit_critic()
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self._submit_planning()
        if self._pipeline_stage == PipelineStage.REASONING:
            return self._submit_reasoning()
        if self._pipeline_stage == PipelineStage.EXECUTE:
            return self._execute_action(filtered_state)

        return self._idle()

    def _on_llm_result(self, result) -> Tuple[Optional[str], Dict]:
        if self._pipeline_stage == PipelineStage.CRITIC:
            return self._handle_critic_result(result)
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self._handle_planning_result(result)
        if self._pipeline_stage == PipelineStage.REASONING:
            return self._handle_reasoning_result(result)
        return self._idle()

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_rescued_victims(self):
        """Derive rescued victims from SharedMemory (if tracked) or memory log."""
        if self.shared_memory:
            rescued = self.shared_memory.retrieve('rescued_victims')
            if rescued:
                return rescued
        # Fallback: scan memory for successful Drop actions at drop zone
        rescued = []
        for entry in self.memory.retrieve_all():
            if isinstance(entry, dict) and entry.get('action') in ('Drop', 'DropObject'):
                victim_id = entry.get('args', {}).get('object_id', '')
                if victim_id and victim_id not in rescued:
                    rescued.append(victim_id)
        return rescued or None

    # ── CRITIC stage ────────────────────────────────────────────────────

    def _submit_critic(self) -> Tuple[Optional[str], Dict]:
        last_name = self._last_action.get('name')
        if not last_name or last_name == 'Idle':
            self._pipeline_stage = PipelineStage.PLANNING
            return self._advance_pipeline(None)

        prompt = self.critic.get_critic_prompt({
            'current_task': self._current_task,
            'last_action': self._last_action,
            'observation': self.WORLD_STATE,
            'all_observations': self.WORLD_STATE_GLOBAL,
        })

        # P1: Inject capability context so critic doesn't suggest infeasible actions
        if self._capabilities and self._capability_knowledge == 'informed':
            cap_text = get_capability_prompt(self._capabilities)
            prompt[0]['content'] += f"\n\nAgent capabilities:\n{cap_text}"

        print(f'[{self.agent_id}] Pipeline: CRITIC — evaluating last action: {last_name}')
        self._submit_llm(prompt)
        return self._idle()

    def _handle_critic_result(self, result) -> Tuple[Optional[str], Dict]:
        text = getattr(result[0], 'content', '') or ''
        try:
            critic_result = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            critic_result = {'success': False, 'reasoning': text, 'critique': text}

        self._pipeline_context['critic_result'] = critic_result
        self.memory.update('critic_feedback', critic_result)
        print(f'[{self.agent_id}] Critic result: success={critic_result.get("success")}')

        self._pipeline_stage = PipelineStage.PLANNING
        return self._advance_pipeline(None)

    # ── PLANNING stage ──────────────────────────────────────────────────

    def _submit_planning(self) -> Tuple[Optional[str], Dict]:
        # P0: Read other agents' tasks from SharedMemory
        other_tasks = {}
        if self.shared_memory:
            all_tasks = self.shared_memory.retrieve(SM_TASK_ASSIGNMENTS) or {}
            other_tasks = {k: v for k, v in all_tasks.items() if k != self.agent_id}

        # Get raw messages for planner (messages flow into planning, not reasoning)
        agent_busy = self._nav_target is not None or self._carry_autopilot is not None
        raw_messages = self.comm.get_messages_for_prompt(limit=5, agent_busy=agent_busy)

        agent_context = {
            'previous_tasks': self.memory.retrieve_all()[-5:],
            'other_agents_tasks': other_tasks or None,
            'position': self.WORLD_STATE.get('agent', {}).get('location'),
            'nearby_objects': self.WORLD_STATE.get('victims', []) + self.WORLD_STATE.get('obstacles', []),
            'observed_objects': self.WORLD_STATE_GLOBAL,
            'carrying': self.WORLD_STATE.get('agent', {}).get('carrying'),
            'rescued_victims': self._get_rescued_victims(),
            'critic_feedback': self._pipeline_context.get('critic_result'),
            'area_exploration': self.area_tracker.get_all_summaries(),
            'messages': [
                f"[{m['from']}] ({m['message_type']}) {m['text']}"
                for m in raw_messages
            ] if raw_messages else None,
        }
        prompt = self.planner.get_planning_prompt(agent_context)

        # P0: Inject capability context so planner doesn't generate infeasible tasks
        if self._capabilities and self._capability_knowledge == 'informed':
            cap_text = get_capability_prompt(self._capabilities)
            prompt[0]['content'] += f"\n\n{cap_text}"

        print(f'[{self.agent_id}] Pipeline: PLANNING — generating next task')
        self._submit_llm(prompt)
        return self._idle()

    def _handle_planning_result(self, result) -> Tuple[Optional[str], Dict]:
        text = getattr(result[0], 'content', '') or ''
        self._pipeline_context['planned_task'] = text.strip()
        print(f'[{self.agent_id}] Planned task: {text.strip()[:100]}')

        self._pipeline_stage = PipelineStage.REASONING
        return self._advance_pipeline(None)

    # ── REASONING stage ─────────────────────────────────────────────────

    def _submit_reasoning(self) -> Tuple[Optional[str], Dict]:
        observation = dict(self.WORLD_STATE)
        global_state = self.WORLD_STATE_GLOBAL
        if any(global_state.get(k) for k in ('victims', 'obstacles', 'doors')):
            observation['known'] = {
                k: v for k, v in global_state.items()
                if k != 'teammate_positions' and v
            }
        observation['area_exploration'] = self.area_tracker.get_all_summaries()

        prompt = self.reasoning.get_reasoning_prompt({
            'task_decomposition': self._pipeline_context.get('planned_task', self._current_task),
            'observation': observation,
            'memory': self.memory.retrieve_all()[-15:],
            'critic_feedback': self._pipeline_context.get('critic_result'),
        })

        # Inject capability info and game rules
        if self._capabilities and self._capability_knowledge == 'informed':
            cap_text = get_capability_prompt(self._capabilities)
            rules_text = get_game_rules(self._capabilities, drop_zone=self.env_info.drop_zone, num_victims=self.env_info.num_victims)
            prompt[0]['content'] += f"\n\n{cap_text}\n\n{rules_text}"
        else:
            rules_text = get_game_rules(drop_zone=self.env_info.drop_zone, num_victims=self.env_info.num_victims)
            prompt[0]['content'] += f"\n\n{rules_text}"

        print(f'[{self.agent_id}] Pipeline: REASONING — choosing action')
        self._submit_llm(prompt, tools=self.tool_schemas)
        return self._idle()

    def _handle_reasoning_result(self, result) -> Tuple[Optional[str], Dict]:
        message = result[0]
        partner = next(
            (i[0] for i in self.teammates if i[0] != self.agent_id), None
        )

        # Path A: structured tool_call
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            tc = tool_calls[0]
            name = tc.function.name
            args_raw = tc.function.arguments
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            print(f'[{self.agent_id}] Tool call: {name}({args})')
            self.send_message(Message(
                content=f'Executing action {name}({args})', from_id=self.agent_id
            ))
        else:
            # Path B: plain-text fallback
            llm_text = getattr(message, 'content', '') or ''
            print(f'[{self.agent_id}] Text response: {llm_text[:120]}')
            name, args = self._mapper.parse_raw(llm_text)
            if name is None:
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle()

        self._pipeline_context['action_name'] = name
        self._pipeline_context['action_args'] = args
        self._pipeline_context['partner'] = partner

        self._pipeline_stage = PipelineStage.EXECUTE
        return self._advance_pipeline(None)

    # ── EXECUTE stage ───────────────────────────────────────────────────

    def _execute_action(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        name = self._pipeline_context['action_name']
        args = self._pipeline_context['action_args']
        partner = self._pipeline_context.get('partner')

        # Validate
        check = self._validate_action(name, args)
        if check is not None:
            self._last_action = {'name': name, 'args': args, 'result': 'validation_failed'}
            self._pipeline_stage = PipelineStage.IDLE
            return check

        # Dispatch
        action_name, kwargs, task_completing = execute_action(name, args, partner, self.agent_id)

        self.memory.update('action', {'action': action_name, 'args': kwargs})
        self._last_action = {'name': action_name, 'args': kwargs}

        # Handle communication actions
        comm_result = self._apply_communication(action_name, kwargs)
        if comm_result is not None:
            self._pipeline_stage = PipelineStage.IDLE
            return comm_result

        # Handle navigation and other actions
        result = self._apply_navigation(action_name, kwargs)
        self._pipeline_stage = PipelineStage.IDLE
        return result

