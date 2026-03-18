"""
LLMAgentBase — Framework base class for all LLM-driven MATRX rescue agents.

Subclasses only need to implement two methods:
    - decide_on_actions(filtered_state)  — reasoning / LLM submission step
    - filter_observations(state)         — what to perceive (optional: base
                                           class provides a sensible default)

Everything else — navigation, carry retry, SharedMemory rendezvous, LLM
polling, action validation, MATRX feasibility checks, and task injection —
is handled automatically by this class.

Typical subclass structure
--------------------------

    class MyAgent(LLMAgentBase):
        def __init__(self, ...):
            super().__init__(...)
            # Agent-specific setup only

        def decide_on_actions(self, filtered_state):
            self._tick_setup(filtered_state)       # state + perception
            if not self._current_task:
                return self._idle()
            result = self._run_preamble(filtered_state)  # infra checks
            if result is not None:
                return result
            # ── My agent-specific reasoning ──
            ...
            return self._idle()
"""

import json
import logging
import re
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from actions1.CustomActions import CarryObject as _CarryObject
from actions1.CustomActions import CarryObjectTogether as _CarryObjectTogether
from actions1.CustomActions import Idle as _Idle
from actions1.CustomActions import RemoveObjectTogether as _RemoveObjectTogether
from matrx.actions.object_actions import RemoveObject as _RemoveObject

from agents1.action_mapper import ActionMapper
from agents1.async_model_prompting import get_llm_result, submit_llm_call
from agents1.modules.communication_module import CommunicationModule
from agents1.modules.execution_module import execute_action
from agents1.modules.perception_module import Perception
from agents1.modules.planning_module import Planning
from brains1.ArtificialBrain import ArtificialBrain
from memory.base_memory import BaseMemory
from memory.shared_memory import SharedMemory

logger = logging.getLogger('LLMAgentBase')

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_NR_TOKENS: int = 3000
TEMPERATURE: float = 0.3
CARRY_WAIT_TIMEOUT_TICKS: int = 100


# ── Base class ─────────────────────────────────────────────────────────────────

class LLMAgentBase(ArtificialBrain, Perception):
    """Infrastructure base class for LLM-driven rescue agents.

    Args:
        slowdown:       ArtificialBrain tick slow-down factor.
        condition:      World condition ('normal' | 'strong' | 'weak').
        name:           Cooperative partner name (human or AI agent ID).
        folder:         Working folder path (passed to ArtificialBrain).
        llm_model:      LiteLLM model string, e.g. ``'ollama/llama3'``.
        include_human:  Whether to include the human in the observation filter.
        shared_memory:  Optional SharedMemory instance for cross-agent sharing.
        planning_mode:  Planning strategy: ``'simple'`` (flat list) or
                        ``'dag'`` (task graph with conditional branching).
    """

    # ── Actions that require a valid, in-range object_id ──────────────────
    _OBJECT_ACTIONS = frozenset({
        'CarryObject', 'CarryObjectTogether',
        'RemoveObject', 'RemoveObjectTogether',
    })

    # ── Actions skipped for MATRX feasibility check ───────────────────────
    _SKIP_MATRX_CHECK = frozenset({
        'Idle', 'MoveTo', 'NavigateToDropZone',
        'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest', 'SendMessage',
    })

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'ollama/llama3',
        include_human: bool = True,
        shared_memory: Optional[SharedMemory] = None,
        planning_mode: str = 'simple',
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        comm_strategy: str = 'always_respond',
    ) -> None:
        super().__init__(slowdown, condition, name, folder)

        self._llm_model = llm_model
        self._api_base = api_base
        self._include_human = include_human
        self._partner_name = name
        self.teammates: set = set()

        # ── Capabilities ──────────────────────────────────────────────────
        self._capabilities = capabilities
        self._capability_knowledge = capability_knowledge

        # ── Communication ─────────────────────────────────────────────────
        self._comm_strategy = comm_strategy

        # ── Memory ─────────────────────────────────────────────────────────
        self.memory = BaseMemory()
        self.shared_memory: Optional[SharedMemory] = shared_memory

        # ── Planning ───────────────────────────────────────────────────────
        self.planner = Planning(mode=planning_mode)
        self.task_num: int = 0

        # ── Navigation ─────────────────────────────────────────────────────
        self._state_tracker: Optional[StateTracker] = None
        self._navigator: Optional[Navigator] = None
        self.state_for_navigation: Optional[State] = None
        self._nav_target: Optional[Tuple[int, int]] = None

        # ── Async LLM state ────────────────────────────────────────────────
        self._pending_future: Optional[Future] = None
        self._reasoning_step: bool = True

        # ── Task ───────────────────────────────────────────────────────────
        self._current_task: Optional[str] = None

        # ── Action feedback ────────────────────────────────────────────────
        self._action_feedback: str = ''

        # ── Cooperative carry state ────────────────────────────────────────
        self._pending_carry_kwargs: Optional[dict] = None
        self._carry_wait_ticks: int = 0

        # ── Text-JSON fallback parser ──────────────────────────────────────
        self._mapper = ActionMapper(partner_name=name)

        # ── World state (populated each tick by _tick_setup) ───────────────
        self.WORLD_STATE: Dict = {}

    # ── MATRX lifecycle ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Called once before the simulation starts."""
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM,
        )
        self.init_global_state()
        self.comm = CommunicationModule(
            agent_id=self.agent_id,
            strategy=self._comm_strategy,
            llm_model=self._llm_model,
            api_base=self._api_base,
        )
        logger.info('[%s] LLMAgentBase ready (model=%s)', self.agent_id, self._llm_model)

    # ── Default perception filter ─────────────────────────────────────────

    def filter_observations(self, state: State) -> State:
        """Restrict observations to 1-block Chebyshev radius + doors + self.

        Saves the full unfiltered state to ``state_for_navigation`` for A*.
        Subclasses may override this for custom perception ranges.
        """
        agent_loc = state[self.agent_id]['location']
        self.state_for_navigation = state.copy()
        filtered = state.copy()
        self.teammates = set()

        keep = {self.agent_id, 'World'}
        if self._include_human:
            keep.add(self._partner_name)
            self.teammates.add((
                self._partner_name,
                state.get(self._partner_name, {}).get('location', [0, 0]),
            ))

        for obj_id, obj_data in filtered.items():
            if obj_id in keep:
                continue
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot'):
                keep.add(obj_id)
                self.teammates.add((obj_id, state.get(obj_id, {}).get('location', [0, 0])))
            if self._is_within_range(agent_loc, obj_data.get('location'), radius=1):
                keep.add(obj_id)
            if 'door' in str(obj_id).lower():
                keep.add(obj_id)

        for obj_id in list(filtered.keys()):
            if obj_id not in keep:
                filtered.remove(obj_id)

        return filtered

    # ── Task injection ────────────────────────────────────────────────────

    def set_current_task(self, task: str) -> None:
        """Inject a high-level task from the EnginePlanner.

        Resets navigation and pending LLM state. Per-agent memory is
        preserved across task transitions.
        """
        if not isinstance(task, str):
            task = json.dumps(task, default=str)
        self._current_task = task
        self._pending_future = None
        self._nav_target = None
        self._reasoning_step = True
        self._pending_carry_kwargs = None
        self._carry_wait_ticks = 0
        self.planner.update_current_task(task)
        print(f'[{self.agent_id}] Task: {task}')

    def set_manual_task_decomposition(self, decomposition: str) -> None:
        """Inject a manual plan (numbered list, possibly with sub-bullets).

        Must be called after ``set_current_task()``.
        """
        lines = decomposition.strip().split('\n')
        entries: list[str] = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'\d+\.', stripped):
                parts = stripped.split('. ', 1)
                entries.append(parts[1] if len(parts) > 1 else stripped)
            elif stripped.startswith('-') and entries:
                entries[-1] += '\n' + line   # merge sub-bullet into parent
        self.planner.set_manual_task_decomposition(entries)
        self.task_num = len(entries)

    # ── Per-tick infrastructure entry-points ──────────────────────────────

    def _tick_setup(self, filtered_state: State) -> None:
        """Update state tracker and perception. Call at the top of ``decide_on_actions``."""
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(
            filtered_state, agent_id=self.agent_id, teammates=self.teammates
        )
        self.process_observations(filtered_state)
        self.comm.process_messages(self.received_messages)

    def _run_preamble(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Handle all infrastructure concerns before the agent's reasoning step.

        Checks (in order):
            1. Cooperative carry retry loop
            2. Ongoing A* navigation
            3. SharedMemory rendezvous navigation
            4. Pending LLM future poll

        Returns an ``(action_name, kwargs)`` tuple if infrastructure needs to
        act this tick, or ``None`` if the agent should proceed with its own
        reasoning.
        """
        carry = self._handle_carry_retry()
        if carry is not None:
            return carry

        nav = self._handle_navigation_tick()
        if nav is not None:
            return nav

        rendezvous = self._handle_rendezvous()
        if rendezvous is not None:
            return rendezvous

        poll = self._poll_llm_future(filtered_state)
        if poll is not None:
            return poll

        return None

    # ── Infrastructure internals ──────────────────────────────────────────

    def _handle_carry_retry(self) -> Optional[Tuple[str, Dict]]:
        """Retry CarryObjectTogether until partner arrives or timeout."""
        if self._pending_carry_kwargs is None:
            return None

        obj_id = self._pending_carry_kwargs.get('object_id', '')
        nearby_ids = (
            {o['id'] for o in self.WORLD_STATE.get('nearby', [])}
            if isinstance(self.WORLD_STATE, dict) else set()
        )

        if obj_id not in nearby_ids:
            print(f"[{self.agent_id}] Carry complete — victim '{obj_id}' delivered")
            self._pending_carry_kwargs = None
            self._carry_wait_ticks = 0
            if self.shared_memory:
                self.shared_memory.update('carry_rendezvous', None)
            self._reasoning_step = True
            return self._idle()

        self._carry_wait_ticks += 1
        if self._carry_wait_ticks > CARRY_WAIT_TIMEOUT_TICKS:
            print(f'[{self.agent_id}] Carry timeout after {CARRY_WAIT_TIMEOUT_TICKS} ticks')
            self.memory.update('carry_failure', {
                'victim_id': obj_id,
                'reason': 'partner_timeout',
                'ticks_waited': CARRY_WAIT_TIMEOUT_TICKS,
            })
            self._action_feedback = (
                f"CarryObjectTogether for victim '{obj_id}' failed: "
                f"partner did not arrive within {CARRY_WAIT_TIMEOUT_TICKS} ticks."
            )
            self._pending_carry_kwargs = None
            self._carry_wait_ticks = 0
            if self.shared_memory:
                self.shared_memory.update('carry_rendezvous', None)
            self._reasoning_step = True
            return self._idle()

        if self.shared_memory:
            agent_loc = (
                self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                if isinstance(self.WORLD_STATE, dict) else [0, 0]
            )
            self.shared_memory.update('carry_rendezvous', {
                'agent': self.agent_id,
                'victim_id': obj_id,
                'location': agent_loc,
                'status': 'waiting_for_partner',
            })
        print(f'[{self.agent_id}] Carry retry {self._carry_wait_ticks}/{CARRY_WAIT_TIMEOUT_TICKS}')
        return _CarryObjectTogether.__name__, dict(self._pending_carry_kwargs)

    def _handle_navigation_tick(self) -> Optional[Tuple[str, Dict]]:
        """Continue A* navigation if a target is set."""
        if self._nav_target is None:
            return None
        move = self._navigator.get_move_action(self._state_tracker)
        if move is not None:
            return move, {}
        # Destination reached
        self._nav_target = None
        self._reasoning_step = True
        return None

    def _handle_rendezvous(self) -> Optional[Tuple[str, Dict]]:
        """Navigate to a partner's carry rendezvous if one is published."""
        if not self.shared_memory:
            return None
        rendezvous = self.shared_memory.retrieve('carry_rendezvous')
        if not (
            rendezvous
            and rendezvous.get('agent') != self.agent_id
            and rendezvous.get('status') == 'waiting_for_partner'
        ):
            return None

        target = tuple(rendezvous['location'])
        my_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
            if isinstance(self.WORLD_STATE, dict) else [0, 0]
        )
        if self._is_within_range(tuple(my_loc), target, radius=1):
            return None

        self._navigator.reset_full()
        self._navigator.add_waypoints([target])
        self._nav_target = target
        self._pending_future = None
        move = self._navigator.get_move_action(self._state_tracker)
        print(f'[{self.agent_id}] Navigating to carry rendezvous at {target}')
        return (move, {}) if move else self._idle()

    def _poll_llm_future(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Poll the async LLM future. Returns result action or Idle if pending."""
        if self._pending_future is None:
            return None
        try:
            result = get_llm_result(self._pending_future)
        except Exception as exc:
            logger.warning('[%s] LLM future raised: %s', self.agent_id, exc)
            self._pending_future = None
            self._reasoning_step = True
            return self._idle()

        if result is None:
            return self._idle()   # still waiting
        return self._handle_llm_result(filtered_state, result)

    def _handle_llm_result(
        self, filtered_state: State, result: List
    ) -> Tuple[str, Dict]:
        """Dispatch LLM response to a MATRX action (tool_call or text fallback)."""
        message = result[0]
        self._pending_future = None

        # Resolve partner
        partner = next(
            (i[0] for i in self.teammates if i[0] != self.agent_id), None
        )

        # ── Path A: structured tool_call ──────────────────────────────────
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            tc = tool_calls[0]
            name = tc.function.name
            args_raw = tc.function.arguments
            args: Dict[str, Any] = (
                json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            )
            print(f'[{self.agent_id}] Tool call: {name}({args})')
            self.send_message(Message(
                content=f'Executing action {name}({args})', from_id=self.agent_id
            ))

            if (check := self._validate_action(name, args)) is not None:
                return check
            action_name, kwargs = execute_action(name, args, partner, self.agent_id)
            if (check := self._check_matrx_action(action_name, kwargs)) is not None:
                return check

            self.planner.advance_task(action_name)
            self.memory.update('action', {'action': action_name, 'args': kwargs})
            self._maybe_share_observation(filtered_state, action_name, kwargs)
            comm_result = self._apply_communication(action_name, kwargs)
            if comm_result is not None:
                return comm_result
            return self._apply_navigation(action_name, kwargs)

        # ── Path B: plain-text JSON fallback ──────────────────────────────
        llm_text = getattr(message, 'content', '') or ''
        print(f'[{self.agent_id}] Text response: {llm_text[:120]}')

        raw_name, raw_args = self._mapper.parse_raw(llm_text)
        if raw_name is None:
            self._reasoning_step = True
            return self._idle()

        if (check := self._validate_action(raw_name, raw_args)) is not None:
            return check
        action_name, kwargs = execute_action(raw_name, raw_args, partner, self.agent_id)
        if (check := self._check_matrx_action(action_name, kwargs)) is not None:
            return check

        self.planner.advance_task(action_name)
        self.memory.update('action', {'action': action_name, 'args': kwargs})
        self._maybe_share_observation(filtered_state, action_name, kwargs)
        comm_result = self._apply_communication(action_name, kwargs)
        if comm_result is not None:
            return comm_result
        return self._apply_navigation(action_name, kwargs)

    def _apply_navigation(
        self, action_name: str, kwargs: Dict[str, Any]
    ) -> Tuple[str, Dict]:
        """Set up A* navigation for MoveTo / NavigateToDropZone.
        All other actions are passed through unchanged.
        """
        self._reasoning_step = True

        if action_name == 'MoveTo':
            coords = (int(kwargs.get('x', 0)), int(kwargs.get('y', 0)))
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()

        if action_name == 'NavigateToDropZone':
            coords = (23, 8)
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()

        if action_name == _CarryObjectTogether.__name__:
            self._pending_carry_kwargs = dict(kwargs)
            self._carry_wait_ticks = 0
            self._reasoning_step = False
            return action_name, kwargs

        return action_name, kwargs

    def _apply_communication(
        self, action_name: str, kwargs: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Handle SendMessage actions. Returns Idle if sent, None if not SendMessage."""
        if action_name != 'SendMessage':
            return None

        send_to = kwargs.get('send_to', '')
        message_type = kwargs.get('message_type', 'message')
        text = kwargs.get('message', '')

        # Build structured content and send via MATRX
        target = None if send_to == 'all' else send_to
        content = {'message_type': message_type, 'text': text}
        msg = Message(content=content, from_id=self.agent_id, to_id=target)
        self.send_message(msg)

        # Auto-announce: if private 'help' reply to someone who asked, broadcast
        if message_type == 'help' and target is not None:
            if self.comm.has_pending_ask_help(from_agent=send_to):
                ann = {
                    'message_type': 'message',
                    'text': f'{self.agent_id} is responding to {send_to} help request',
                }
                self.send_message(Message(content=ann, from_id=self.agent_id, to_id=None))

        self._reasoning_step = True
        return self._idle()

    # ── Action validation ─────────────────────────────────────────────────

    def _validate_action(
        self, name: str, args: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Validate object_id against the current WORLD_STATE before dispatching.

        Returns ``None`` when valid. Returns an Idle tuple on failure and
        populates ``_action_feedback`` for the next LLM prompt.
        """
        if name not in self._OBJECT_ACTIONS:
            return None

        obj_id = args.get('object_id', '')
        nearby = (
            self.WORLD_STATE.get('nearby', [])
            if isinstance(self.WORLD_STATE, dict) else []
        )
        actionable_types = {'victim', 'rock', 'stone', 'tree'}
        nearby_summary = ', '.join(
            f"{o['id']} ({o['type']}"
            + (f", {o.get('severity')}" if o.get('severity') else '')
            + f" at {o['location']})"
            for o in nearby if o.get('type') in actionable_types
        ) or 'none'
        agent_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )

        if not obj_id:
            self._action_feedback = (
                f'Action {name} requires an object_id but none was provided. '
                f'Nearby actionable objects: [{nearby_summary}]. '
                f'Agent location: {agent_loc}.'
            )
            self._reasoning_step = True
            logger.warning('[%s] %s', self.agent_id, self._action_feedback)
            return self._idle()

        nearby_ids = {o['id'] for o in nearby}
        if obj_id not in nearby_ids:
            self._action_feedback = (
                f"Action {name} failed: object '{obj_id}' is not within reach "
                f'(1-block range) or does not exist. '
                f'Nearby actionable objects: [{nearby_summary}]. '
                f'Agent location: {agent_loc}. '
                f'Move closer to the target or choose a different object.'
            )
            self._reasoning_step = True
            logger.warning('[%s] %s', self.agent_id, self._action_feedback)
            return self._idle()

        self._action_feedback = ''
        return None

    def _check_matrx_action(
        self, action_name: str, kwargs: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Ask MATRX whether the action is feasible before dispatching.

        Returns ``None`` if possible. Returns an Idle tuple and populates
        ``_action_feedback`` if MATRX rejects it.
        """
        if action_name in self._SKIP_MATRX_CHECK:
            return None

        check_kwargs = dict(kwargs)
        check_kwargs.setdefault('grab_range', 1)
        check_kwargs.setdefault('max_objects', 1)

        succeeded, action_result = self.is_action_possible(action_name, check_kwargs)
        if succeeded:
            return None

        nearby = (
            self.WORLD_STATE.get('nearby', [])
            if isinstance(self.WORLD_STATE, dict) else []
        )
        actionable_types = {'victim', 'rock', 'stone', 'tree'}
        nearby_summary = ', '.join(
            f"{o['id']} ({o['type']}"
            + (f", {o.get('severity')}" if o.get('severity') else '')
            + f" at {o['location']})"
            for o in nearby if o.get('type') in actionable_types
        ) or 'none'
        agent_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )
        self._action_feedback = (
            f'MATRX rejected action {action_name}: {action_result.result} '
            f'Nearby actionable objects: [{nearby_summary}]. '
            f'Agent location: {agent_loc}.'
        )
        self._reasoning_step = True
        logger.warning('[%s] %s', self.agent_id, self._action_feedback)
        return self._idle()

    # ── Shared memory publishing ──────────────────────────────────────────

    def _maybe_share_observation(
        self, state: State, action_name: str, kwargs: Dict[str, Any]
    ) -> None:
        """Publish carry / obstacle events to SharedMemory for other agents."""
        if self.shared_memory is None:
            return
        agent_loc = list(state.get(self.agent_id, {}).get('location', [0, 0]))

        if action_name in (_CarryObject.__name__, _CarryObjectTogether.__name__):
            obj_id = kwargs.get('object_id', '')
            if obj_id:
                self.shared_memory.update(f'victim_{obj_id}', {
                    'agent': self.agent_id,
                    'victim_id': obj_id,
                    'location': agent_loc,
                    'action': action_name,
                })

        if action_name in (_RemoveObject.__name__, _RemoveObjectTogether.__name__):
            obj_id = kwargs.get('object_id', '')
            if obj_id:
                self.shared_memory.update(f'obstacle_{obj_id}', {
                    'agent': self.agent_id,
                    'obstacle_id': obj_id,
                    'location': agent_loc,
                    'action': action_name,
                })

    # ── LLM submission ────────────────────────────────────────────────────

    def _submit_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: str = 'auto',
    ) -> None:
        """Submit an async LLM call. The result is retrieved by ``_poll_llm_future``."""
        self._pending_future = submit_llm_call(
            llm_model=self._llm_model,
            messages=messages,
            max_token_num=MAX_NR_TOKENS,
            temperature=TEMPERATURE,
            tools=tools,
            tool_choice=tool_choice if tools else 'none',
            api_base=self._api_base,
        )
        self._reasoning_step = False

    # ── Messaging ─────────────────────────────────────────────────────────

    def _send_message(
        self, content: str, sender: str, target_id: Optional[str] = None
    ) -> None:
        """Send a MATRX message to teammates (deduplicates before sending)."""
        msg = Message(content=content, from_id=sender, to_id=target_id)
        if content not in [m.content for m in self.messages_to_send]:
            self.send_message(msg)

    # ── Utilities ─────────────────────────────────────────────────────────

    def _idle(self) -> Tuple[str, Dict]:
        """Convenience: return an Idle action."""
        return _Idle.__name__, {'duration_in_ticks': 1}

    def _is_within_range(
        self,
        pos1: Optional[Tuple[int, int]],
        pos2: Optional[Tuple[int, int]],
        radius: int,
    ) -> bool:
        """Chebyshev distance check."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius
