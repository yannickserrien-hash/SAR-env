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
from agents1.modules.execution_module import execute_action
from agents1.modules.message_handler import MessageHandler, MessageRecord
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
AREAS_CONFIG = [
        # World Bounds
        {"id": "world_bounds", "pos": (0, 0), "w": 25, "h": 24, "door": None, "mat": None},
        
        # Row 1
        {"id": 1, "pos": (1, 1), "w": 5, "h": 4, "door": (3, 4), "mat": (3, 5), "enter": "North"},
        {"id": 2, "pos": (7, 1), "w": 5, "h": 4, "door": (9, 4), "mat": (9, 5), "enter": "North"},
        {"id": 3, "pos": (13, 1), "w": 5, "h": 4, "door": (15, 4), "mat": (15, 5), "enter": "North"},
        {"id": 4, "pos": (19, 1), "w": 5, "h": 4, "door": (21, 4), "mat": (21, 5), "enter": "North"},
        
        # Row 2
        {"id": 5, "pos": (1, 7), "w": 5, "h": 4, "door": (3, 7), "mat": (3, 6), "enter": "South"},
        {"id": 6, "pos": (7, 7), "w": 5, "h": 4, "door": (9, 7), "mat": (9, 6), "enter": "South"},
        {"id": 7, "pos": (13, 7), "w": 5, "h": 4, "door": (15, 7), "mat": (15, 6), "enter": "South"},
        
        # Row 3 (Previously Commented Out)
        {"id": 8, "pos": (1, 13), "w": 5, "h": 4, "door": (3, 16), "mat": (3, 17), "enter": "North"},
        {"id": 9, "pos": (7, 13), "w": 5, "h": 4, "door": (9, 16), "mat": (9, 17), "enter": "North"},
        {"id": 10, "pos": (13, 13), "w": 5, "h": 4, "door": (15, 16), "mat": (15, 17), "enter": "North"},
        
        # Row 4 (Previously Commented Out)
        {"id": 11, "pos": (1, 19), "w": 5, "h": 4, "door": (3, 19), "mat": (3, 18), "enter": "South"},
        {"id": 12, "pos": (7, 19), "w": 5, "h": 4, "door": (9, 19), "mat": (9, 18), "enter": "South"},
        {"id": 13, "pos": (13, 19), "w": 5, "h": 4, "door": (15, 19), "mat": (15, 18), "enter": "South"},
        {"id": 14, "pos": (19, 19), "w": 5, "h": 4, "door": (21, 19), "mat": (21, 18), "enter": "South"}
    ]

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
        'Idle', 'MoveTo', 'NavigateToDropZone', 'Drop'
        'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest', 'SendMessage',
        _CarryObjectTogether.__name__,   # managed by the carry retry loop
    })
    
    _OBJECT_TYPES = frozenset({"victim", "tree", "rock", "stone"})

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
    ) -> None:
        super().__init__(slowdown, condition, name, folder)

        self._llm_model = llm_model
        self._api_base = api_base
        self._include_human = include_human
        self._partner_name = name
        self.teammates: set = set()

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

        # ── Cooperative carry state ────────────────────────────────────────
        self._pending_carry_kwargs: Optional[dict] = None
        self._carry_wait_ticks: int = 0

        # ── Text-JSON fallback parser ──────────────────────────────────────
        self._mapper = ActionMapper(partner_name=name)

        # ── World state (populated each tick by _tick_setup) ───────────────
        self.WORLD_STATE: Dict = {}

        # ── Communication ─────────────────────────────────────────────────
        self._msg_handler = MessageHandler()
        self._comm_strategy: str = 'priority'  # 'priority' or 'scheduled'
        self._current_tick: int = 0

        # Strategy 1 (priority): async reply generation
        self._priority_reply_future: Optional[Future] = None
        self._priority_reply_target: Optional[str] = None

        # Strategy 2 (scheduled): busyness-based scheduling
        self._deferred_message: Optional[MessageRecord] = None
        self._pending_reply_tick: Optional[int] = None
        self._in_conversation_with: Optional[str] = None

    # ── MATRX lifecycle ───────────────────────────────────────────────────

    def initialize(self) -> None:
        """Called once before the simulation starts."""
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM,
        )
        self._msg_handler.agent_id = self.agent_id
        self.init_global_state()
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

    def process_observations(self, filtered_state: State) -> None:
        """Update state tracker and perception. Call at the top of ``decide_on_actions``."""
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(
            filtered_state, agent_id=self.agent_id, teammates=self.teammates
        )
        # Track tick count for communication scheduling
        world_data = filtered_state.get('World', {})
        self._current_tick = world_data.get('nr_ticks', self._current_tick + 1)
        self.update_observation(filtered_state)
        

    def _run_preamble(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Handle all infrastructure concerns before the agent's reasoning step.

        Checks (in order):
            1. Cooperative carry retry loop
            2. Ongoing A* navigation
            3. SharedMemory rendezvous navigation
            4. Incoming message handling (strategy-dependent)
            5. Pending LLM future poll

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

        # Communication: handle incoming messages based on strategy
        if self._comm_strategy == 'priority':
            comm = self._handle_priority_reply()
        else:
            comm = self._handle_scheduled_reply()
        if comm is not None:
            return comm

        poll = self.check_if_llm_response_ready(filtered_state)
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

            self.memory.update('action_failure', {
                'victim_id': obj_id,
                'action': "Carry",
                'ticks_waited': CARRY_WAIT_TIMEOUT_TICKS,
                'feedback': (
                f"CarryObjectTogether for victim '{obj_id}' failed: "
                f"partner did not arrive within {CARRY_WAIT_TIMEOUT_TICKS} ticks."
            )
            })
            
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

    def check_if_llm_response_ready(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
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
        return self._handle_llm_result(result)

    def _handle_llm_result(
        self, result: List
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
            action_name, kwargs, task_completing = execute_action(name, args, partner, self.agent_id)
        else:
            # ── Path B: plain-text JSON fallback ──────────────────────────────
            llm_text = getattr(message, 'content', '') or ''
            print(f'[{self.agent_id}] Text response: {llm_text[:120]}')

            raw_name, raw_args = self._mapper.parse_raw(llm_text)
            if raw_name is None:
                self._reasoning_step = True
                return self._idle()

            if (check := self._validate_action(raw_name, raw_args)) is not None:
                return check
            action_name, kwargs, task_completing = execute_action(raw_name, raw_args, partner, self.agent_id)

        self.planner.advance_task(task_completing)
        self.memory.update('action', {'action': action_name, 'args': kwargs})
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
        
        if action_name == 'MoveToArea':
            target = (int(kwargs.get('area', 0)))
            self._navigator.reset_full()
            door = next((area["door"] for area in AREAS_CONFIG if area["id"] == target), None)
            if door == None:
                self.memory.update("action_failure", "Area {area} does not exist. Try a different one.")
                self._reasoning_step = True
                return self._idle()
            print("Door:")
            print(door)
            self._navigator.add_waypoints([door])
            self._nav_target = door
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()
        
        if action_name == 'EnterArea':
            target = (int(kwargs.get('area', 0)))
            direction = next((area["enter"] for area in AREAS_CONFIG if area["id"] == target), None)
            if direction == 'North':
                return ('MoveNorth', {})
            if direction == 'South':
                return ('MoveSouth', {})

        if action_name == 'NavigateToDropZone':
            coords = (23, 8)
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()

        if action_name == 'SendMessage':
            send_to = kwargs.get('send_to', '')
            tag = kwargs.get('tag', 'share_info')
            raw_message = kwargs.get('message', '')
            content = f"[tag:{tag}] {raw_message}"
            target = None if 'all' in send_to else send_to
            self._send_message(content=content, sender=self.agent_id, target_id=target)
            self.memory.update('sent_message', {
                'to': send_to, 'tag': tag, 'content': raw_message,
            })
            self._reasoning_step = True
            return self._idle()

        if action_name == _CarryObjectTogether.__name__:
            self._pending_carry_kwargs = dict(kwargs)
            self._carry_wait_ticks = 0
            self._reasoning_step = False
            return action_name, kwargs

        return action_name, kwargs

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
            self.WORLD_STATE.get('current_observation', [])
            if isinstance(self.WORLD_STATE, dict) else []
        )
        nearby_summary = ', '.join(
            f"{o['id']} ({o['type']}"
            + (f", {o.get('severity')}" if o.get('severity') else '')
            + f" at {o['location']})"
            for o in nearby if o.get('type') in self._OBJECT_TYPES
        ) or 'none'
        agent_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )

        if not obj_id:
            self.memory.update("action_failure", (
                f'Action {name} requires an object_id but none was provided. '
                f'Nearby actionable objects: [{nearby_summary}]. '
            ))

            self._reasoning_step = True
            return self._idle()

        nearby_ids = {o['id'] for o in nearby}
        if obj_id not in nearby_ids:
            self.memory.update("action_failure", (
                f"Action {name} failed: object '{obj_id}' is not within reach "
                f'Nearby actionable objects: [{nearby_summary}]. '
                f'Move closer to the target or choose a different object.'
            ))
            self._reasoning_step = True
            return self._idle()
        return None

    # ── LLM submission ────────────────────────────────────────────────────

    def _submit_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: str = 'auto',
    ) -> None:
        """Submit an async LLM call. The result is retrieved by ``check_if_llm_response_ready``."""
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

    # ── Communication strategies ──────────────────────────────────────────

    def _handle_priority_reply(self) -> Optional[Tuple[str, Dict]]:
        """Strategy 1: Private messages trigger a priority async LLM reply.

        Broadcasts are NOT interrupted — they are absorbed into the
        communication context that ``get_context_for_prompt()`` provides.
        """
        # Poll an in-flight reply
        if self._priority_reply_future is not None:
            result = get_llm_result(self._priority_reply_future)
            if result is None:
                return self._idle()  # still generating reply
            reply_text = getattr(result[0], 'content', '') or ''
            self._send_message(
                content=f'[tag:reply] {reply_text}',
                sender=self.agent_id,
                target_id=self._priority_reply_target,
            )
            self.memory.update('sent_reply', {
                'to': self._priority_reply_target, 'content': reply_text,
            })
            self._priority_reply_future = None
            self._priority_reply_target = None
            self._reasoning_step = True
            return self._idle()

        # Check for new unprocessed private messages
        new_privates = self._msg_handler.get_unprocessed_private()
        if not new_privates:
            return None

        msg = new_privates[0]
        msg.processed = True
        return self._submit_reply_llm_call(msg)

    def _handle_scheduled_reply(self) -> Optional[Tuple[str, Dict]]:
        """Strategy 2: Event-driven message scheduling based on busyness.

        Scheduling rules:
            - Heavy action + non-help tag → drop message
            - Light action → defer by 2 ticks
            - Idle + private → defer by 2 ticks
            - Heavy + ask_help → defer by 1 tick
            - Already in conversation with someone else → reply "busy"
        """
        # Poll an in-flight reply
        if self._priority_reply_future is not None:
            result = get_llm_result(self._priority_reply_future)
            if result is None:
                return self._idle()
            reply_text = getattr(result[0], 'content', '') or ''
            self._send_message(
                content=f'[tag:reply] {reply_text}',
                sender=self.agent_id,
                target_id=self._priority_reply_target,
            )
            self._priority_reply_future = None
            self._priority_reply_target = None
            self._in_conversation_with = None
            self._reasoning_step = True
            return self._idle()

        # Check if a deferred reply is due
        if (
            self._deferred_message is not None
            and self._pending_reply_tick is not None
            and self._current_tick >= self._pending_reply_tick
        ):
            msg = self._deferred_message
            self._deferred_message = None
            self._pending_reply_tick = None
            return self._submit_reply_llm_call(msg)

        # Process new messages
        self._msg_handler.parse_new_messages(self.received_messages)
        for msg in self._msg_handler.get_unprocessed():
            if msg.from_id == self.agent_id:
                msg.processed = True
                continue

            # Already in conversation with someone else
            if (
                self._in_conversation_with is not None
                and msg.from_id != self._in_conversation_with
                and msg.is_private
            ):
                self._send_message(
                    content='[tag:reply] I\'m busy with another task, try again later.',
                    sender=self.agent_id,
                    target_id=msg.from_id,
                )
                msg.processed = True
                continue

            busyness = self._classify_busyness()

            if busyness == 'heavy' and msg.tag != 'ask_help':
                msg.processed = True  # drop
                continue
            elif busyness == 'heavy' and msg.tag == 'ask_help':
                self._deferred_message = msg
                self._pending_reply_tick = self._current_tick + 1
                msg.processed = True
                break
            elif busyness == 'light':
                self._deferred_message = msg
                self._pending_reply_tick = self._current_tick + 2
                msg.processed = True
                break
            elif busyness == 'idle' and msg.is_private:
                self._deferred_message = msg
                self._pending_reply_tick = self._current_tick + 2
                msg.processed = True
                break
            else:
                msg.processed = True  # broadcast while idle — absorbed into context

        return None

    def _classify_busyness(self) -> str:
        """Determine current busyness level for Strategy 2 scheduling."""
        if self._pending_carry_kwargs is not None:
            return 'heavy'
        if self._nav_target is not None:
            return 'light'
        if self._pending_future is not None:
            return 'light'
        return 'idle'

    def _submit_reply_llm_call(self, msg: MessageRecord) -> Tuple[str, Dict]:
        """Submit an async LLM call to generate a reply to a teammate message."""
        agent_pos = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )
        reply_prompt = [
            {'role': 'system', 'content': (
                'You received a message from a teammate in a search and rescue mission. '
                'Write a brief, helpful reply (1-2 sentences). '
                'If they ask for help, say whether you can help based on your current task.'
            )},
            {'role': 'user', 'content': (
                f'FROM: {msg.from_id}\n'
                f'TAG: {msg.tag}\n'
                f'MESSAGE: {msg.content}\n'
                f'YOUR CURRENT TASK: {self._current_task}\n'
                f'YOUR POSITION: {agent_pos}'
            )},
        ]
        self._priority_reply_future = submit_llm_call(
            llm_model=self._llm_model,
            messages=reply_prompt,
            max_token_num=200,
            temperature=TEMPERATURE,
            api_base=self._api_base,
        )
        self._priority_reply_target = msg.from_id
        self._in_conversation_with = msg.from_id
        return self._idle()

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
