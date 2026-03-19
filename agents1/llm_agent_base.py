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
from matrx.actions.object_actions import DropObject as _DropObject
from matrx.actions.object_actions import RemoveObject as _RemoveObject

from agents1.action_mapper import ActionMapper
from agents1.async_model_prompting import get_llm_result, submit_llm_call
from agents1.modules.communication_module import CommunicationModule
from agents1.modules.execution_module import execute_action
from agents1.modules.perception_module import Perception
from agents1.modules.planning_module import Planning
from agents1.modules.validator_module import ActionValidator
from brains1.ArtificialBrain import ArtificialBrain
from memory.base_memory import BaseMemory
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation

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

    # ── Actions skipped for MATRX feasibility check ───────────────────────
    _SKIP_MATRX_CHECK = frozenset({
        'Idle', 'MoveTo', 'NavigateToDropZone', 'Drop',
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
        env_info: Optional[EnvironmentInformation] = None,
    ) -> None:
        super().__init__(slowdown, condition, name, folder)

        self._llm_model = llm_model
        self._api_base = api_base
        self._include_human = include_human
        self._partner_name = name
        self.teammates: set = set()

        # ── Environment info ─────────────────────────────────────────────
        self.env_info: EnvironmentInformation = env_info or EnvironmentInformation()
        # Update Perception mixin's drop zone from env_info
        Perception.set_drop_zone(self.env_info.drop_zone)

        # ── Capabilities ──────────────────────────────────────────────────
        self._capabilities = capabilities
        self._capability_knowledge = capability_knowledge

        # ── Action validator ──────────────────────────────────────────────
        self._validator = ActionValidator(
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
        )

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
        self._carry_autopilot: Optional[Dict] = None  # {'victim_id', 'destination', 'role'}
        self._negotiation_ticks: int = 0  # ticks since conflicting help requests detected

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

    def update_knowledge(self, filtered_state: State) -> None:
        """Update state tracker and perception. Call at the top of ``decide_on_actions``."""
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(
            filtered_state, agent_id=self.agent_id, teammates=self.teammates
        )
        self.update_state_belief(filtered_state)
        self.comm.process_messages(self.received_messages)

    def _run_infra(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Handle infrastructure concerns only (no LLM polling).

        Checks (in order):
            1. Carry autopilot (navigating to drop zone after successful carry)
            2. Cooperative carry retry loop
            3. Ongoing A* navigation
            4. SharedMemory rendezvous / help request navigation

        Returns an ``(action_name, kwargs)`` tuple if infrastructure needs to
        act this tick, or ``None`` if the agent should proceed.
        """
        self._check_carry_notifications()

        autopilot = self._handle_carry_autopilot()
        if autopilot is not None:
            return autopilot

        carry = self._handle_carry_retry()
        if carry is not None:
            return carry

        nav = self._handle_navigation_tick()
        if nav is not None:
            return nav

        rendezvous = self._handle_rendezvous()
        if rendezvous is not None:
            return rendezvous

        return None

    def _run_preamble(self, filtered_state: State) -> Optional[Tuple[str, Dict]]:
        """Handle all infrastructure concerns before the agent's reasoning step.

        Checks (in order):
            1. Infrastructure (carry, navigation, rendezvous)
            2. Pending LLM future poll

        Returns an ``(action_name, kwargs)`` tuple if infrastructure needs to
        act this tick, or ``None`` if the agent should proceed with its own
        reasoning.
        """
        infra = self._run_infra(filtered_state)
        if infra is not None:
            return infra

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
            {o['id'] for o in self.WORLD_STATE.get('victims', []) + self.WORLD_STATE.get('obstacles', [])}
            if isinstance(self.WORLD_STATE, dict) else set()
        )

        if obj_id not in nearby_ids:
            # Carry action succeeded — victim is in agent's inventory.
            # Enter autopilot: navigate to drop zone, then drop.
            print(f"[{self.agent_id}] Carry grabbed — entering autopilot to drop zone")
            self._pending_carry_kwargs = None
            self._carry_wait_ticks = 0
            dest = self.env_info.drop_zone
            self._carry_autopilot = {
                'victim_id': obj_id,
                'destination': dest,
                'role': 'carrier',
            }
            if self.shared_memory:
                self.shared_memory.update('carry_rendezvous', None)
                self.shared_memory.update('carry_autopilot', {
                    'carrier': self.agent_id,
                    'victim_id': obj_id,
                    'destination': dest,
                })
                # Clear any lingering help request
                self.shared_memory.update(f'help_request_{self.agent_id}', None)
            # Start navigation to drop zone
            self._navigator.reset_full()
            self._navigator.add_waypoints([dest])
            self._nav_target = dest
            self._reasoning_step = False  # stay in infrastructure mode
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()

        self._carry_wait_ticks += 1
        if self._carry_wait_ticks > CARRY_WAIT_TIMEOUT_TICKS:
            print(f'[{self.agent_id}] Carry timeout after {CARRY_WAIT_TIMEOUT_TICKS} ticks')
            
            self.memory.update('action_failure', {
                'victim_id': obj_id,
                'action': "Carry",
                'ticks_waited': CARRY_WAIT_TIMEOUT_TICKS,
                'feedback': (
                f"CarryObjectTogether for victim '{obj_id}' failed: "
                f"partner did not arrive within {CARRY_WAIT_TIMEOUT_TICKS} ticks.")
            })
            
            self._pending_carry_kwargs = None
            self._carry_wait_ticks = 0
            if self.shared_memory:
                self.shared_memory.update('carry_rendezvous', None)
                self.shared_memory.update(f'help_request_{self.agent_id}', None)
            self._negotiation_ticks = 0
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

    def _handle_carry_autopilot(self) -> Optional[Tuple[str, Dict]]:
        """Handle autopilot navigation to drop zone after a cooperative carry.

        Both the carrier (who has the victim in inventory) and the partner
        are forced to navigate to the drop zone simultaneously.  When the
        carrier arrives it returns a DropObject action; the partner simply
        idles once it arrives.
        """
        # ── Carrier agent path ────────────────────────────────────────────
        if self._carry_autopilot and self._carry_autopilot['role'] == 'carrier':
            if self._nav_target is not None:
                return None  # let _handle_navigation_tick do the movement
            # Arrived at drop zone — drop the victim
            victim_id = self._carry_autopilot['victim_id']
            print(f'[{self.agent_id}] Autopilot: arrived at drop zone, dropping {victim_id}')
            self.memory.update('action', {
                'action': 'CarryObjectTogether',
                'result': 'delivered',
                'victim_id': victim_id,
            })
            self._carry_autopilot = None
            if self.shared_memory:
                self.shared_memory.update('carry_autopilot', None)
            self._reasoning_step = True
            return _DropObject.__name__, {'object_id': victim_id}

        if self._carry_autopilot and self._carry_autopilot['role'] == 'partner':
            # Check if carrier already finished (signal cleared)
            if self.shared_memory:
                ap = self.shared_memory.retrieve('carry_autopilot')
                if ap is None:
                    # Carrier done — clear own autopilot
                    print(f'[{self.agent_id}] Autopilot: carrier finished, resuming pipeline')
                    self._carry_autopilot = None
                    self._reasoning_step = True
                    return self._idle()
            if self._nav_target is not None:
                return None  # still navigating
            # Arrived at drop zone — wait for carrier to finish
            print(f'[{self.agent_id}] Autopilot: partner arrived at drop zone, waiting')
            self.memory.update('carry_participation', {
                'victim_id': self._carry_autopilot['victim_id'],
                'status': 'delivered',
            })
            self._carry_autopilot = None
            self._reasoning_step = True
            return self._idle()

        # ── Partner detection: check if another agent initiated autopilot ─
        if not self.shared_memory:
            return None
        ap = self.shared_memory.retrieve('carry_autopilot')
        if not ap or ap.get('carrier') == self.agent_id:
            return None

        # Another agent is carrying to drop zone — join as partner
        dest = tuple(ap['destination'])
        my_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
            if isinstance(self.WORLD_STATE, dict) else [0, 0]
        )
        if self._is_within_range(tuple(my_loc), dest, radius=1):
            # Already at drop zone
            return None

        print(f'[{self.agent_id}] Autopilot: joining partner carry to drop zone {dest}')
        self._carry_autopilot = {
            'victim_id': ap['victim_id'],
            'destination': dest,
            'role': 'partner',
        }
        self._pending_future = None  # cancel any pending LLM call
        self._reasoning_step = False
        self._navigator.reset_full()
        self._navigator.add_waypoints([dest])
        self._nav_target = dest
        # Clear own help request if any
        if self.shared_memory:
            self.shared_memory.update(f'help_request_{self.agent_id}', None)
        move = self._navigator.get_move_action(self._state_tracker)
        return (move, {}) if move else self._idle()

    def _check_carry_notifications(self) -> None:
        """Check if we were recruited into a carry and store in memory."""
        if not self.shared_memory:
            return
        key = f'carry_notification_{self.agent_id}'
        notification = self.shared_memory.retrieve(key)
        if notification:
            self.memory.update('carry_participation', notification)
            self.shared_memory.update(key, None)

    def _handle_navigation_tick(self) -> Optional[Tuple[str, Dict]]:
        """Continue A* navigation if a target is set."""
        if self._nav_target is None:
            return None
        move = self._navigator.get_move_action(self._state_tracker)
        if move is not None:
            return move, {}
        # Destination reached
        self._nav_target = None
        if not self._carry_autopilot:
            self._reasoning_step = True
        return None

    def _handle_rendezvous(self) -> Optional[Tuple[str, Dict]]:
        """Navigate to a partner's carry rendezvous or help request.

        Checks carry_rendezvous first, then help_request_* keys.
        Only auto-navigates to a help request if this agent does NOT have
        its own pending help request (prevents deadlock).
        """
        if not self.shared_memory:
            return None

        # ── Carry rendezvous (partner is retrying CarryObjectTogether) ────
        rendezvous = self.shared_memory.retrieve('carry_rendezvous')
        if (
            rendezvous
            and rendezvous.get('agent') != self.agent_id
            and rendezvous.get('status') == 'waiting_for_partner'
        ):
            target = tuple(rendezvous['location'])
            my_loc = (
                self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                if isinstance(self.WORLD_STATE, dict) else [0, 0]
            )
            if not self._is_within_range(tuple(my_loc), target, radius=1):
                self._navigator.reset_full()
                self._navigator.add_waypoints([target])
                self._nav_target = target
                self._pending_future = None
                move = self._navigator.get_move_action(self._state_tracker)
                print(f'[{self.agent_id}] Navigating to carry rendezvous at {target}')
                return (move, {}) if move else self._idle()

        # ── Help request (partner asked for help via SendMessage) ─────────
        # Only respond if we do NOT have our own pending help request
        own_request = self.shared_memory.retrieve(f'help_request_{self.agent_id}')
        if own_request:
            # Both agents need help — let LLM negotiate, don't auto-navigate.
            # Safeguard: if negotiation stalls 20+ ticks, lower ID agent yields.
            self._negotiation_ticks += 1
            if self._negotiation_ticks > 20:
                # Find the other agent's help request
                all_mem = self.shared_memory.retrieve_all()
                other_requests = [
                    (k, v) for k, v in all_mem.items()
                    if isinstance(k, str) and k.startswith('help_request_')
                    and v is not None and v.get('agent') != self.agent_id
                ]
                if other_requests and self.agent_id < other_requests[0][1]['agent']:
                    # Lower ID yields: clear own request, go help the other agent
                    print(f'[{self.agent_id}] Negotiation timeout — yielding to help other agent')
                    self.shared_memory.update(f'help_request_{self.agent_id}', None)
                    self._negotiation_ticks = 0
                    # Fall through to the help request scan below
                else:
                    return None
            else:
                return None

        # Scan for any other agent's help request
        all_mem = self.shared_memory.retrieve_all()
        for key, val in all_mem.items():
            if (
                isinstance(key, str)
                and key.startswith('help_request_')
                and val is not None
                and val.get('agent') != self.agent_id
                and val.get('status') == 'waiting'
            ):
                target = tuple(val['location'])
                my_loc = (
                    self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                    if isinstance(self.WORLD_STATE, dict) else [0, 0]
                )
                if self._is_within_range(tuple(my_loc), target, radius=1):
                    return None  # already adjacent
                self._navigator.reset_full()
                self._navigator.add_waypoints([target])
                self._nav_target = target
                self._pending_future = None
                move = self._navigator.get_move_action(self._state_tracker)
                print(f'[{self.agent_id}] Navigating to help request from {val["agent"]} at {target}')
                return (move, {}) if move else self._idle()

        return None

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
            action_name, kwargs, task_completed = execute_action(name, args, partner, self.agent_id)

        else: # ── Path B: plain-text JSON fallback ──────────────────────────────
            llm_text = getattr(message, 'content', '') or ''
            print(f'[{self.agent_id}] Text response: {llm_text[:120]}')

            raw_name, raw_args = self._mapper.parse_raw(llm_text)
            if raw_name is None:
                self._reasoning_step = True
                return self._idle()

            if (check := self._validate_action(raw_name, raw_args)) is not None:
                return check
            action_name, kwargs, task_completed = execute_action(raw_name, raw_args, partner, self.agent_id)

        self.planner.advance_task(action_name)
        self.memory.update('action', {'action': action_name, 'args': kwargs})
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
            coords = self.env_info.drop_zone
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()
        
        if action_name == 'MoveToArea':
            target = int(kwargs.get('area', 0))
            self._navigator.reset_full()
            door = self.env_info.get_door(target)
            if door is None:
                self.memory.update("action_failure", f"Area {target} does not exist. Try a different one.")
                self._reasoning_step = True
                return self._idle()
            print(f"Door: {door}")
            self._navigator.add_waypoints([door])
            self._nav_target = door
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else self._idle()

        if action_name == 'EnterArea':
            target = int(kwargs.get('area', 0))
            direction = self.env_info.get_enter_direction(target)
            if direction == 'North':
                return ('MoveNorth', {})
            if direction == 'South':
                return ('MoveSouth', {})

        if action_name == _CarryObjectTogether.__name__:
            # Teammate adjacency already validated by ActionValidator
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

        # Publish help request to SharedMemory so partner can auto-navigate
        if message_type == 'ask_help' and self.shared_memory:
            agent_loc = (
                self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                if isinstance(self.WORLD_STATE, dict) else [0, 0]
            )
            self.shared_memory.update(f'help_request_{self.agent_id}', {
                'agent': self.agent_id,
                'location': agent_loc,
                'message': text,
                'status': 'waiting',
            })
            print(f'[{self.agent_id}] Published help request to SharedMemory')

        self._reasoning_step = True
        return self._idle()

    # ── Action validation ─────────────────────────────────────────────────

    def _validate_action(
        self, name: str, args: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Validate an action before dispatching to MATRX.

        Delegates to :class:`ActionValidator` which checks every action type.
        Returns ``None`` when valid.  Returns an Idle tuple on failure and
        populates memory with an ``action_failure`` entry for the next LLM prompt.
        """
        result = self._validator.validate(name, args, self.WORLD_STATE, self.teammates)
        if result.valid:
            return None
        self.memory.update("action_failure", result.feedback)
        self._reasoning_step = True
        return self._idle()

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