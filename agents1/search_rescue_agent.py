"""
SearchRescueAgent — MATRX AgentBrain subclass powered by MARBLE's LLM layer.
"""

import json
import logging
from concurrent.futures import Future
import re
from typing import Optional, Tuple, Dict, Any, List
from agents1.modules.planning_module import Planning
from agents1.modules.reasoning_module import ReasoningIO, ReasoningBase
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents.agent_utils.state import State
from matrx.messages.message import Message

from brains1.ArtificialBrain import ArtificialBrain

# Action class aliases (needed for _maybe_share_observation comparisons
# and _Idle fallback returns).
from actions1.CustomActions import Idle as _Idle
from actions1.CustomActions import CarryObject as _CarryObject
from actions1.CustomActions import CarryObjectTogether as _CarryObjectTogether
from actions1.CustomActions import DropObjectTogether as _DropObjectTogether
from matrx.actions.object_actions import RemoveObject as _RemoveObject
from actions1.CustomActions import RemoveObjectTogether as _RemoveObjectTogether

# Extracted modules
from memory.base_memory import BaseMemory
from memory.shared_memory import SharedMemory
from agents1.async_model_prompting import submit_llm_call, get_llm_result
from agents1.modules.perception_module import Perception
from agents1.action_mapper import ActionMapper
from agents1.modules.execution_module import execute_action
from agents1.tool_registry import REASONING_STRATEGIES, build_tool_schemas

logger = logging.getLogger('SearchRescueAgent')

MAX_NR_TOKENS = 3000
TEMPERATURE = 0.3

# ── Agent ──────────────────────────────────────────────────────────────────────

class SearchRescueAgent(ArtificialBrain, Perception):
    """Lightweight MARBLE-powered rescue agent for the MATRX environment.

    Designed for concurrency: LLM calls go into a ThreadPoolExecutor;
    the agent returns Idle every tick while waiting so it never blocks others.

    The LLM receives OpenAI-style tool schemas and returns a structured
    tool_call.  Plain-text JSON is accepted as a fallback.

    Args:
        slowdown:       ArtificialBrain parameter (tick slow-down factor).
        condition:      World condition ('normal' | 'strong' | 'weak').
        name:           Default cooperative partner name (human or AI agent ID).
        folder:         Working folder path (passed to ArtificialBrain).
        llm_model:      LiteLLM model string, e.g. ``"ollama/llama3"``.
        strategy:       Reasoning strategy: 'cot', 'react', or 'reflexion'.
        include_human:  Whether to include the human in the observation filter.
        shared_memory:  Optional SharedMemory instance for cross-agent sharing.
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
    ) -> None:
        super().__init__(slowdown, condition, name, folder)

        self._llm_model = llm_model
        self._strategy = strategy if strategy in REASONING_STRATEGIES else 'react'
        self._include_human = include_human
        self._partner_name = name
        self.teammates = set("rescuebot1,rescuebot0".split(',')) - {self.agent_id}


        # ── Tool registry ──────────────────────────────────────────────────
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        # ── MARBLE memory ──────────────────────────────────────────────────
        self.memory = BaseMemory()
        self.shared_memory: Optional[SharedMemory] = shared_memory

        # ── Navigation ─────────────────────────────────────────────────────
        self._state_tracker: Optional[StateTracker] = None
        self._navigator: Optional[Navigator] = None
        self.state_for_navigation: Optional[State] = None
        self._nav_target: Optional[Tuple[int, int]] = None

        # ── Async LLM state ────────────────────────────────────────────────
        self._pending_future: Optional[Future] = None
        self._reasoning_step: bool = True   # True = submit a new LLM call next tick

        # ── Task ───────────────────────────────────────────────────────────
        self._current_task: Optional[str] = None

        # ── Helpers ────────────────────────────────────────────────────────
        self._mapper = ActionMapper(partner_name=name)   # text-JSON fallback parser

        self.planner = Planning()
        self.reasoning = ReasoningIO("EMPTY")
        self.WORLD_STATE = ''
        self.task_decomposition = ''

        # ── Action feedback ───────────────────────────────────────────────
        self._action_feedback: str = ''  # Feedback from failed action validation

        # ── Atomic carry-drop state ──────────────────────────────────────
        self._carry_drop_mode: bool = False      # In atomic carry→navigate→drop sequence
        self._carry_partner: str = ''            # Partner name for DropObjectTogether
        self._carry_wait_ticks: int = 0          # Ticks spent waiting for partner at victim
        
        self.task_num = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Called once before the world starts. Set up navigation tools."""
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM,
        )
        self.init_global_state()
        print(f"[{self.agent_id}] SearchRescueAgent ready "
              f"(model={self._llm_model}, strategy={self._strategy}, "
              f"tools={len(self.tool_schemas)})")

    # ── Task management ────────────────────────────────────────────────────

    def set_current_task(self, task: str) -> None:
        """Inject a high-level task from the EnginePlanner.

        Resets navigation and pending LLM state for the new task.
        Per-agent memory is preserved across task transitions.
        """
        if not isinstance(task, str):
            task = json.dumps(task, default=str)
        self._current_task = task
        self._pending_future = None
        self._nav_target = None
        self._reasoning_step = True
        self._carry_drop_mode = False
        self._carry_partner = ''
        self._carry_wait_ticks = 0
        self.planner.update_current_task(task)
        print(f"[{self.agent_id}] Task: {task}")
        
    def set_manual_task_decomposition(self, decomposition: str) -> None:
        """Override the PlanningModule by directly setting the plan text.

        Must be called after set_current_task() (which resets self.task_decomposition).
        """
        task_array = re.findall(r'\d+\.\s*(.*)', decomposition)

        self.planner.set_manual_task_decomposition(task_array)
        self.task_num = len(task_array)

    # ── MATRX hooks ───────────────────────────────────────────────────────

    def _is_within_range(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        radius: int,
    ) -> bool:
        """Chebyshev distance check."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius

    def filter_observations(self, state: State) -> State:
        """Restrict observations to 1-block Chebyshev radius + doors + self.

        Full unfiltered state is saved to ``state_for_navigation`` for A*.
        """
        agent_loc = state[self.agent_id]['location']
        self.state_for_navigation = state.copy()
        filtered = state.copy()
        self.teammates = set()

        keep = {self.agent_id, 'World'}
        if self._include_human:
            keep.add(self._partner_name)
            self.teammates.add((self._partner_name, state.get(self._partner_name, {}).get('location', [0, 0])))
                        

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

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:

        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(filtered_state, agent_id=self.agent_id, teammates=self.teammates)
        self.process_observations(filtered_state)

        if not self._current_task:
            return _Idle.__name__, {'duration_in_ticks': 1}

        # ── Invisible partner detection (cooperative carry in progress) ──
        my_data = filtered_state.get(self.agent_id, {})
        my_opacity = my_data.get('visualization', {}).get('opacity', 1)
        if my_opacity == 0:
            # We are the invisible partner in a cooperative carry — idle silently
            return _Idle.__name__, {'duration_in_ticks': 1}

        # ── Atomic carry-drop sequence (handles its own navigation) ────
        if self._carry_drop_mode:
            if self._nav_target is not None:
                # Still navigating to drop zone
                move = self._navigator.get_move_action(self._state_tracker)
                if move is not None:
                    return move, {}
                # Arrived at drop zone — drop and exit mode
                self._nav_target = None
                self._carry_drop_mode = False
                self._reasoning_step = True
                print(f"[{self.agent_id}] Arrived at drop zone, executing DropObjectTogether")
                return _DropObjectTogether.__name__, {'partner_name': self._carry_partner}
            else:
                # Just finished carry action — set up navigation to drop zone
                drop_zone = (23, 8)
                self._navigator.reset_full()
                self._navigator.add_waypoints([drop_zone])
                self._nav_target = drop_zone
                move = self._navigator.get_move_action(self._state_tracker)
                print(f"[{self.agent_id}] Carry-drop: navigating to drop zone {drop_zone}")
                return (move, {}) if move else (_Idle.__name__, {'duration_in_ticks': 1})

        # continue navigation (normal LLM-driven mode)
        if self._nav_target is not None:
            move = self._navigator.get_move_action(self._state_tracker)
            if move is not None:
                return move, {}
            # Reached destination
            self._nav_target = None
            self._reasoning_step = True

        planner_prompt = self.planner.get_task_decomposition_prompt({
            'world_state': self.WORLD_STATE,
            'memory': self.memory.retrieve_all()[-10:],
            'feedback': self.planner._feedback,
        })

        # ── SharedMemory rendezvous: navigate to partner's carry request ─
        if self.shared_memory and not self._carry_drop_mode:
            rendezvous = self.shared_memory.retrieve('carry_rendezvous')
            if (rendezvous
                    and rendezvous.get('agent') != self.agent_id
                    and rendezvous.get('status') == 'waiting_for_partner'):
                target = tuple(rendezvous['location'])
                my_loc = (
                    self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                    if isinstance(self.WORLD_STATE, dict) else [0, 0]
                )
                if not self._is_within_range(tuple(my_loc), target, radius=1):
                    # Navigate to partner's location to assist with carry
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([target])
                    self._nav_target = target
                    self._pending_future = None  # Cancel any pending LLM call
                    move = self._navigator.get_move_action(self._state_tracker)
                    print(f"[{self.agent_id}] Navigating to carry rendezvous at {target}")
                    return (move, {}) if move else (_Idle.__name__, {'duration_in_ticks': 1})

        # poll LLM future
        if self._pending_future is not None:
            try:
                result = get_llm_result(self._pending_future)
            except Exception as exc:
                logger.warning("[%s] LLM future raised: %s", self.agent_id, exc)
                self._pending_future = None
                self._reasoning_step = True
                return _Idle.__name__, {'duration_in_ticks': 1}

            if result is None:
                return _Idle.__name__, {'duration_in_ticks': 1}  # still waiting
            return self._handle_llm_result(filtered_state, result)

        # submit LLM call with tool schemas
        if self._reasoning_step:
            if self.task_num <= 0:
                print(f"[{self.agent_id}] No more tasks in decomposition, idling.")
                return _Idle.__name__, {'duration_in_ticks': 1}
            observation = dict(self.WORLD_STATE)
            global_state = self.WORLD_STATE_GLOBAL
            if global_state['victims'] or global_state['obstacles'] or global_state['doors']:
                observation['known'] = {
                    k: v for k, v in global_state.items()
                    if k != 'teammate_positions' and v
                }
            reasoning_prompt = self.reasoning.get_reasoning_prompt({
                'task_decomposition': self.planner.task_decomposition[-self.task_num:],
                'observation': observation,
                'feedback': self._action_feedback,
                'memory': self.memory.retrieve_all()[-15:],
            })
            print(f"[{self.agent_id}] Submitting LLM call with reasoning prompt: {reasoning_prompt}")
                
            self._pending_future = submit_llm_call(
                llm_model=self._llm_model,
                messages=reasoning_prompt,
                max_token_num=MAX_NR_TOKENS,
                temperature=TEMPERATURE,
                tools=self.tool_schemas,
                tool_choice='auto',
            )
            self.task_num -= 1  
            self._action_feedback = ''  # Clear after consumption
            print(f"[{self.agent_id}] LLM call submitted")

        return _Idle.__name__, {'duration_in_ticks': 1}

    # ── LLM result handling ────────────────────────────────────────────────

    def _handle_llm_result(
        self, filtered_state: State, result: List
    ) -> Tuple[str, Dict]:
        """Dispatch the LLM response to a MATRX action.

        Prefers structured tool_call; falls back to text-JSON parsing
        (ActionMapper) when the model returns plain text.
        """
        message = result[0]  # model_prompting returns List[Message]
        self._pending_future = None

        # ── Find partner (shared by Path A and Path B) ────────────────
        my_location = None
        for i in self.teammates:
            if i[0] == self.agent_id:
                my_location = i[1]
                break
            else:
                my_location = [0, 0]
        partner = None
        for i in self.teammates:
            if i[0] != self.agent_id:
                partner = i[0]
                break

        # ── Path A: structured tool_call ──────────────────────────────────
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            tc = tool_calls[0]
            name = tc.function.name
            args_raw = tc.function.arguments
            args: Dict[str, Any] = (
                json.loads(args_raw)
                if isinstance(args_raw, str)
                else (args_raw or {})
            )
            print(f"[{self.agent_id}] Tool call: {name}({args})")
            self.send_message(Message(content=f"Executing action {name}({args})", from_id=self.agent_id))

            # Validate object actions against perceived world state.
            validation_result = self._validate_action(name, args)
            if validation_result is not None:
                return validation_result

            action_name, kwargs = execute_action(
                name, args, partner, self.agent_id
            )

            # Check with MATRX before dispatching
            matrx_check = self._check_matrx_action(action_name, kwargs)
            if matrx_check is not None:
                return matrx_check

            self.memory.update('action', {'action': action_name, 'args': kwargs})
            self._maybe_share_observation(filtered_state, action_name, kwargs)
            return self._apply_navigation(action_name, kwargs)

        # ── Path B: plain-text JSON fallback ─────────────────────────────
        llm_text = getattr(message, 'content', '') or ''
        print(f"[{self.agent_id}] Text response: {llm_text[:120]}")

        raw_name, raw_args = self._mapper.parse_raw(llm_text)
        if raw_name is None:
            # Unparseable response — re-query the LLM next tick.
            self._reasoning_step = True
            return _Idle.__name__, {'duration_in_ticks': 1}

        # Validate object actions against perceived world state.
        validation_result = self._validate_action(raw_name, raw_args)
        if validation_result is not None:
            return validation_result

        action_name, kwargs = execute_action(
            raw_name, raw_args, partner, self.agent_id
        )

        # Check with MATRX before dispatching
        matrx_check = self._check_matrx_action(action_name, kwargs)
        if matrx_check is not None:
            return matrx_check

        self.memory.update('action', {'action': action_name, 'args': kwargs})
        self._maybe_share_observation(filtered_state, action_name, kwargs)
        return self._apply_navigation(action_name, kwargs)

    def _apply_navigation(
        self, action_name: str, kwargs: Dict[str, Any]
    ) -> Tuple[str, Dict]:
        """Set up the A* navigator for MoveTo / NavigateToDropZone, then
        return the first step.  All other actions are passed through unchanged.
        """
        self._reasoning_step = True   # re-query LLM after this action completes

        if action_name == 'MoveTo':
            coords = (int(kwargs.get('x', 0)), int(kwargs.get('y', 0)))
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else (_Idle.__name__, {'duration_in_ticks': 1})

        if action_name == 'NavigateToDropZone':
            coords = (int(kwargs.get('x', 23)), int(kwargs.get('y', 8)))
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move = self._navigator.get_move_action(self._state_tracker)
            return (move, {}) if move else (_Idle.__name__, {'duration_in_ticks': 1})

        # ── Atomic carry-drop: trigger mode on CarryObjectTogether ────
        if action_name == _CarryObjectTogether.__name__:
            self._carry_drop_mode = True
            self._carry_partner = kwargs.get('partner_name', '')
            self._carry_wait_ticks = 0
            self._reasoning_step = False  # Don't re-query LLM during atomic sequence
            if self.shared_memory:
                self.shared_memory.update('carry_rendezvous', None)  # Clear rendezvous
            print(f"[{self.agent_id}] Entering atomic carry-drop mode")
            return action_name, kwargs

        return action_name, kwargs

    # ── Action validation ──────────────────────────────────────────────────

    # Actions that require a valid, in-range object_id.
    _OBJECT_ACTIONS = frozenset({
        'CarryObject', 'CarryObjectTogether',
        'RemoveObject', 'RemoveObjectTogether',
    })

    def _validate_action(
        self,
        name: str,
        args: Dict[str, Any],
    ) -> Optional[Tuple[str, Dict]]:
        """Validate *object_id* against ``WORLD_STATE`` before dispatching.

        Returns ``None`` when the action is valid (caller should proceed with
        ``execute_action``).  Returns ``(Idle, kwargs)`` when the action is
        invalid — the caller should return this tuple and skip dispatching.
        On failure, ``self._action_feedback`` is populated with a description
        that will be injected into the next LLM prompt.
        """
        if name not in self._OBJECT_ACTIONS:
            return None  # No validation needed for movement / idle / etc.

        obj_id = args.get('object_id', '')
        nearby = (
            self.WORLD_STATE.get('nearby', [])
            if isinstance(self.WORLD_STATE, dict) else []
        )

        # Build a compact summary of actionable objects currently in range.
        actionable_types = {'victim', 'rock', 'stone', 'tree'}
        actionable = [
            o for o in nearby if o.get('type') in actionable_types
        ]
        nearby_summary = ', '.join(
            f"{o['id']} ({o['type']}"
            + (f", {o.get('severity')}" if o.get('severity') else "")
            + f" at {o['location']})"
            for o in actionable
        ) or 'none'
        agent_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )

        if not obj_id:
            self._action_feedback = (
                f"Action {name} requires an object_id but none was provided. "
                f"Nearby actionable objects: [{nearby_summary}]. "
                f"Agent location: {agent_loc}."
            )
            self._reasoning_step = True
            logger.warning("[%s] %s", self.agent_id, self._action_feedback)
            return _Idle.__name__, {'duration_in_ticks': 1}

        # Check if the requested object_id is among perceived nearby objects.
        nearby_ids = {o['id'] for o in nearby}
        if obj_id not in nearby_ids:
            self._action_feedback = (
                f"Action {name} failed: object '{obj_id}' is not within reach "
                f"(1-block range) or does not exist. "
                f"Nearby actionable objects: [{nearby_summary}]. "
                f"Agent location: {agent_loc}. "
                f"Move closer to the target or choose a different object."
            )
            self._reasoning_step = True
            logger.warning("[%s] %s", self.agent_id, self._action_feedback)
            return _Idle.__name__, {'duration_in_ticks': 1}

        # Valid — clear any stale feedback.
        self._action_feedback = ''
        return None

    # ── MATRX authority check ─────────────────────────────────────────────

    # Actions we skip the MATRX feasibility check for (meta-actions handled
    # by the navigator, or always-succeed actions).
    _SKIP_MATRX_CHECK = frozenset({
        'Idle', 'MoveTo', 'NavigateToDropZone',
        'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest',
    })

    def _check_matrx_action(
        self,
        action_name: str,
        kwargs: Dict[str, Any],
    ) -> Optional[Tuple[str, Dict]]:
        """Ask MATRX whether *action_name* with *kwargs* would succeed.

        Returns ``None`` if the action is possible (caller proceeds).
        Returns ``(Idle, kwargs)`` if MATRX rejects it, with feedback
        populated in ``self._action_feedback``.
        """
        if action_name in self._SKIP_MATRX_CHECK:
            return None

        # Enrich kwargs the same way ArtificialBrain.decide_on_action does
        # (grab_range, max_objects) so the MATRX action class receives the
        # parameters it expects when evaluating feasibility.
        check_kwargs = dict(kwargs)
        check_kwargs.setdefault('grab_range', 1)
        check_kwargs.setdefault('max_objects', 1)
        if 'injured' not in kwargs.get('object_id', ''):
                print(f"[{self.agent_id}] MATRX rejected {action_name} for non-victim object '{kwargs.get('object_id', '')}'")
                return _Idle.__name__, {'duration_in_ticks': 1}

        succeeded, action_result = self.is_action_possible(action_name, check_kwargs)
        if succeeded:
            return None

        # ── Wait-for-partner on CarryObjectTogether failure ──────────
        if action_name == _CarryObjectTogether.__name__:
            self._carry_wait_ticks += 1
            if self._carry_wait_ticks <= 30:  # Wait up to 30 ticks for partner
                self._reasoning_step = False  # Don't re-query LLM, just wait
                # Publish rendezvous so partner navigates to us
                if self.shared_memory:
                    agent_loc = (
                        self.WORLD_STATE.get('agent', {}).get('location', [0, 0])
                        if isinstance(self.WORLD_STATE, dict) else [0, 0]
                    )
                    obj_id = kwargs.get('object_id', '')
                    self.shared_memory.update('carry_rendezvous', {
                        'agent': self.agent_id,
                        'victim_id': obj_id,
                        'location': agent_loc,
                        'status': 'waiting_for_partner',
                    })
                print(f"[{self.agent_id}] Waiting for partner ({self._carry_wait_ticks}/30)")
                return _Idle.__name__, {'duration_in_ticks': 1}
            # After 30 ticks, give up and re-query LLM
            self._carry_wait_ticks = 0
            print(f"[{self.agent_id}] Gave up waiting for partner")
            # Fall through to normal failure handling

        # Build informative feedback for the LLM
        nearby = (
            self.WORLD_STATE.get('nearby', [])
            if isinstance(self.WORLD_STATE, dict) else []
        )
        actionable_types = {'victim', 'rock', 'stone', 'tree'}
        nearby_summary = ', '.join(
            f"{o['id']} ({o['type']}"
            + (f", {o.get('severity')}" if o.get('severity') else "")
            + f" at {o['location']})"
            for o in nearby if o.get('type') in actionable_types
        ) or 'none'
        agent_loc = (
            self.WORLD_STATE.get('agent', {}).get('location', 'unknown')
            if isinstance(self.WORLD_STATE, dict) else 'unknown'
        )

        self._action_feedback = (
            f"MATRX rejected action {action_name}: {action_result.result} "
            f"Nearby actionable objects: [{nearby_summary}]. "
            f"Agent location: {agent_loc}."
        )
        self._reasoning_step = True
        logger.warning("[%s] %s", self.agent_id, self._action_feedback)
        return _Idle.__name__, {'duration_in_ticks': 1}

    # ── Shared memory publishing ───────────────────────────────────────────

    def _maybe_share_observation(
        self,
        state: State,
        action_name: str,
        kwargs: Dict[str, Any],
    ) -> None:
        """Publish victim / obstacle discoveries to SharedMemory so other
        agents can read them on the next tick.
        """
        if self.shared_memory is None:
            return

        agent_loc = list(state.get(self.agent_id, {}).get('location', [0, 0]))

        if action_name in (_CarryObject.__name__, _CarryObjectTogether.__name__):
            obj_id = kwargs.get('object_id', '')
            if obj_id:
                self.shared_memory.update(
                    f'victim_{obj_id}',
                    {
                        'agent': self.agent_id,
                        'victim_id': obj_id,
                        'location': agent_loc,
                        'action': action_name,
                    },
                )

        if action_name in (_RemoveObject.__name__, _RemoveObjectTogether.__name__):
            obj_id = kwargs.get('object_id', '')
            if obj_id:
                self.shared_memory.update(
                    f'obstacle_{obj_id}',
                    {
                        'agent': self.agent_id,
                        'obstacle_id': obj_id,
                        'location': agent_loc,
                        'action': action_name,
                    },
                )

    # ── Messaging ──────────────────────────────────────────────────────────

    def _send_message(self, content: str, sender: str, target_id: str = None) -> None:
        """Send a MATRX message to teammates (deduplicates before sending)."""
        msg = Message(content=content, from_id=sender, to_id=target_id)
        if content not in [m.content for m in self.messages_to_send]:
            self.send_message(msg)
