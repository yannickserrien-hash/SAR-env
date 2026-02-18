"""
RescueAgent for LLM-MAS

This agent implements a modular architecture with clear separation of concerns:
- Perception: Converts MATRX state to structured representation
- Memory: Stores and retrieves experiences
- Communication: Handles inter-agent messaging (with LLM)
- Cognitive (Planning + Reasoning): Makes decisions (with LLM)
- Profile: Defines agent identity and capabilities
"""

import enum
import logging
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from agents1.modules.ReasoningModule import ReasoningIO
from memory.short_term_memory import ShortTermMemory

from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from engine.llm_utils import query_llm_async, parse_json_response


class RescueAgent(ArtificialBrain):
    """
    Modular LLM + Navigator agent implementing the target architecture.

    Data Flow:
    Environment State -> filter_observations -> LLM reasoning -> Actions

    The agent orchestrates the modules and handles low-level execution
    (navigation, pickup, drop) while the LLM handles decision-making.
    """

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'llama3:8b'
    ):
        super().__init__(slowdown, condition, name, folder)

        # Configuration
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._llm_model = llm_model
        self._intro_done = False

        self._human_start_location = None

        # Navigation targets (from action decisions)
        self._target_room: Optional[str] = None
        self._target_victim: Optional[str] = None
        self._target_obstacle: Optional[str] = None

        # MATRX components (initialized in initialize())
        self._state_tracker: Optional[StateTracker] = None
        self._navigator: Optional[Navigator] = None

        # Cooperative state
        self._carrying_together = False
        self._carrying = False

        self.drop_zone_location = (23, 8)

        # Task context from EnginePlanner
        self._current_task = None

        # Hybrid LLM/Navigator: track navigation state
        self._nav_target = None  # (x, y) target from LLM
        self._last_nav_location: Optional[Tuple[int, int]] = None  # position before last nav tick
        self._nav_stuck_ticks: int = 0  # consecutive ticks with no position change during navigation
        self._resume_nav_target: Optional[Tuple[int, int]] = None  # re-navigate here after auto-removing obstacle

        # Event-driven LLM state machine
        self._reasoning_step = True           # Start by requesting first LLM call
        self._oneshot_executed = False         # True after a one-shot action was returned to MATRX
        self._current_llm_action: Optional[dict] = None  # LLM response currently being executed
        self._action_history: List[str] = []  # High-level action history for LLM context

        # Async LLM state — all access guarded by _llm_lock
        self._pending_llm_future: Optional[concurrent.futures.Future] = None
        self._last_llm_result: Optional[dict] = None  # most recent valid LLM parse
        self._llm_lock = threading.Lock()

        # Async memory extraction future (separate from reasoning LLM)
        self._pending_memory_future: Optional[concurrent.futures.Future] = None

        self.actions: List[str] = []
        self.profile = 'Rescue Agent'
        # system_message is built in initialize() once prompts are loaded from YAML
        self.system_message = None

        # Structured short-term memory (LLM-curated, persists across tasks)
        self.memory = ShortTermMemory(memory_limit=20, llm_model=self._llm_model)
        self.agent_graph = [self._human_name]
        self.task_history = []
        
        self.state_from_engine: State = None
        # Debug
        self._verbose = True
        self._logger = logging.getLogger('RescueAgent')
        
        # Modules
        self.reasoning_module = None


    def initialize(self):
        """Initialize all components when world starts."""
        import os
        import yaml

        # Initialize MATRX components
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM
        )

        # Load prompt templates from YAML
        _prompts_file = os.path.join(
            os.path.dirname(__file__), 'prompts_rescue_agent.yaml'
        )
        with open(_prompts_file, 'r') as f:
            self._prompts = yaml.safe_load(f)

        # Build system_message from YAML template now that agent_id is available
        self.system_message = self._prompts['agent_profile_system'].format(
            agent_id=self.agent_id,
            profile=self.profile,
        )

        self.reasoning_module = ReasoningIO(
            profile_type_prompt="",
            memory=None,
            _llm_model=[self._llm_model],
            prompts=self._prompts,
        )

        if self._verbose:
            print(f"[RescueAgent] Initialized with LLM model {self._llm_model}")

    def set_current_task(self, task: str):
        """Set the current high-level task from the EnginePlanner."""
        self._current_task = task
        self.task_history.append(task)
        print(f"Agent '{self.agent_id}' acting on task '{task}'.")

        # Record task assignment in memory (persists across tasks)
        tick = 0
        if self.state_from_engine is not None:
            try:
                tick = self.state_from_engine['World']['nr_ticks']
            except (KeyError, TypeError):
                pass
        self.memory.update('task', {
            'type': 'task_assigned',
            'task': task,
            'tick': tick,
        })

        # Reset navigation and LLM state for new task (memory is NOT reset)
        self._nav_target = None
        self._last_nav_location = None
        self._nav_stuck_ticks = 0
        self._resume_nav_target = None
        self._reasoning_step = True
        self._oneshot_executed = False
        self._current_llm_action = None
        self._action_history = []
        with self._llm_lock:
            self._pending_llm_future = None
            self._last_llm_result = None

    def filter_observations(self, state: State) -> State:
        """
        Filter observations to only include objects within 1 block (Chebyshev distance).
        """
        agent_info = state[self.agent_id]
        agent_location = agent_info['location']

        filtered_state = state.copy()
        self.state_from_engine = state

        ids_to_include = set()
        ids_to_include.add(self.agent_id)
        ids_to_include.add('World')
        ids_to_include.add('humanagent')

        for obj_id, obj_data in filtered_state.items():
            if obj_id == self.agent_id or obj_id == 'humanagent' or obj_id == 'World':
                continue
            if self._is_within_range(agent_location, obj_data.get('location'), radius=1):
                ids_to_include.add(obj_id)
            if 'door' in str(obj_id).lower():
                ids_to_include.add(obj_id)
                
        keys_start = list(filtered_state.keys())
        for obj_id in keys_start:
            if obj_id not in ids_to_include:
                filtered_state.remove(obj_id)

        return filtered_state

    def decide_on_actions(self, state: State) -> Tuple[Optional[str], Dict]:
        """
            Decision method using LLM reasoning.
        """
        # Update state tracker every tick (required for Navigator pathfinding)
        self._state_tracker.update(state)
        
        filtered_state = self.filter_observations(state)

        # --- Harvest memory extraction results (non-blocking) ---
        self._harvest_memory_future()

        agent_loc = state[self.agent_id]['location']

        # If no task assigned yet, idle
        if not self._current_task:
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Phase A: If navigating, keep moving (no LLM interruption) ---
        if self._nav_target is not None:
            move_action = self._navigator.get_move_action(self._state_tracker)

            if move_action is not None:
                # -- Proactive obstacle scan: check if next cell has a removable obstacle --
                _move_deltas = {
                    'MoveNorth': (0, -1), 'MoveSouth': (0, 1),
                    'MoveEast': (1, 0), 'MoveWest': (-1, 0),
                }
                if move_action in _move_deltas:
                    d = _move_deltas[move_action]
                    next_cell = (agent_loc[0] + d[0], agent_loc[1] + d[1])
                    full_st = self.state_from_engine or {}
                    for oid, od in full_st.items():
                        if not isinstance(od, dict):
                            continue
                        if 'ObstacleObject' not in od.get('class_inheritance', []):
                            continue
                        if tuple(od.get('location', ())) != next_cell:
                            continue
                        oid_l = str(oid).lower()
                        if 'stone' in oid_l or 'tree' in oid_l:
                            obj_type = 'stone' if 'stone' in oid_l else 'tree'
                            self._record_action_completion(
                                f"MoveTo({self._nav_target}) -> proactive removal of "
                                f"{obj_type} '{oid}' at {next_cell}",
                                state=filtered_state,
                            )
                            print(f"[RescueAgent] *** PROACTIVE REMOVE *** "
                                  f"{obj_type} '{oid}' at {next_cell}")
                            self._resume_nav_target = self._nav_target
                            self._nav_target = None
                            self._nav_stuck_ticks = 0
                            self._last_nav_location = None
                            self._current_llm_action = None
                            self._oneshot_executed = True
                            return RemoveObject.__name__, {
                                'object_id': oid, 'remove_range': 1
                            }
                        break  # obstacle found but it's a rock — let nav continue

                # -- Stuck detection: position unchanged for 2+ ticks --
                if self._last_nav_location is not None and agent_loc == self._last_nav_location:
                    self._nav_stuck_ticks += 1
                else:
                    self._nav_stuck_ticks = 0
                self._last_nav_location = agent_loc

                if self._nav_stuck_ticks >= 2:
                    result = self._handle_nav_blocked(agent_loc, 'STUCK', obs_state=filtered_state)
                    if result is not None:
                        return result
                    # else: fall through to Phase D (reasoning_step already set)
                else:
                    print(f"[RescueAgent] Phase A: navigating to {self._nav_target} | "
                          f"pos={agent_loc} | action={move_action} | stuck={self._nav_stuck_ticks}")
                    return move_action, {}

            else:
                # Navigator returned None
                if self._navigator.is_done:
                    # Genuinely arrived
                    print(f"[RescueAgent] Phase A: ARRIVED at {self._nav_target} | pos={agent_loc}")
                    self._record_action_completion(
                        f"MoveTo({self._nav_target}) -> arrived at {agent_loc}",
                        state=filtered_state,
                    )
                    self._nav_target = None
                    self._nav_stuck_ticks = 0
                    self._last_nav_location = None
                    self._current_llm_action = None
                    self._reasoning_step = True
                else:
                    # No path — try auto-remove
                    print(f"[RescueAgent] Phase A: NO PATH to {self._nav_target} from {agent_loc}")
                    result = self._handle_nav_blocked(agent_loc, 'NO_PATH', obs_state=filtered_state)
                    if result is not None:
                        return result
                    # else: fall through (reasoning_step already set)

        # --- Phase B: Detect one-shot action completion ---
        if self._oneshot_executed:
            action_name = self._current_llm_action.get('action', '?') if self._current_llm_action else '?'
            target_val = (self._current_llm_action.get('params') or
                          self._current_llm_action.get('target', '')) if self._current_llm_action else ''
            succeeded = (hasattr(self, 'previous_action_result')
                         and self.previous_action_result is not None
                         and self.previous_action_result.succeeded)
            if succeeded:
                result_str = 'succeeded'
            else:
                matrx_reason = ''
                if hasattr(self, 'previous_action_result') and self.previous_action_result is not None:
                    matrx_reason = f' ({self.previous_action_result.result})'
                result_str = f'FAILED{matrx_reason}'
            self._record_action_completion(
                f"{action_name}({target_val}) -> {result_str}",
                state=filtered_state,
            )
            print(f"[RescueAgent] Phase B: one-shot completed | {action_name}({target_val}) -> {result_str}")
            self._oneshot_executed = False
            self._current_llm_action = None

            # Resume navigation after successful auto-remove
            if self._resume_nav_target is not None and succeeded:
                coords = self._resume_nav_target
                self._resume_nav_target = None
                self._navigator.reset_full()
                self._navigator.add_waypoints([coords])
                self._nav_target = coords
                self._nav_stuck_ticks = 0
                self._last_nav_location = None
                self._reasoning_step = False
                print(f"[RescueAgent] Phase B: auto-remove succeeded, "
                      f"resuming navigation to {coords}")
                move_action = self._navigator.get_move_action(self._state_tracker)
                if move_action is not None:
                    return move_action, {}
                # Still can't navigate — fall through to LLM
                print(f"[RescueAgent] Phase B: still no path after removal, querying LLM")
                self._nav_target = None

            # Clear resume target if removal failed or not applicable
            self._resume_nav_target = None
            self._reasoning_step = True

        # --- Phase C: Harvest completed LLM future ---
        with self._llm_lock:
            if self._pending_llm_future is not None and self._pending_llm_future.done():
                try:
                    raw_response = self._pending_llm_future.result()
                    parsed = parse_json_response(raw_response)
                    if parsed is not None:
                        self._last_llm_result = parsed
                        print(f"[RescueAgent] Phase C: LLM response received -> {parsed}")
                    else:
                        print(f"[RescueAgent] Phase C: LLM response PARSE FAILED | raw={raw_response[:200]}")
                except Exception as e:
                    self._logger.warning(f"[RescueAgent] LLM future raised: {e}")
                    print(f"[RescueAgent] Phase C: LLM future EXCEPTION: {e}")
                finally:
                    self._pending_llm_future = None

        # --- Phase C.5: Detect parse failure and retry ---
        with self._llm_lock:
            no_pending = self._pending_llm_future is None
        if (no_pending and self._last_llm_result is None
                and self._current_llm_action is None
                and not self._reasoning_step):
            # LLM call completed but produced no usable result — retry
            print(f"[RescueAgent] Phase C.5: no usable LLM result, retrying")
            self._reasoning_step = True

        # --- Phase D: Reasoning step (LLM via ReasoningModule) ---
        if self._reasoning_step:
            with self._llm_lock:
                no_pending = self._pending_llm_future is None
            if no_pending:
                user_prompt = self._build_reasoning_prompt(filtered_state)
                print(f"[RescueAgent] Phase D: dispatching LLM call | "
                      f"pos={agent_loc} | task={self._current_task if self._current_task else 'None'}")
                print(f"[RescueAgent] Phase D: prompt observations:\n{self._serialize_state_for_llm(filtered_state)}")
                future = self.reasoning_module(user_prompt)
                with self._llm_lock:
                    self._pending_llm_future = future
                self._reasoning_step = False  # Request submitted

        # --- Phase E: Act on newly arrived LLM result (consume once) ---
        if self._last_llm_result is not None:
            self._current_llm_action = self._last_llm_result
            self._last_llm_result = None  # Consume — one-time use
            print(f"[RescueAgent] Phase E: executing LLM action -> "
                  f"{self._current_llm_action.get('action')} "
                  f"params={self._current_llm_action.get('params')} "
                  f"reasoning={self._current_llm_action.get('reasoning', '')[:80]}")
            return self._llm_action_to_matrx(self._current_llm_action, filtered_state)

        # --- Phase F: Waiting for LLM response ---
        return Idle.__name__, {'duration_in_ticks': 1}

    def _serialize_state_for_llm(self, state: State) -> str:
        """Convert filtered MATRX state to text for LLM prompt."""
        agent_info = state[self.agent_id]
        agent_loc = agent_info['location']

        parts = [f"Your position: {agent_loc}"]

        # Check if carrying
        is_carrying = agent_info.get('is_carrying', [])
        if is_carrying:
            carried_names = [obj.get('obj_id', str(obj)) if isinstance(obj, dict)
                           else str(obj) for obj in is_carrying]
            parts.append(f"Carrying: {carried_names}")
            self._carrying = True
        else:
            parts.append("Carrying: nothing")
            self._carrying = False

        # Human position
        human_info = state.get('humanagent', None)
        if human_info and isinstance(human_info, dict):
            parts.append(f"Human teammate '{self._human_name}' at {human_info.get('location', 'unknown')}")

        # Nearby objects — use class_inheritance for reliable type detection
        nearby = []
        for obj_id, obj_data in state.items():
            if obj_id in (self.agent_id, 'World', 'humanagent'):
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location', 'unknown')
            img = obj_data.get('img_name', '')
            class_inh = obj_data.get('class_inheritance', [])
            is_traversable = obj_data.get('is_traversable', True)

            # --- Victims (collectable objects) ---
            if obj_data.get('is_collectable', False):
                victim_type = 'unknown'
                if 'critical' in str(img).lower():
                    victim_type = 'critically injured'
                elif 'mild' in str(img).lower():
                    victim_type = 'mildly injured'
                elif 'healthy' in str(img).lower():
                    victim_type = 'healthy'
                nearby.append(f"  Victim '{obj_id}' ({victim_type}) at {loc}")
            # --- Walls ---
            elif 'Wall' in class_inh:
                nearby.append(f"  Wall at {loc} [non-traversable]")
            # --- Doors ---
            elif 'Door' in class_inh:
                door_open = obj_data.get('is_open', True)
                nearby.append(f"  Door '{obj_id}' at {loc} ({'open' if door_open else 'closed, non-traversable'})")
            # --- Obstacles: type-specific info ---
            elif 'ObstacleObject' in class_inh:
                oid_lower = str(obj_id).lower()
                if 'rock' in oid_lower:
                    nearby.append(
                        f"  Rock '{obj_id}' at {loc} "
                        f"[requires BOTH agents to remove with RemoveObjectTogether]")
                elif 'stone' in oid_lower:
                    nearby.append(
                        f"  Stone '{obj_id}' at {loc} "
                        f"[removable by you alone with RemoveObject]")
                elif 'tree' in oid_lower:
                    nearby.append(
                        f"  Tree '{obj_id}' at {loc} "
                        f"[removable by you alone with RemoveObject]")
                else:
                    nearby.append(
                        f"  Obstacle '{obj_id}' at {loc} [non-traversable]")
            # --- Roof tiles (indicate agent is inside a room) ---
            elif 'roof' in str(obj_id).lower():
                nearby.append(f"  Roof at {loc}")
            # --- Any other non-traversable object ---
            elif not is_traversable:
                nearby.append(f"  Blocked object '{obj_id}' at {loc} [non-traversable]")
            # --- Skip traversable non-interesting objects (AreaTile, floor, etc.) ---

        if nearby:
            parts.append("Nearby objects (within 1 block):")
            parts.extend(nearby)
        else:
            parts.append("Nearby objects: none visible")

        # Recent messages
        if hasattr(self, 'received_messages') and self.received_messages:
            msg_texts = [m.content for m in self.received_messages[-3:]]
            parts.append(f"Recent messages: {msg_texts}")

        return "\n".join(parts)

    def _record_action_completion(self, description: str, state: State = None):
        """Record a completed high-level action for LLM context and logging.

        Appends to _action_history (capped at 15 most-recent entries to prevent
        prompt bloat) and to self.actions (full session log).  Prints to stdout
        when verbose mode is enabled.

        If *state* is provided and no memory extraction is already in flight,
        fires an async LLM call to decide what to save to short-term memory.
        """
        # Append to full session log (no cap — used for output logging)
        self.actions.append(description)

        # Append to rolling LLM-context window (capped at 15)
        self._action_history.append(description)
        if len(self._action_history) > 15:
            self._action_history = self._action_history[-15:]

        if self._verbose:
            print(f"[RescueAgent] Action completed: {description} "
                  f"(history: {len(self._action_history)} entries)")

        # Fire async memory extraction (non-blocking)
        if state is not None and self._pending_memory_future is None:
            observation = self._serialize_state_for_llm(state)
            self._extract_memory_async(observation, description)

    # ------------------------------------------------------------------
    # Memory extraction (async, non-blocking)
    # ------------------------------------------------------------------

    def _extract_memory_async(self, observation: str, action_desc: str):
        """Ask the LLM what from the latest action/observation to save.

        Dispatches a background LLM call whose result is harvested later by
        ``_harvest_memory_future`` at the top of ``decide_on_actions``.
        """
        # Gather recent communications
        comms = "None"
        if hasattr(self, 'received_messages') and self.received_messages:
            comms = "\n".join(m.content for m in self.received_messages[-5:])

        user_prompt = self._prompts['memory_extract_user'].format(
            observation=observation,
            action_description=action_desc,
            communications=comms,
            current_task=self._current_task or "No task assigned",
            existing_memory=self.memory.get_compact_str(),
        )
        system_prompt = self._prompts['memory_extract_system']

        self._pending_memory_future = query_llm_async(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1,
        )
        if self._verbose:
            print("[RescueAgent] Memory extraction dispatched")

    def _harvest_memory_future(self):
        """Poll the memory extraction future; store entries if ready.

        Called at the top of every ``decide_on_actions`` tick so the
        result is consumed as soon as it is available without blocking.
        """
        if self._pending_memory_future is None:
            return
        if not self._pending_memory_future.done():
            return  # still running — try again next tick

        try:
            response = self._pending_memory_future.result()
            parsed = parse_json_response(response)
            if parsed and 'entries' in parsed and isinstance(parsed['entries'], list):
                for entry in parsed['entries']:
                    if isinstance(entry, dict) and 'type' in entry:
                        self.memory.update(entry.get('type', ''), entry)
                        if self._verbose:
                            print(f"[RescueAgent] Memory stored: {entry}")
                if not parsed['entries'] and self._verbose:
                    print("[RescueAgent] Memory extraction: nothing new to save")
            else:
                if self._verbose:
                    print(f"[RescueAgent] Memory extraction: unparseable response")
        except Exception as e:
            print(f"[RescueAgent] Memory extraction failed: {e}")
        finally:
            self._pending_memory_future = None
    
    def _build_reasoning_prompt(self, filtered_state: State) -> str:
        """Build the user prompt string for the ReasoningModule.

        Includes the current sub-task, serialized observations, structured
        memory snapshot, action history, and the previous MATRX action result.
        """
        state_text = self._serialize_state_for_llm(filtered_state)

        # Build action history text
        if self._action_history:
            history_text = "\n".join(
                f"  {i + 1}. {a}" for i, a in enumerate(self._action_history)
            )
        else:
            history_text = "  (no actions taken yet)"

        # Build memory context (compact structured JSON)
        if self.memory.storage:
            memory_text = self.memory.get_compact_str()
            # Truncate if too long to avoid blowing up token budget
            if len(memory_text) > 500:
                memory_text = memory_text[:500] + "..."
        else:
            memory_text = "(empty)"

        # Determine obstacle-specific hint from YAML (appended when last action was blocked)
        blocked_hint = ""
        if self._action_history:
            last = self._action_history[-1]
            if 'BLOCKED' in last:
                if 'stone' in last.lower() or 'tree' in last.lower():
                    blocked_hint = self._prompts['reasoning_blocked_stone_or_tree'].strip()
                elif 'rock' in last.lower():
                    blocked_hint = self._prompts['reasoning_blocked_rock'].strip()
                else:
                    blocked_hint = self._prompts['reasoning_blocked_unknown'].strip()

        user_prompt = self._prompts['reasoning_user'].format(
            current_task=self._current_task,
            state_text=state_text,
            memory_text=memory_text,
            history_text=history_text,
            previous_action=self.previous_action if hasattr(self, 'previous_action') else 'None',
            blocked_hint=blocked_hint,
        )

        if self._verbose:
            print(f"[RescueAgent] Reasoning prompt dispatched. "
                  f"History: {len(self._action_history)} actions")

        return user_prompt

    def _llm_action_to_matrx(self, llm_response: dict, state: State) -> Tuple[str, Dict]:
        """Convert LLM action decision to MATRX action tuple.

        Supports both ReasoningModule schema  (action + params dict) and
        legacy schema (action + target string).

        For MoveTo: sets _nav_target so Phase A drives the navigator.
        For all other actions: sets _oneshot_executed so Phase B records
        completion and triggers a new LLM call.
        """
        action = llm_response.get('action', 'Idle')
        params = llm_response.get('params') or {}
        # Fallback: legacy schema used 'target' instead of 'params'
        target = llm_response.get('target')
        # Resolve object_id: prefer params['object_id'], fall back to target
        object_id = params.get('object_id') or target

        if self._verbose:
            print(f"[RescueAgent] LLM decision: {action} params={params} "
                  f"target={target} reason={llm_response.get('reasoning', '')}")

        # --- MoveTo: multi-tick navigation (completion detected in Phase A) ---
        if action == 'MoveTo':
            # Try params dict first (ReasoningModule schema), then target
            coords = self._parse_coordinates(params) or self._parse_coordinates(target)
            if coords is not None:
                self._navigator.reset_full()
                self._navigator.add_waypoints([coords])
                self._nav_target = coords
                self._nav_stuck_ticks = 0
                self._last_nav_location = None
                move_action = self._navigator.get_move_action(self._state_tracker)
                if move_action is not None:
                    return move_action, {}
            # Failed to parse or navigate
            self._record_action_completion(f"MoveTo({params or target}) -> failed", state=state)
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Single-step moves (one-shot) ---
        if action in ('MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'):
            self._oneshot_executed = True
            return action, {}

        # --- Door action ---
        if action == 'OpenDoorAction':
            if object_id:
                self._oneshot_executed = True
                return 'OpenDoorAction', {'object_id': object_id, 'door_range': 1}
            self._record_action_completion("OpenDoorAction -> no object_id", state=state)
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Carry actions ---
        if action == 'CarryObject' and object_id:
            self._oneshot_executed = True
            return CarryObject.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        if action == 'CarryObjectTogether' and object_id:
            self._oneshot_executed = True
            return CarryObjectTogether.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        # --- Drop actions ---
        if action == 'Drop':
            self._oneshot_executed = True
            return Drop.__name__, {'human_name': self._human_name}

        if action == 'DropObjectTogether':
            self._oneshot_executed = True
            return DropObjectTogether.__name__, {'human_name': self._human_name}

        # --- Remove actions ---
        if action == 'RemoveObject' and object_id:
            self._oneshot_executed = True
            return RemoveObject.__name__, {
                'object_id': object_id,
                'remove_range': 1
            }

        if action == 'RemoveObjectTogether' and object_id:
            from actions1.CustomActions import RemoveObjectTogether as ROT
            self._oneshot_executed = True
            return ROT.__name__, {
                'object_id': object_id,
                'remove_range': 1,
                'human_name': self._human_name
            }

        # --- Navigate to drop zone (fixed destination, no params needed) ---
        if action == 'NavigateToDropZone':
            coords = self.drop_zone_location  # (23, 8)
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            self._nav_stuck_ticks = 0
            self._last_nav_location = None
            move_action = self._navigator.get_move_action(self._state_tracker)
            if move_action is not None:
                return move_action, {}
            # No path available yet — let LLM decide next tick
            self._record_action_completion("NavigateToDropZone -> no path", state=state)
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Idle or unrecognized: re-query after 1 tick ---
        self._oneshot_executed = True
        return Idle.__name__, {'duration_in_ticks': 1}

    ## HELPERS

    def _handle_nav_blocked(self, agent_loc, reason_tag: str,
                            obs_state: State = None) -> Optional[Tuple[str, Dict]]:
        """Handle a blocked navigation: auto-remove solo obstacles, request
        cooperation for rocks, or fall back to LLM re-query.

        Returns a (action, kwargs) tuple if an auto-remove action should be
        executed immediately, or None if the caller should fall through to the
        LLM reasoning step (which is armed by this method).
        """
        saved_target = self._nav_target
        obstacle = self._identify_blocking_obstacle(agent_loc, self._nav_target)

        def _clear_nav():
            self._nav_target = None
            self._nav_stuck_ticks = 0
            self._last_nav_location = None
            self._current_llm_action = None

        if obstacle and obstacle['solo_removable'] and \
                self._is_within_range(agent_loc, obstacle['location'], radius=1):
            # --- Auto-remove stone or tree ---
            self._record_action_completion(
                f"MoveTo({saved_target}) -> BLOCKED by {obstacle['obj_type']} "
                f"'{obstacle['obj_id']}' at {obstacle['location']}. Auto-removing.",
                state=obs_state,
            )
            print(f"[RescueAgent] *** AUTO-REMOVE ({reason_tag}) *** "
                  f"{obstacle['obj_type']} '{obstacle['obj_id']}' at {obstacle['location']}")
            self._resume_nav_target = saved_target
            _clear_nav()
            self._oneshot_executed = True
            return RemoveObject.__name__, {
                'object_id': obstacle['obj_id'], 'remove_range': 1
            }

        elif obstacle and not obstacle['solo_removable']:
            # --- Rock: needs cooperation ---
            self._record_action_completion(
                f"MoveTo({saved_target}) -> BLOCKED by {obstacle['obj_type']} "
                f"'{obstacle['obj_id']}' at {obstacle['location']}. "
                f"Requires human cooperation to remove.",
                state=obs_state,
            )
            print(f"[RescueAgent] *** BLOCKED BY ROCK ({reason_tag}) *** "
                  f"'{obstacle['obj_id']}' at {obstacle['location']}")
            self._send_message(
                f"Found rock blocking my path at {obstacle['location']}. "
                f"Can you come help me remove it?",
                'RescueBot'
            )
            _clear_nav()
            self._reasoning_step = True
            return None  # let LLM decide

        else:
            # --- No identifiable obstacle ---
            old_action = self._current_llm_action or {}
            prev_result_msg = ''
            if hasattr(self, 'previous_action_result') and \
                    self.previous_action_result is not None:
                prev_result_msg = f" MATRX says: {self.previous_action_result.result}"
            self._record_action_completion(
                f"MoveTo({saved_target}) -> BLOCKED at {agent_loc} ({reason_tag})."
                f"{prev_result_msg} "
                f"Original reasoning: {old_action.get('reasoning', 'N/A')}",
                state=obs_state,
            )
            print(f"[RescueAgent] *** NAV BLOCKED ({reason_tag}) *** at {agent_loc} "
                  f"target={saved_target}")
            _clear_nav()
            self._reasoning_step = True
            return None  # let LLM decide

    def _parse_coordinates(self, target) -> Optional[Tuple[int, int]]:
        """Parse an LLM-provided target into an (x, y) tuple.

        Handles formats: [3, 5], (3, 5), "3, 5", "(3, 5)", "[3, 5]",
        {"x": 3, "y": 5}, and already-parsed lists/tuples.
        """
        if target is None:
            return None
        # Already a list or tuple of two numbers
        if isinstance(target, (list, tuple)) and len(target) == 2:
            try:
                return (int(target[0]), int(target[1]))
            except (ValueError, TypeError):
                return None
        # Dict with x/y keys
        if isinstance(target, dict):
            try:
                return (int(target['x']), int(target['y']))
            except (KeyError, ValueError, TypeError):
                return None
        # String: strip brackets/parens and split on comma
        if isinstance(target, str):
            import re
            nums = re.findall(r'-?\d+', target)
            if len(nums) >= 2:
                return (int(nums[0]), int(nums[1]))
        return None

    def _is_within_range(self, pos1: Tuple[int, int], pos2: Tuple[int, int], radius: int) -> bool:
        """Check if two positions are within a given radius using Chebyshev distance."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius

    def _identify_blocking_obstacle(self, agent_loc, nav_target) -> Optional[Dict]:
        """Scan full state for an ObstacleObject adjacent to the agent in the
        direction of nav_target.

        Returns dict with keys obj_id, obj_type, location, solo_removable,
        or None if no obstacle found.
        """
        full_state = self.state_from_engine
        if full_state is None or nav_target is None:
            return None

        # Direction from agent toward target (sign only)
        dx = nav_target[0] - agent_loc[0]
        dy = nav_target[1] - agent_loc[1]
        sx = (1 if dx > 0 else -1 if dx < 0 else 0)
        sy = (1 if dy > 0 else -1 if dy < 0 else 0)

        # Priority cells: direct forward, then cardinal components
        priority_cells = []
        if sx != 0 or sy != 0:
            priority_cells.append((agent_loc[0] + sx, agent_loc[1] + sy))
        if sx != 0:
            priority_cells.append((agent_loc[0] + sx, agent_loc[1]))
        if sy != 0:
            priority_cells.append((agent_loc[0], agent_loc[1] + sy))

        # All 8 neighbours as fallback
        all_neighbors = set()
        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                if ddx == 0 and ddy == 0:
                    continue
                all_neighbors.add((agent_loc[0] + ddx, agent_loc[1] + ddy))

        def _classify(obj_id_str):
            oid = str(obj_id_str).lower()
            if 'rock' in oid:
                return 'rock', False
            elif 'stone' in oid:
                return 'stone', True
            elif 'tree' in oid:
                return 'tree', True
            return 'unknown', False

        # First pass: check priority cells
        for obj_id, obj_data in full_state.items():
            if not isinstance(obj_data, dict):
                continue
            if 'ObstacleObject' not in obj_data.get('class_inheritance', []):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue
            loc_t = tuple(loc)
            if loc_t in priority_cells:
                obj_type, solo = _classify(obj_id)
                return {'obj_id': obj_id, 'obj_type': obj_type,
                        'location': loc_t, 'solo_removable': solo}

        # Second pass: any adjacent obstacle
        for obj_id, obj_data in full_state.items():
            if not isinstance(obj_data, dict):
                continue
            if 'ObstacleObject' not in obj_data.get('class_inheritance', []):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue
            loc_t = tuple(loc)
            if loc_t in all_neighbors:
                obj_type, solo = _classify(obj_id)
                return {'obj_id': obj_id, 'obj_type': obj_type,
                        'location': loc_t, 'solo_removable': solo}

        return None

    ### COMMUNICATION
    def _send_message(self, content: str, sender: str):
        """Send message to teammates."""
        msg = Message(content=content, from_id=sender)
        if content not in [m.content for m in self.messages_to_send]:
            self.send_message(msg)

    ### ENGINE INTEGRATION METHODS
    def plan_task(self) -> str:
        """
        Generate next subtask using cognitive module.
        Called by the Engine during the planning phase.
        """
        return self._current_task or "explore nearest area"
    
    def _is_task_completed(self, result: Any) -> bool:
        """
        Determine if the task is completed based on the result of the last action.

        Args:
            result (Any): The result from the last action.

        Returns:
            bool: True if task is completed, False otherwise.
        """
        # Placeholder logic; implement actual completion criteria
        if isinstance(result, str):
            return "completed" in result.lower()
        return False

