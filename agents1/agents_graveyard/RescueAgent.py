"""
RescueAgent for LLM-MAS

This agent implements a modular architecture with clear separation of concerns:
- Perception: Converts MATRX state to structured representation
- Memory: Stores and retrieves experiences
- Communication: Handles inter-agent messaging (with LLM)
- Cognitive (Planning + Reasoning): Makes decisions (with LLM)
- Profile: Defines agent identity and capabilities
"""

import json
import logging
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any
import os
import yaml
from engine.toon_utils import to_toon
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from agents1.agents_graveyard.ReasoningModule import ReasoningIO
from agents1.agents_graveyard.PerceptionModule import PerceptionModule
from agents1.modules.CommunicationModule import CommunicationModule
from agents1.agents_graveyard.PlanningModule import PlanningModule
from memory.short_term_memory import ShortTermMemory

from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from engine.llm_utils import parse_json_response


class RescueAgent(PerceptionModule, ArtificialBrain):
    """
    Modular LLM-based agent implementing the target architecture.

    Data Flow:
    Environment State -> filter_observations -> Planning => Reasoning -> Action

    The agent orchestrates the modules and handles low-level execution
    (navigation, pickup, drop) while the LLM handles decision-making.
    """

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'qwen3:8b',
        include_human: bool = True,
        ollama_port: int = 11434,
        shared_message_log: list = None,
    ):
        super().__init__(slowdown, condition, name, folder)

        # Configuration
        self._human_name = name
        self._llm_model = llm_model
        self._include_human = include_human
        self._api_url = f"http://localhost:{ollama_port}"
        self.profile = 'Rescue Agent'
        self.system_message = None

        self._state_tracker: Optional[StateTracker] = None
        self._navigator: Optional[Navigator] = None
        self.state_for_navigation: State = None
        self._nav_target = None  # (x, y) target from LLM

        self.drop_zone_location = (23, 8)

        # Task context from EnginePlanner
        self._current_task = None

        # Event-driven LLM state machine
        self._reasoning_step = True           # Start by requesting first LLM call
        self.past_actions: list = []   # rolling list of the last 5 executed action strings

        # Async LLM state — all access guarded by _llm_lock
        self._pending_llm_action: Optional[concurrent.futures.Future] = None
        self._last_llm_result: Optional[dict] = None  # most recent valid LLM parse
        self._llm_lock = threading.Lock()

        self.WORLD_STATE_FILTERED: Dict[str, Any] = {
            'victims': [],       # {"id", "type", "location"}
            'obstacles': [],     # {"id", "type", "location"}
            'doors': [],         # {"id", "door"}
            'help_requests': [],  # {"location", "message", "sender"}
            'teammate_positions': {},  # {agent_id: [x, y]}
        }
        self.OBS = {}
        
        # Action failure detection (state-comparison)
        self._prev_tick_location = None
        self._prev_tick_carrying_ids = []
        self._last_action_target_id = None
        self._last_action_feedback = ''
        self._consecutive_failures = 0

        # Debug
        self._verbose = True
        self._logger = logging.getLogger('RescueAgent')
        
        # Modules
        self.reasoning_module = None
        self.communication_module = None
        self.planning_module = None
        self.short_term_memory = ShortTermMemory(memory_limit=20, llm_model=self._llm_model, api_url=self._api_url)
        
        # Planning state
        self.PLAN = ''  # current plan from PlanningModule
        self._replan_feedback = ''  # feedback from Replan action for PlanningModule

        # Agent ↔ Planner communication
        self._planner_channel = None
        self._planner_responses: list = []  # recent planner response strings

        # Mid-iteration re-tasking: callback injected by GridWorld, future for the LLM call
        self._request_task_callback = None   # callable() -> Future; set by run_with_planner
        self._retask_future: Optional[concurrent.futures.Future] = None

        # Shared message log across all agents (passed from WorldBuilder)
        self._shared_message_log = shared_message_log if shared_message_log is not None else []

    def initialize(self):
        """Initialize all components when world starts."""
        
        self._load_prompts()

        self._state_tracker = StateTracker(agent_id=self.agent_id)
        
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM
        )

        self.reasoning_module = ReasoningIO(
            llm_model=self._llm_model,
            prompts=self._prompts,
            api_url=self._api_url,
        )

        self.communication_module = CommunicationModule(
            agent_id=self.agent_id,
            send_message_fn=self._send_message,
            shared_message_log=self._shared_message_log,
        )

        self.planning_module = PlanningModule(
            llm_model=self._llm_model,
            prompts=self._prompts,
            api_url=self._api_url,
        )
        
    def set_planner_channel(self, channel):
        """Set the PlannerChannel for agent-planner communication."""
        self._planner_channel = channel

    def _load_prompts(self) -> dict:
        _prompts_file = os.path.join(
            os.path.dirname(__file__), 'prompts_rescue_agent.yaml'
        )
        with open(_prompts_file, 'r') as f:
            self._prompts = yaml.safe_load(f)
            
        self.system_message = self._prompts['agent_profile_system'].format(
            agent_id=self.agent_id,
            profile=self.profile,
        )

    def set_current_task(self, task):
        """Set the current high-level task from the EnginePlanner."""
        if not isinstance(task, str):
            task = json.dumps(task, default=str)
        self._current_task = task

        # Record in memory
        tick = 0
        if self.state_for_navigation is not None:
            try:
                tick = self.state_for_navigation['World']['nr_ticks']
            except (KeyError, TypeError):
                pass
        self.short_term_memory.update('task', {'task': task, 'tick': tick})

        # Reset all state for new task (memory is NOT reset)
        self._nav_target = None
        self._reasoning_step = False
        self._replan_feedback = ''
        self.PLAN = ''
        self._retask_future = None
        with self._llm_lock:
            self._pending_llm_action = None
            self._last_llm_result = None
        if self.planning_module is not None:
            self.planning_module.reset()

    def set_manual_plan(self, plan: str):
        """Override PlanningModule by directly setting the plan text.
        Must be called after set_current_task() (which resets self.PLAN).
        """
        if plan and plan.strip():
            self.PLAN = plan.strip()
            self._reasoning_step = True
            print(f"[{self.agent_id}] Manual plan set: {self.PLAN[:80]}...")

    def filter_observations(self, state: State) -> State:
        """
        Filter observations to only include objects within 1 block (Chebyshev distance).
        Door objects are always included regardless of distance. The agent's own state and the World state are also always included.
        This is called automatically and decide_on_actions receives the filtered state. 
        """
        agent_location = state[self.agent_id]['location']

        self.state_for_navigation = state.copy()
        filtered_state = state.copy()

        ids_to_include = set()
        ids_to_include.add(self.agent_id)
        ids_to_include.add('World')
        if self._include_human:
            ids_to_include.add(self._human_name)

        for obj_id, obj_data in filtered_state.items():
            if obj_id in ids_to_include:
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

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        # 1. PERCEPTION — always runs
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE_FILTERED = self.process_observations(filtered_state)
        self.OBS = self.observation_to_dict(filtered_state)
        self._last_filtered_state = filtered_state

        # 1b. ACTION FAILURE DETECTION — compare with previous tick's state
        self._last_action_feedback = self._detect_action_failure(filtered_state)
        if self._last_action_feedback:
            self._consecutive_failures += 1
            print(f"[{self.agent_id}] FAILED: {self._last_action_feedback}")
            # Annotate the most recent past action with the failure
            if self.past_actions:
                last = self.past_actions[-1]
                if '[FAILED' not in last:
                    self.past_actions[-1] = f"{last} [FAILED: {self._last_action_feedback}]"
            # Auto-replan after 3 consecutive failures
            if self._consecutive_failures >= 3 and self.PLAN:
                print(f"[{self.agent_id}] {self._consecutive_failures} consecutive failures, replanning")
                self._replan_feedback = f"Repeated failures: {self._last_action_feedback}"
                self.planning_module.reset()
                self.PLAN = ''
                self._reasoning_step = False
                self._consecutive_failures = 0
                return Idle.__name__, {'duration_in_ticks': 1}
        else:
            if self.previous_action is not None:
                self._consecutive_failures = 0

        # Save current state for next tick's failure detection
        self._prev_tick_location = tuple(filtered_state[self.agent_id]['location'])
        is_carrying_raw = filtered_state[self.agent_id].get('is_carrying', [])
        self._prev_tick_carrying_ids = sorted(
            o.get('obj_id', str(o)) if isinstance(o, dict) else str(o)
            for o in is_carrying_raw
        )

        # 2. RE-TASK polling
        if self._retask_future is not None:
            if self._retask_future.done():
                try:
                    task_result = self._retask_future.result()
                    agent_tasks = task_result.get('agent_tasks', {})
                    new_task = agent_tasks.get(
                        self.agent_id,
                        task_result.get('rescuebot_task', 'explore nearest area')
                    )
                    print(f"[{self.agent_id}] Re-tasked: {new_task}")
                    self.set_current_task(new_task)
                except Exception as e:
                    self._logger.warning(f"[{self.agent_id}] Re-task failed: {e}")
                    self._retask_future = None
            else:
                return Idle.__name__, {'duration_in_ticks': 1}

        # 3. No task → Idle
        if not self._current_task:
            return Idle.__name__, {'duration_in_ticks': 1}

        # 4. NAVIGATION — if mid-route, keep moving
        if self._nav_target is not None:
            move_action = self._navigator.get_move_action(self._state_tracker)
            if move_action is not None:
                return move_action, {}
            else:
                self._nav_target = None
                self._reasoning_step = True

        # 5. PLANNING — if no plan, submit/poll planning
        if self.PLAN == '':
            return self._handle_planning()

        # 6. REASONING — poll/submit LLM action
        return self._handle_reasoning()

    def _handle_planning(self) -> Tuple[str, Dict]:
        """Submit or poll the PlanningModule. Returns Idle while planning."""
        # Submit plan request if nothing in flight
        if not self.planning_module.is_planning and not self.planning_module.plan_ready:
            self.planning_module.plan(
                task=self._current_task,
                world_state=to_toon(self.WORLD_STATE_FILTERED),
                memory=self.short_term_memory.get_compact_str()[:400]
                    if self.short_term_memory.storage else '',
                feedback=self._replan_feedback,
            )
            self._replan_feedback = ''

        # Poll for plan readiness
        if not self.planning_module.plan_ready:
            self.planning_module.is_plan_ready()

            # Handle AskPlanner — needs clarification from EnginePlanner
            if self.planning_module.needs_clarification and self._planner_channel is not None:
                # Submit question only once (needs_clarification stays True until answered)
                question = self.planning_module.get_question()
                if question and question != getattr(self, '_last_submitted_question', ''):
                    tick = 0
                    try:
                        tick = self.state_for_navigation['World']['nr_ticks']
                    except (KeyError, TypeError):
                        pass
                    self._planner_channel.submit_question(
                        agent_id=self.agent_id,
                        content=question,
                        tick=tick,
                        context={'task': self._current_task},
                    )
                    self._last_submitted_question = question
                    print(f"[{self.agent_id}] Asked planner: {question}")

            # Poll for planner's response
            if self._planner_channel is not None:
                responses = self._planner_channel.poll_responses(self.agent_id)
                if responses:
                    answer = responses[0].content
                    self.planning_module.receive_answer(answer)
                    print(f"[{self.agent_id}] Planner answered: {answer}")

            return Idle.__name__, {'duration_in_ticks': 1}

        # Plan is ready — extract and move to reasoning
        self.PLAN = self.planning_module.get_plan
        self._reasoning_step = True
        print(f"[{self.agent_id}] Plan ready: {self.PLAN}")
        return Idle.__name__, {'duration_in_ticks': 1}

    def _handle_reasoning(self) -> Tuple[str, Dict]:
        """Poll/submit the ReasoningModule. Returns action or Idle."""
        print(f"[{self.agent_id}] Reasoning step")
        if self.communication_module is not None:
            self.communication_module.poll_inbound(self.received_messages)

        # Poll pending LLM future
        with self._llm_lock:
            if self._pending_llm_action is not None and self._pending_llm_action.done():
                try:
                    self._last_llm_result = self._pending_llm_action.result()
                except Exception as e:
                    self._logger.warning(f"[{self.agent_id}] LLM future raised: {e}")
                finally:
                    self._pending_llm_action = None

        # Execute result if available
        if self._last_llm_result is not None:
            result = self._last_llm_result
            self._last_llm_result = None
            # Tool-calling returns a dict {"name":..,"arguments":..}
            # Fallback returns a raw string
            if isinstance(result, dict):
                return self.tool_call_to_action(result)
            return self.text_to_action(result)

        # Submit new LLM call
        if self._reasoning_step:
            print(f"[{self.agent_id}] Submitting reasoning to LLM.")
            with self._llm_lock:
                if self._pending_llm_action is None:
                    self._pending_llm_action = self.reasoning_module(
                        self.PLAN,
                        observation=self.OBS,
                        previous_action=self.past_actions,
                        world_state=self.WORLD_STATE_FILTERED,
                        feedback=self._last_action_feedback,
                    )
                    self._reasoning_step = True
        print(f"[{self.agent_id}] No LLM result yet, idling")
        return Idle.__name__, {'duration_in_ticks': 1}

    def _detect_action_failure(self, filtered_state) -> str:
        """Compare current state with saved pre-action state to detect failures.

        Returns a short feedback string describing the failure, or '' if no
        failure detected.
        """
        prev_action = self.previous_action
        if prev_action is None or self._prev_tick_location is None:
            return ''

        current_loc = tuple(filtered_state[self.agent_id]['location'])
        is_carrying_raw = filtered_state[self.agent_id].get('is_carrying', [])
        current_carrying_ids = sorted(
            o.get('obj_id', str(o)) if isinstance(o, dict) else str(o)
            for o in is_carrying_raw
        )

        # --- Move failed: location unchanged ---
        if prev_action in ('MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest'):
            if current_loc == self._prev_tick_location:
                return f"{prev_action} failed: location unchanged at {list(current_loc)}, path blocked"

        # --- Carry failed: carrying list unchanged ---
        if prev_action in ('CarryObject', 'CarryObjectTogether'):
            if current_carrying_ids == self._prev_tick_carrying_ids:
                return f"{prev_action} failed: not carrying anything new"

        # --- Drop failed: still carrying ---
        if prev_action in ('Drop', 'DropObjectTogether'):
            if current_carrying_ids:
                return f"{prev_action} failed: still carrying {current_carrying_ids}"

        # --- Remove failed: target obstacle still in observation ---
        if prev_action in ('RemoveObject', 'RemoveObjectTogether'):
            target = self._last_action_target_id
            if target:
                for key in self.OBS:
                    if target in str(key):
                        return f"{prev_action} failed: {target} still present"

        return ''

    def _record_action_in_memory(self, action:str):
        """
            Record a completed high-level action for LLM context and logging.
        """
        self.past_actions.append(action)
        if len(self.past_actions) > 5:
            self.past_actions = self.past_actions[-5:]
        self.short_term_memory.update('action', {
            'action': action
        })
        
    def tool_call_to_action(self, tool_call: dict) -> Tuple[str, Dict]:
        """Convert a structured tool-call dict to a MATRX action.

        ``tool_call`` has the shape ``{"name": str, "arguments": dict}``
        as returned by Ollama's /api/chat tool-calling feature.  We
        translate it into the same ``(action, params)`` format that
        ``text_to_action`` produces, then reuse the shared dispatch logic.
        """
        action = tool_call.get('name', 'Idle')
        params = tool_call.get('arguments') or {}
        print(f"[{self.agent_id}]| Tool call: {action}({params})")

        # Build the same dict that text_to_action would have parsed
        parsed = {'action': action, 'params': params}
        return self._dispatch_action(parsed)

    def text_to_action(self, llm_response: str) -> Tuple[str, Dict]:
        """
            Convert LLM action decision (free-form text) to MATRX action.
            Fallback path used when the model returns plain text instead
            of a structured tool call.
        """
        self._pending_llm_action = None
        # --- Ask Planner for guidance ---
        if 'AskPlanner' in llm_response:
            question = llm_response.strip()
            if self._planner_channel is not None and question:
                tick = 0
                try:
                    tick = self.state_for_navigation['World']['nr_ticks']
                except (KeyError, TypeError):
                    pass
                context = {
                    'position': list(self._last_filtered_state[self.agent_id]['location']),
                    'memory_summary': self.short_term_memory.get_compact_str()[:300]
                        if self.short_term_memory.storage else '',
                }
                self._planner_channel.submit_question(
                    agent_id=self.agent_id,
                    content=question,
                    tick=tick,
                    context=context,
                )
                self._logger.info(f"[{self.agent_id}] Asked planner: {question}")
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        parsed_response = parse_json_response(llm_response)
        if parsed_response is not None:
                print(f"[{self.agent_id}]| Action: {parsed_response}")
        else:
            print(f"[{self.agent_id}]| PARSE of LLM response FAILED")
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        return self._dispatch_action(parsed_response)

    # ── Shared dispatch logic (used by both tool_call_to_action and text_to_action) ──

    def _dispatch_action(self, parsed_response: dict) -> Tuple[str, Dict]:
        """Route a parsed action dict to the corresponding MATRX action.

        ``parsed_response`` must have the shape
        ``{"action": str, "params": dict, ...}``  (from text_to_action)
        or  ``{"name": str, "arguments": dict}``  (from tool_call_to_action,
        which normalises it before calling here).
        """
        # Normalise: tool_call_to_action sends {"action":..,"params":..}
        action = parsed_response.get('action') or parsed_response.get('name', 'Idle')
        params = parsed_response.get('params') or parsed_response.get('arguments') or {}
        object_id = params.get('object_id')

        # Track target object for failure detection on next tick
        if action in ('CarryObject', 'CarryObjectTogether',
                       'RemoveObject', 'RemoveObjectTogether'):
            if object_id is None:
                self._reasoning_step = True
                self._last_action_feedback = f"{action} missing object_id in params. Give the correct object_id in params to execute properly."
            self._last_action_target_id = object_id
        else:
            self._last_action_target_id = None

        # --- Single-step moves ---
        if action in ('MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'):
            return action, {}

        self._record_action_in_memory(f"{action}({params}), {object_id}")

        # --- MoveTo navigation ---
        if action == 'MoveTo':
            coords = self._parse_coordinates(params)
            if coords is not None:
                self._navigator.reset_full()
                self._navigator.add_waypoints([coords])
                self._nav_target = coords
                move_action = self._navigator.get_move_action(self._state_tracker)
                if move_action is not None:
                    return move_action, {}

        # --- Carry actions ---
        if action == 'CarryObject' and object_id:
            return CarryObject.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        if action == 'CarryObjectTogether' and object_id:
            return CarryObjectTogether.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        # --- Drop actions ---
        if action == 'Drop':
            return Drop.__name__, {'human_name': self._human_name}

        if action == 'DropObjectTogether':
            return DropObjectTogether.__name__, {'human_name': self._human_name}

        # --- Remove actions (multi-tick: ArtificialBrain adds duration) ---
        if action == 'RemoveObject' and object_id:
            return RemoveObject.__name__, {
                'object_id': object_id,
                'remove_range': 1
            }

        if action == 'RemoveObjectTogether' and object_id:
            from actions1.CustomActions import RemoveObjectTogether as ROT
            return ROT.__name__, {
                'object_id': object_id,
                'remove_range': 1,
                'human_name': self._human_name
            }

        # --- Communication actions ---
        if action in ('BroadcastObservation', 'SendMessage', 'SendAcceptHelpMessage'):
            tick = 0
            try:
                tick = self.state_for_navigation['World']['nr_ticks']
            except (KeyError, TypeError):
                pass

            if action == 'BroadcastObservation':
                obs_json = self.process_observations(self._last_filtered_state)
                self.communication_module.send_templated_message(
                    'broadcast', tick,
                    observation_json=to_toon(obs_json),
                    current_task=self._current_task or 'none',
                )
            elif action == 'SendMessage':
                self.communication_module.send_templated_message(
                    'help_request', tick,
                    target_location=json.dumps([params.get('target_x', 0), params.get('target_y', 0)]),
                    action_needed=params.get('action_needed', 'RemoveObjectTogether'),
                    object_id=params.get('object_id', ''),
                )
            elif action == 'SendAcceptHelpMessage':
                self.communication_module.send_templated_message(
                    'accept_help', tick,
                    help_message=params.get('help_message', ''),
                    agent_location=json.dumps(list(
                        self._last_filtered_state[self.agent_id]['location']
                    )),
                )

            self._reasoning_step = True    # re-query for physical action
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Direct message to specific agent ---
        if action == 'SendDirectMessage':
            tick = 0
            try:
                tick = self.state_for_navigation['World']['nr_ticks']
            except (KeyError, TypeError):
                pass
            target_agent = params.get('target_agent', '')
            message_intent = params.get('message', '')
            context = params.get('context', '')
            if target_agent and message_intent and self.communication_module is not None:
                self.communication_module.send_direct_message(
                    target_agent_id=target_agent,
                    tick=tick,
                    message_intent=message_intent,
                    context=context,
                )
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Navigate to drop zone ---
        if action == 'NavigateToDropZone':
            coords = self.drop_zone_location
            self._navigator.reset_full()
            self._navigator.add_waypoints([coords])
            self._nav_target = coords
            move_action = self._navigator.get_move_action(self._state_tracker)
            if move_action is not None:
                return move_action, {}

        # --- Replan: reasoning can't make progress with current plan ---
        if action == 'Replan':
            reason = params.get('reason', '')
            print(f"[{self.agent_id}] Replanning: {reason}")
            self._replan_feedback = reason
            self.planning_module.reset()
            self.PLAN = ''
            self._reasoning_step = False
            return Idle.__name__, {'duration_in_ticks': 1}

        # --- Task completion: request new task from planner ---
        if action == 'TaskComplete':
            print(f"[{self.agent_id}] Task completed, requesting new task")
            if self._request_task_callback is not None:
                self._retask_future = self._request_task_callback()
                self._current_task = None
                self.PLAN = ''
            else:
                # No callback — clear task and idle
                self._current_task = None
                self.PLAN = ''
            return Idle.__name__, {'duration_in_ticks': 1}

        return Idle.__name__, {'duration_in_ticks': 1}

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

    ### COMMUNICATION
    def _send_message(self, content: str, sender: str, target_id: str = None):
        """Send message to teammates. If target_id is set, send direct message."""
        msg = Message(content=content, from_id=sender, to_id=target_id)
        if content not in [m.content for m in self.messages_to_send]:
            self.send_message(msg)

