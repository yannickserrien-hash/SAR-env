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
import json
import logging
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any
from engine.toon_utils import to_toon
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from agents1.modules.ReasoningModule import ReasoningIO
from agents1.modules.PerceptionModule import PerceptionModule
from agents1.modules.CommunicationModule import CommunicationModule
from memory.short_term_memory import ShortTermMemory

from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from engine.llm_utils import parse_json_response


class RescueAgent(PerceptionModule, ArtificialBrain):
    """
    Modular LLM-based agent implementing the target architecture.

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
        llm_model: str = 'llama3:8b',
        include_human: bool = True,
        ollama_port: int = 11434
    ):
        super().__init__(slowdown, condition, name, folder)

        # Configuration
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._llm_model = llm_model
        self._include_human = include_human
        self._api_url = f"http://localhost:{ollama_port}"
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

        # Persistent ASCII map: (x,y) -> single char symbol, updated every tick.
        self._known_cells: dict = {}         # (int,int) -> str
        self._known_cell_owners: dict = {}   # (int,int) -> obj_id of removable object at that cell

        # Persistent world state: accumulates interesting objects across ticks.
        # Grouped by type -> list of items.  Types with actionable objects
        # (victims, obstacles, doors) store {id, pos}; wall/blocked store just [x,y].
        # _world_state_index maps obj_id -> type for fast lookup / removal.
        self._world_state: Dict[str, list] = {}
        self._world_state_index: Dict[str, str] = {}  # obj_id -> type

        # Task context from EnginePlanner
        self._current_task = None

        # Hybrid LLM/Navigator: track navigation state
        self._nav_target = None  # (x, y) target from LLM
        self._resume_nav_target: Optional[Tuple[int, int]] = None  # re-navigate here after auto-removing obstacle

        # Event-driven LLM state machine
        self._reasoning_step = True           # Start by requesting first LLM call
        self.action_completed = False         # True after a one-shot action was returned to MATRX
        self._current_llm_action: Optional[dict] = None  # LLM response currently being executed
        self.last_action = None

        # Async LLM state — all access guarded by _llm_lock
        self._pending_llm_future: Optional[concurrent.futures.Future] = None
        self._last_llm_result: Optional[dict] = None  # most recent valid LLM parse
        self._llm_lock = threading.Lock()


        # Mid-iteration re-tasking: callback injected by GridWorld, future for the LLM call
        self._request_task_callback = None   # callable() -> Future; set by run_with_planner
        self._retask_future: Optional[concurrent.futures.Future] = None

        self.profile = 'Rescue Agent'
        # system_message is built in initialize() once prompts are loaded from YAML
        self.system_message = None

        # Structured short-term memory (LLM-curated, persists across tasks)
        self.memory = ShortTermMemory(memory_limit=20, llm_model=self._llm_model, api_url=self._api_url)
        self.agent_graph = [self._human_name]

        # Persistent world-knowledge dict updated every tick via add_new_obs().
        # Passed directly to the LLM as structured context.
        self.MEMORY: Dict[str, Any] = {
            'known_victims': [],       # {"id", "type", "location", "rescued"}
            'known_obstacles': [],     # {"id", "type", "location"}
            'door_houses': [],         # {"id", "door"}
            'pending_help_requests': [],  # {"location", "message", "sender"}
            'teammate_positions': {},  # {agent_id: [x, y]}
        }
        
        self.state_from_engine: State = None
        # Debug
        self._verbose = True
        self._logger = logging.getLogger('RescueAgent')
        
        # Modules
        self.reasoning_module = None
        self.communication_module = None


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
            memory=self.memory,
            llm_model=[self._llm_model],
            prompts=self._prompts,
            api_url=self._api_url,
        )

        self.communication_module = CommunicationModule(
            llm_model=self._llm_model,
            prompts=self._prompts,
            agent_id=self.agent_id,
            send_message_fn=self._send_message,
            memory=self.memory,
            world_memory=self.MEMORY,
            api_url=self._api_url,
        )

    def set_current_task(self, task: str):
        """Set the current high-level task from the EnginePlanner."""
        self._current_task = task
        print(f"[{self.agent_id}] Acting on task '{task}'.")
        self._send_message(f"[{self.agent_id}] Received task: {task}", self.agent_id)

        # Record task assignment in memory (persists across tasks)
        tick = 0
        if self.state_from_engine is not None:
            try:
                tick = self.state_from_engine['World']['nr_ticks']
            except (KeyError, TypeError):
                pass
        self.memory.update('task', {
            'task': task,
            'tick': tick,
        })

        # Reset navigation and LLM state for new task (memory is NOT reset)
        self._nav_target = None
        self._resume_nav_target = None
        self._reasoning_step = True
        self.action_completed = False
        self._current_llm_action = None
        with self._llm_lock:
            self._pending_llm_future = None
            self._last_llm_result = None
        self._retask_future = None  # discard any pending re-task when a new task arrives

    def filter_observations(self, state: State) -> State:
        """
        Filter observations to only include objects within 1 block (Chebyshev distance).
        """
        agent_info = state[self.agent_id]
        agent_location = agent_info['location']

        self.state_from_engine = state.copy()
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

    def decide_on_actions(self, state: State) -> Tuple[Optional[str], Dict]:
        """
            Agent Brain.
        """
        # Process observation
        self.MEMORY = self.observation_to_json(state)
  
        self._state_tracker.update(self.state_from_engine)
        self._last_filtered_state = state
        
        if self.communication_module is not None:
            self.communication_module.poll_outbound()
            self.communication_module.poll_inbound(self.received_messages)

        if self.action_completed:
            self.action_completed = False
            self._reasoning_step = True

        agent_loc = state[self.agent_id]['location']

        # If no task assigned yet, idle
        if not self._current_task:
            return Idle.__name__, {'duration_in_ticks': 1}

        # Let Agent finish MoveTo() action
        if self._nav_target is not None:
            move_action = self._navigator.get_move_action(self._state_tracker)

            if move_action is not None: # Still moving to target
                print(f"[{self.agent_id}] Navigating to {self._nav_target} | "
                        f"pos={agent_loc} | action={move_action}")
                
                return move_action, {}
            else: # Arrived at target or stuck 
                self._nav_target = None
                self._current_llm_action = None
                self._reasoning_step = True

        # --- Do action ---
        if self._last_llm_result is not None:
            self._current_llm_action = self._last_llm_result
            self._last_llm_result = None
            return self.text_to_action(self._current_llm_action)
        
        # --- Get Reasoning Module response (Async) ---
        with self._llm_lock:
            if self._pending_llm_future is not None and self._pending_llm_future.done():
                try:
                    self._last_llm_result = self._pending_llm_future.result()
                except Exception as e:
                    self._logger.warning(f"[{self.agent_id}] LLM future raised: {e}")
                finally:
                    self._pending_llm_future = None

        # --- Reasoning step (LLM via ReasoningModule) ---
        if self._reasoning_step:
            with self._llm_lock:
                no_pending = self._pending_llm_future is None
            if no_pending:
                future = self.reasoning_module(self._current_task, observation=self.state_to_json(state), previous_action=self.last_action, world_state=to_toon(self.MEMORY))
                
                with self._llm_lock:
                    self._pending_llm_future = future
                self._reasoning_step = False 

        # --- Waiting for LLM response ---
        return Idle.__name__, {'duration_in_ticks': 1}

    def _record_action_in_memory(self, action:str, description: str):
        """
            Record a completed high-level action for LLM context and logging.
        """
        self.last_action = action
        self.memory.update('action', {
            'action': action
        })
        
    def text_to_action(self, llm_response: str) -> Tuple[str, Dict]:
        """
            Convert LLM action decision to MATRX action.
        """
        parsed_response = parse_json_response(llm_response)
        print(f"[{self.agent_id}] Parsed LLM response: {parsed_response}")
        if parsed_response is not None:
                print(f"[{self.agent_id}] LLM response received -> {parsed_response}")
        else:
            print(f"[{self.agent_id}] PARSE of LLM response FAILED")
            self._reasoning_step = True
            return Idle.__name__, {'duration_in_ticks': 1}
            
        action = parsed_response.get('action', 'Idle')
        params = parsed_response.get('params') or {}
        object_id = params.get('object_id')
        reasoning = params.get('reasoning', '')[:100]

        # --- Single-step moves ---
        if action in ('MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'):
            self.action_completed = True
            return action, {}

        self._record_action_in_memory(f"{action}({params}), {object_id}", reasoning)
        
        print(f"[{self.agent_id}] Executing LLM action -> {action} params={params}")

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
            self.action_completed = True
            return CarryObject.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        if action == 'CarryObjectTogether' and object_id:
            self.action_completed = True
            return CarryObjectTogether.__name__, {
                'object_id': object_id,
                'human_name': self._human_name
            }

        # --- Drop actions ---
        if action == 'Drop':
            self.action_completed = True
            return Drop.__name__, {'human_name': self._human_name}

        if action == 'DropObjectTogether':
            self.action_completed = True
            return DropObjectTogether.__name__, {'human_name': self._human_name}

        # --- Remove actions ---
        if action == 'RemoveObject' and object_id:
            self.action_completed = True
            return RemoveObject.__name__, {
                'object_id': object_id,
                'remove_range': 1
            }

        if action == 'RemoveObjectTogether' and object_id:
            from actions1.CustomActions import RemoveObjectTogether as ROT
            self.action_completed = True
            return ROT.__name__, {
                'object_id': object_id,
                'remove_range': 1,
                'human_name': self._human_name
            }
            
        # --- Communication actions ---
        if action in ('BroadcastObservation', 'SendMessage', 'SendAcceptHelpMessage'):
            tick = 0
            try:
                tick = self.state_from_engine['World']['nr_ticks']
            except (KeyError, TypeError):
                pass

            if action == 'BroadcastObservation':
                obs_json = self.observation_to_json(self._last_filtered_state)
                self.communication_module.generate_message(
                    'broadcast', tick,
                    observation_json=to_toon(obs_json),
                    world_state_str="",
                    current_task=self._current_task or 'none',
                )
            elif action == 'SendMessage':
                self.communication_module.generate_message(
                    'help_request', tick,
                    target_location=json.dumps(params.get('target_location', [0, 0])),
                    action_needed=params.get('action_needed', 'RemoveObjectTogether'),
                    object_id=params.get('object_id', ''),
                    context=reasoning,
                )
            elif action == 'SendAcceptHelpMessage':
                self.communication_module.generate_message(
                    'accept_help', tick,
                    help_message=params.get('help_message', ''),
                    agent_location=json.dumps(list(
                        self._last_filtered_state[self.agent_id]['location']
                    )),
                )

            self._reasoning_step = True    # re-query for physical action
            self.action_completed = False   # no MATRX action performed
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
        """Return True when the LLM signals the current task is done.
        """
        return result.contains('TASK COMPLETED')

