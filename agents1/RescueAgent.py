"""
RescueAgent for LLM-MAS

This agent implements a modular architecture with clear separation of concerns:
- Perception: Converts MATRX state to structured representation
- Memory: Stores and retrieves experiences
- Communication: Handles inter-agent messaging (with LLM)
- Cognitive (Planning + Reasoning): Makes decisions (with LLM)
- Profile: Defines agent identity and capabilities

The agent uses a HYBRID approach:
- LLM modules decide HIGH-LEVEL goals and actions
- Navigator handles LOW-LEVEL movement (A* pathfinding)
"""

import enum
import logging
from typing import List, Optional, Tuple, Dict, Any
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from engine.llm_utils import query_llm, parse_json_response


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

        # State machine for low-level execution
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

        self.drop_zone_location = (23, 11)

        # Task context from EnginePlanner
        self._current_task = None

        # Hybrid LLM/Navigator: track navigation state
        self._nav_target = None  # (x, y) target from LLM
        self._ticks_since_llm = 0
        self._llm_call_interval = 8  # Re-query LLM every N ticks when navigating

        # Debug
        self._verbose = True
        self._logger = logging.getLogger('RescueAgent')

    def initialize(self):
        """Initialize all components when world starts."""
        # Initialize MATRX components
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(
            agent_id=self.agent_id,
            action_set=self.action_set,
            algorithm=Navigator.A_STAR_ALGORITHM
        )

        if self._verbose:
            print(f"[RescueAgent] Initialized with LLM model {self._llm_model}")

    def set_current_task(self, task: str):
        """Set the current high-level task from the EnginePlanner."""
        self._current_task = task
        # Reset navigation when task changes
        self._nav_target = None
        self._ticks_since_llm = 0

    def filter_observations(self, state: State) -> State:
        """
        Filter observations to only include objects within 1 block (Chebyshev distance).
        """
        agent_info = state[self.agent_id]
        agent_location = agent_info['location']

        filtered_state = state.copy()

        ids_to_include = set()
        ids_to_include.add(self.agent_id)
        ids_to_include.add('World')
        ids_to_include.add('humanagent')

        for obj_id, obj_data in filtered_state.items():
            if obj_id == self.agent_id or obj_id == 'humanagent' or obj_id == 'World':
                continue
            if self._is_within_range(agent_location, obj_data.get('location'), radius=1):
                ids_to_include.add(obj_id)

        keys_start = list(filtered_state.keys())
        for obj_id in keys_start:
            if obj_id not in ids_to_include:
                filtered_state.remove(obj_id)

        return filtered_state

    def decide_on_actions(self, state: State) -> Tuple[Optional[str], Dict]:
        """
        Main decision method using LLM reasoning.

        Uses the current task from the EnginePlanner and the filtered state
        to decide on the next action via LLM.
        """
        if not self._intro_done:
            return self._handle_intro(state)

        filtered_state = self.filter_observations(state)

        # If no task assigned yet, idle
        if not self._current_task:
            return Idle.__name__, {'duration_in_ticks': 1}

        # If navigating and haven't hit the LLM interval, continue nav
        if self._nav_target and self._ticks_since_llm < self._llm_call_interval:
            self._ticks_since_llm += 1
            move_action = self._navigator.get_move_action(self._state_tracker)
            if move_action is not None:
                return move_action[0], move_action[1] if len(move_action) > 1 else {}
            # Arrived at destination — fall through to LLM
            self._nav_target = None

        # Query LLM for next action
        self._ticks_since_llm = 0
        state_text = self._serialize_state_for_llm(filtered_state)
        llm_response = self._query_llm_for_action(state_text)

        if llm_response is None:
            return Idle.__name__, {'duration_in_ticks': 1}

        return self._llm_action_to_matrx(llm_response, filtered_state)

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

        # Nearby objects
        nearby = []
        for obj_id, obj_data in state.items():
            if obj_id in (self.agent_id, 'World', 'humanagent'):
                continue
            if not isinstance(obj_data, dict):
                continue
            name = obj_data.get('name', obj_id)
            loc = obj_data.get('location', 'unknown')
            img = obj_data.get('img_name', '')

            # Categorize
            if obj_data.get('is_collectable', False):
                victim_type = 'unknown'
                if 'critical' in str(img).lower():
                    victim_type = 'critically injured'
                elif 'mild' in str(img).lower():
                    victim_type = 'mildly injured'
                elif 'healthy' in str(img).lower():
                    victim_type = 'healthy'
                nearby.append(f"  Victim '{obj_id}' ({victim_type}) at {loc}")
            elif 'rock' in str(obj_id) or 'stone' in str(obj_id) or 'tree' in str(obj_id):
                nearby.append(f"  Obstacle '{obj_id}' at {loc}")
            elif obj_data.get('is_open_door', False) is not None and 'door' in str(obj_id).lower():
                door_open = obj_data.get('is_open', True)
                nearby.append(f"  Door '{obj_id}' at {loc} ({'open' if door_open else 'closed'})")

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

    def _query_llm_for_action(self, state_text: str) -> Optional[dict]:
        """Query the LLM for the next action decision."""
        system_prompt = (
            "You are RescueBot, an AI rescue agent in a 25x24 grid world. "
            "You must decide your next action based on what you see and your current task. "
            "Always respond with valid JSON only.\n\n"
            "Available actions:\n"
            "- MoveNorth: Move one cell up (y decreases)\n"
            "- MoveEast: Move one cell right (x increases)\n"
            "- MoveSouth: Move one cell down (y increases)\n"
            "- MoveWest: Move one cell left (x decreases)\n"
            "- CarryObject: Pick up a mildly injured victim alone (need object_id)\n"
            "- CarryObjectTogether: Pick up a critically injured victim with human nearby (need object_id)\n"
            "- Drop: Drop a victim you are carrying\n"
            "- DropObjectTogether: Drop a victim carried together\n"
            "- RemoveObject: Remove a tree or small stone alone (need object_id)\n"
            "- RemoveObjectTogether: Remove a big rock with human nearby (need object_id)\n"
            "- OpenDoorAction: Open a closed door (need object_id)\n"
            "- Idle: Wait and do nothing\n\n"
            "The drop-off zone is at x=23, y=8 to y=15.\n"
            "Critically injured victims REQUIRE both agents (CarryObjectTogether).\n"
            "Mildly injured victims can be carried alone (CarryObject)."
        )

        user_prompt = f"""## Current Task
{self._current_task}

## State
{state_text}

## Previous Action
{self.previous_action if hasattr(self, 'previous_action') else 'None'}

Decide your next action. Choose the action that best progresses your current task.
If you need to reach a distant location, move in the direction that gets you closer.

Respond in JSON:
{{
  "action": "action_name",
  "target": "object_id_or_null",
  "reasoning": "brief explanation"
}}"""

        response = query_llm(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.5
        )

        return parse_json_response(response)

    def _llm_action_to_matrx(self, llm_response: dict, state: State) -> Tuple[str, Dict]:
        """Convert LLM action decision to MATRX action tuple."""
        action = llm_response.get('action', 'Idle')
        target = llm_response.get('target')

        if self._verbose:
            self._logger.info(
                f"LLM decision: {action} target={target} "
                f"reason={llm_response.get('reasoning', '')}"
            )

        # Movement actions
        if action in ('MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'):
            return action, {}

        # Door action
        if action == 'OpenDoorAction':
            if target:
                return 'OpenDoorAction', {'object_id': target, 'door_range': 1}
            return Idle.__name__, {'duration_in_ticks': 1}

        # Carry actions
        if action == 'CarryObject' and target:
            return CarryObject.__name__, {
                'object_id': target,
                'human_name': self._human_name
            }

        if action == 'CarryObjectTogether' and target:
            return CarryObjectTogether.__name__, {
                'object_id': target,
                'human_name': self._human_name
            }

        # Drop actions
        if action == 'Drop':
            return Drop.__name__, {'human_name': self._human_name}

        if action == 'DropObjectTogether':
            return DropObjectTogether.__name__, {'human_name': self._human_name}

        # Remove actions
        if action == 'RemoveObject' and target:
            return RemoveObject.__name__, {
                'object_id': target,
                'remove_range': 1
            }

        if action == 'RemoveObjectTogether' and target:
            from actions1.CustomActions import RemoveObjectTogether as ROT
            return ROT.__name__, {
                'object_id': target,
                'remove_range': 1,
                'human_name': self._human_name
            }

        # Default: Idle
        return Idle.__name__, {'duration_in_ticks': 1}

    ## HELPERS
    def _is_within_range(self, pos1: Tuple[int, int], pos2: Tuple[int, int], radius: int) -> bool:
        """Check if two positions are within a given radius using Chebyshev distance."""
        if pos1 is None or pos2 is None:
            return False
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= radius

    def _handle_intro(self, state):
        """Wait for human to start moving, then begin workflow."""
        for obj_id, obj_data in state.items():
            if isinstance(obj_data, dict) and obj_data.get('is_human_agent', False):
                current_loc = obj_data.get('location')
                if self._human_start_location is None:
                    self._human_start_location = current_loc
                    self._send_message(
                        'Hello! I am RescueBot powered by RescueAgent with modular architecture. '
                        'Together we will search and rescue victims. '
                        'I use LLM for Planning, Reasoning, and Communication. '
                        'Start moving when you are ready!',
                        'RescueBot'
                    )
                    return None, {}
                elif current_loc != self._human_start_location:
                    self._intro_done = True
                    return Idle.__name__, {'duration_in_ticks': 1}
                return None, {}

        # Human not visible — start anyway
        self._intro_done = True
        return Idle.__name__, {'duration_in_ticks': 1}

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

    def act(self, task: str, world) -> Dict[str, Any]:
        """
        Execute task in MATRX world by running multiple ticks.
        Called by the Engine to execute a planned task.
        """
        self.set_current_task(task)
        result = {
            'task': task,
            'status': 'in_progress',
            'ticks_executed': 0,
            'observations': [],
            'communications': []
        }

        if self._verbose:
            print(f"[RescueAgent.act] Starting task: {task}")

        # The actual execution happens through MATRX's tick loop
        # (via decide_on_actions being called each tick)
        # This method is used when running through the Engine directly
        result['status'] = 'delegated_to_tick_loop'
        result['communications'] = [m.content for m in self.messages_to_send] if hasattr(self, 'messages_to_send') else []

        return result
