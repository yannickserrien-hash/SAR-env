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
import random
import os
import yaml
from engine.toon_utils import to_toon
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from agents1.modules.ReasoningModule import ReasoningIO
from agents1.modules.PerceptionModule import PerceptionModule
from agents1.modules.CommunicationModule import CommunicationModule
from agents1.modules.PlanningModule import PlanningModule
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
        llm_model: str = 'llama3:8b',
        include_human: bool = True,
        ollama_port: int = 11434
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
        
        # Debug
        self._verbose = True
        self._logger = logging.getLogger('RescueAgent')
        
        # Modules
        self.reasoning_module = None
        self.communication_module = None
        self.planning_module = None
        self.short_term_memory = ShortTermMemory(memory_limit=20, llm_model=self._llm_model, api_url=self._api_url)
        
        # Planning state
        self._planning_in_progress = False
        self._completed_action_count = 0  # counts completed actions for comm gating
        self.PLAN = ''  # current plan from PlanningModule

        # Agent ↔ Planner communication
        self._planner_channel = None
        self._planner_responses: list = []  # recent planner response strings

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
            llm_model=self._llm_model,
            prompts=self._prompts,
            agent_id=self.agent_id,
            send_message_fn=self._send_message,
            api_url=self._api_url,
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
        """
            Set the current high-level task from the EnginePlanner.
        """
        # Normalise: EnginePlanner may pass a dict instead of a plain string
        if not isinstance(task, str):
            task = json.dumps(task, default=str)
        self._current_task = task

        # Record task assignment in memory (persists across tasks)
        tick = 0
        if self.state_for_navigation is not None:
            try:
                tick = self.state_for_navigation['World']['nr_ticks']
            except (KeyError, TypeError):
                pass
        self.short_term_memory.update('task', {
            'task': task,
            'tick': tick,
        })

        # Reset navigation and LLM state for new task (memory is NOT reset)
        self._nav_target = None
        self._reasoning_step = False
        with self._llm_lock:
            self._pending_llm_action = None
            self._last_llm_result = None

        # Reset planning state and trigger task decomposition
        self._completed_action_count = 0
        self._planning_in_progress = True
        self.PLAN = ''       

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
        # Configs
        self._state_tracker.update(self.state_for_navigation)
        
        # Process observation
        self.WORLD_STATE_FILTERED = self.process_observations(filtered_state)
        self.OBS = self.observation_to_dict(filtered_state)
        
        self._last_filtered_state = filtered_state
        
        if self._planning_in_progress and self.PLAN == '':
            self.planning_module.plan(
                task=self._current_task,
                world_state=to_toon(self.WORLD_STATE_FILTERED),
                memory=self.short_term_memory.get_compact_str()[:400] if self.short_term_memory.storage else '',
            )
            self._planning_in_progress = False
        
        # Check if PlanningModule generated a plan
        if self.planning_module.is_plan_ready():
            print(self.planning_module.get_plan)
            if "AskPlanner" in self.planning_module.get_plan:
                self._reasoning_step = False
                self.text_to_action(self.planning_module.get_plan)
            else:
                self._reasoning_step = True
                self.PLAN = self.planning_module.get_plan

        # If no task assigned yet, move to a random direction to explore
        if not self._current_task or not self.planning_module.plan_ready:
            action = random.choice(['MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'])
            return action, {}
        
        # Let Agent finish MoveTo() action
        if self._nav_target is not None:
            move_action = self._navigator.get_move_action(self._state_tracker)

            if move_action is not None: # Still moving to target
                print(f"[{self.agent_id}] Navigating to {self._nav_target} | "
                        f"action={move_action}")
                return move_action, {}
            else: # Stop navigation (arrived or path blocked)
                self._nav_target = None
                self._reasoning_step = True

        if self.communication_module is not None:
            self.communication_module.poll_outbound()
            self.communication_module.poll_inbound(
                self.received_messages,
                action_count=self._completed_action_count,
            )

        # Poll for planner responses (immediate — every tick)
        if self._planner_channel is not None:
            responses = self._planner_channel.poll_responses(self.agent_id)
            for resp in responses:
                self._planner_responses.append(resp.content)
                self._planner_responses = self._planner_responses[-3:]
                self.short_term_memory.update('planner_response', {
                    'type': 'planner_response',
                    'content': resp.content,
                    'tick': resp.tick,
                })
                self._reasoning_step = True  # re-reason with planner's advice

        # Get LLM action if reasoning step is done
        with self._llm_lock:
            if self._pending_llm_action is not None and self._pending_llm_action.done():
                try:
                    self._last_llm_result = self._pending_llm_action.result()
                except Exception as e:
                    self._logger.warning(f"[{self.agent_id}] LLM future raised: {e}")
                finally:
                    self._pending_llm_action = None

        if self._last_llm_result is not None:
            action = self._last_llm_result
            self._last_llm_result = None
            return self.text_to_action(action)

        # Submit new LLM call
        if self._reasoning_step:
            with self._llm_lock:
                if self._pending_llm_action is None and self.planning_module.plan_ready:
                    planner_ctx = ""
                    if self._planner_responses:
                        planner_ctx = "\n".join(
                            f"- {r}" for r in self._planner_responses[-2:]
                        )
                    self.PLAN = self.planning_module.get_plan.split("1.", 1)[1].strip()
                    self._pending_llm_action = self.reasoning_module(
                        self.PLAN,
                        observation=self.OBS,
                        previous_action=self.past_actions,
                        planner_context=planner_ctx,
                    )
        return Idle.__name__, {'duration_in_ticks': 1}

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
        
    def text_to_action(self, llm_response: str) -> Tuple[str, Dict]:
        """
            Convert LLM action decision to MATRX action.
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
            
        action = parsed_response.get('action', 'Idle')
        params = parsed_response.get('params') or {}
        object_id = params.get('object_id')

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
                    context="",
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
                self.communication_module.generate_direct_message(
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

