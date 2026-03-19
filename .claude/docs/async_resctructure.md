 Ready to code?                                                                           

 Here is Claude's plan:                                                                                                                          
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Minecraft-Style Client-Server Async Architecture                                                                                          
                                                                                                                                               
 Context                             

 The current MATRX game loop is synchronous — each tick blocks on every agent sequentially. Even though LLM calls are async internally, agents
 can't truly operate independently. The goal is a Minecraft-style architecture where a game server runs the world simulation and agent clients
 connect via WebSocket, acting at their own pace.

 Key shift: Agents become independent clients, not callbacks inside the tick loop.

 Architecture Overview

 ┌─────────────────────────────────────────────────┐
 │  GAME SERVER (single process)                    │
 │                                                  │
 │  ┌──────────────┐  ┌───────────────────────────┐ │
 │  │ WorldExecutor │  │ EnginePlanner             │ │
 │  │ (action queue │  │ (task gen, summarize,     │ │
 │  │  validate     │  │  Q&A — all async via      │ │
 │  │  mutate)      │  │  ThreadPoolExecutor)      │ │
 │  └──────┬───────┘  └───────────┬───────────────┘ │
 │         │                      │                  │
 │  ┌──────┴──────────────────────┴───────────────┐ │
 │  │          WebSocket Server (port 3002)        │ │
 │  │  - per-agent channels                       │ │
 │  │  - state push, action receive               │ │
 │  │  - message relay, planner Q&A relay         │ │
 │  └──────────────────────────────────────────────┘ │
 │                                                  │
 │  ┌──────────────────────────────────────────────┐ │
 │  │  VizTimer → existing Flask API (port 3001)   │ │
 │  │  (publishes snapshots for browser GUI)       │ │
 │  └──────────────────────────────────────────────┘ │
 └──────────────────┬──────────────────────────────┘
                    │ WebSocket (ws://localhost:3002)
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Agent 0 │ │ Agent 1 │ │ Agent N │
    │ (own    │ │ (own    │ │ (own    │
    │ process │ │ process │ │ process │
    │ or thrd)│ │ or thrd)│ │ or thrd)│
    └─────────┘ └─────────┘ └─────────┘
    Each has own Ollama port, runs LLM independently

 ---
 WebSocket Protocol

 All messages are JSON with a type field.

 Server → Agent

 ┌──────────────────┬──────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
 │       Type       │                       Payload                        │                      When                      │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ welcome          │ {agent_id, capabilities, capability_knowledge, task} │ On registration                                │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ state_update     │ {state, tick}                                        │ After agent's action executes, or periodically │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ action_result    │ {action_name, result, succeeded}                     │ After action validation+execution              │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ task_assignment  │ {task, plans?}                                       │ When planner assigns new task                  │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ message          │ {from_id, content, type}                             │ When another agent sends a message             │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ planner_response │ {msg_id, content}                                    │ Answer to agent's planner question             │
 ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
 │ simulation_end   │ {reason, score}                                      │ Game over                                      │
 └──────────────────┴──────────────────────────────────────────────────────┴────────────────────────────────────────────────┘

 Agent → Server

 ┌──────────────────┬──────────────────────────────────┬────────────────────────────────┐
 │       Type       │             Payload              │              When              │
 ├──────────────────┼──────────────────────────────────┼────────────────────────────────┤
 │ register         │ {agent_id, agent_type}           │ On connect                     │
 ├──────────────────┼──────────────────────────────────┼────────────────────────────────┤
 │ action           │ {action_name, action_kwargs}     │ Agent submits decided action   │
 ├──────────────────┼──────────────────────────────────┼────────────────────────────────┤
 │ message          │ {content, send_to, message_type} │ Inter-agent communication      │
 ├──────────────────┼──────────────────────────────────┼────────────────────────────────┤
 │ planner_question │ {content, context}               │ Ask planner a question         │
 ├──────────────────┼──────────────────────────────────┼────────────────────────────────┤
 │ request_state    │ {}                               │ Explicitly request fresh state │
 └──────────────────┴──────────────────────────────────┴────────────────────────────────┘

 ---
 Files to Create/Modify

 ┌───────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
 │               File                │                                      Action                                      │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ server/game_server.py             │ Create — Main server: WebSocket + WorldExecutor + PlannerRunner + VizTimer       │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ server/ws_handler.py              │ Create — WebSocket connection handler, message routing                           │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ server/action_queue.py            │ Create — Thread-safe action queue                                                │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ server/planner_runner.py          │ Create — EnginePlanner state machine (extracted from run_with_planner)           │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ agents1/agent_client.py           │ Create — WebSocket client base class for agents                                  │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ agents1/ws_search_rescue_agent.py │ Create — WebSocket-based SearchRescueAgent (or adapt existing)                   │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ matrx/grid_world.py               │ Modify — expose public wrappers for __perform_action, __get_complete_state, etc. │
 ├───────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ main.py                           │ Modify — add async_mode flag to launch server + agent clients                    │
 └───────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘

 Unchanged: engine/engine_planner.py, actions1/CustomActions.py, agents1/tool_registry.py, memory/, worlds1/

 ---
 Step 1: Create server/action_queue.py

 Thread-safe queue for agent actions, bridging WebSocket handler → WorldExecutor.

 from threading import Lock
 from typing import Any, Dict, Optional, Tuple
 from collections import OrderedDict

 class ActionQueue:
     def __init__(self):
         self._actions: OrderedDict[str, Tuple[str, Dict]] = OrderedDict()
         self._lock = Lock()

     def submit(self, agent_id: str, action_name: str, action_kwargs: Dict[str, Any]):
         with self._lock:
             self._actions[agent_id] = (action_name, action_kwargs)

     def pop_all(self) -> OrderedDict:
         with self._lock:
             ready = self._actions
             self._actions = OrderedDict()
             return ready

 ---
 Step 2: Create server/ws_handler.py

 WebSocket connection manager using websockets library (or flask-socketio).

 class WSHandler:
     """Manages per-agent WebSocket connections."""

     def __init__(self, action_queue, planner_channel):
         self._connections: Dict[str, WebSocket] = {}  # agent_id → ws
         self._action_queue = action_queue
         self._planner_channel = planner_channel

     async def handle_connection(self, websocket):
         """Handle one agent's WebSocket lifecycle."""
         agent_id = await self._handle_register(websocket)
         self._connections[agent_id] = websocket
         try:
             async for raw_msg in websocket:
                 msg = json.loads(raw_msg)
                 await self._route_message(agent_id, msg)
         finally:
             del self._connections[agent_id]

     async def _route_message(self, agent_id, msg):
         msg_type = msg['type']
         if msg_type == 'action':
             self._action_queue.submit(agent_id, msg['action_name'], msg['action_kwargs'])
         elif msg_type == 'message':
             self._relay_agent_message(agent_id, msg)
         elif msg_type == 'planner_question':
             self._planner_channel.submit_question(agent_id, msg['content'], ...)
         elif msg_type == 'request_state':
             await self._send_state(agent_id)

     # --- Server → Agent push methods (called by WorldExecutor) ---

     async def send_state_update(self, agent_id, state, tick):
         ws = self._connections.get(agent_id)
         if ws:
             await ws.send(json.dumps({'type': 'state_update', 'state': state, 'tick': tick}))

     async def send_action_result(self, agent_id, action_name, result):
         ...

     async def send_task_assignment(self, agent_id, task, plans=None):
         ...

     async def broadcast(self, msg):
         for ws in self._connections.values():
             await ws.send(json.dumps(msg))

 ---
 Step 3: Create server/planner_runner.py

 Extracts the EnginePlanner state machine from run_with_planner (grid_world.py lines 344-484) into a standalone class.

 class PlannerRunner:
     """Runs the EnginePlanner iteration state machine."""

     PHASES = ('needs_planning', 'planning', 'executing',
               'needs_summarization', 'summarizing')

     def __init__(self, planner, agents_meta, grid_world, ws_handler,
                  ticks_per_iteration=1200):
         self._planner = planner
         self._agents_meta = agents_meta  # {agent_id: agent_info}
         self._gw = grid_world
         self._ws = ws_handler
         self._ticks_per_iteration = ticks_per_iteration
         self._phase = 'needs_planning'
         self._iteration = 0
         self._actions_in_iteration = 0
         self._planning_future = None
         self._summary_future = None
         self._iteration_data = None

     def advance(self):
         """Non-blocking: check current phase, advance if ready.
         Called by WorldExecutor each loop iteration."""

         if self._phase == 'needs_planning':
             self._start_planning()

         elif self._phase == 'planning':
             self._check_planning_done()

         elif self._phase == 'executing':
             if self._actions_in_iteration >= self._ticks_per_iteration:
                 self._phase = 'needs_summarization'

         elif self._phase == 'needs_summarization':
             self._start_summarization()

         elif self._phase == 'summarizing':
             self._check_summarization_done()

         # Always process agent questions (non-blocking)
         self._planner.process_agent_questions()

     def on_action_executed(self):
         """Called by WorldExecutor after each action."""
         self._actions_in_iteration += 1

     def _start_planning(self):
         from engine.iteration_data import IterationData
         self._iteration_data = IterationData(iteration=self._iteration)
         self._actions_in_iteration = 0
         world_state = self._gw.get_complete_state()
         world_map = self._gw.process_map_to_dict(world_state)
         self._planner.set_world_state(world_map)
         self._planning_future = self._planner.submit_generate_tasks(
             list(self._agents_meta.values()))
         self._phase = 'planning'

     def _check_planning_done(self):
         if not self._planning_future or not self._planning_future.done():
             return
         try:
             task_assignments = self._planning_future.result()
         except Exception as e:
             print(f"[PlannerRunner] Planning error: {e}")
             task_assignments = {'tasks': {}}
         self._planning_future = None
         self._iteration_data.task_assignments = task_assignments

         # Distribute tasks via WebSocket
         agent_tasks = task_assignments.get('tasks', {})
         for agent_id, task in agent_tasks.items():
             # Push task to connected agent
             self._ws.send_task_assignment(agent_id, task,
                 task_assignments.get('plans', {}).get(agent_id))

         self._phase = 'executing'

     def _start_summarization(self):
         world_state = self._gw.get_complete_state()
         world_map = self._gw.process_map_to_dict(world_state)
         self._summary_future = self._planner.submit_summarize(
             self._iteration_data, world_map)
         self._phase = 'summarizing'

     def _check_summarization_done(self):
         if not self._summary_future or not self._summary_future.done():
             return
         try:
             summary = self._summary_future.result()
         except Exception:
             summary = "Summary unavailable"
         self._summary_future = None
         self._iteration_data.summary = summary

         should_continue = self._planner.decide_next_step(self._iteration_data)
         self._planner.iteration_history.append(self._iteration_data)

         if not should_continue:
             self._ws.broadcast({'type': 'simulation_end',
                                 'reason': 'planner_terminated'})
             return  # Signal stop to WorldExecutor

         self._planner.update_progress(self._iteration_data)
         self._iteration += 1
         self._phase = 'needs_planning'

 Key planner integration preserved:
 - planner.submit_generate_tasks() — async task generation
 - planner.submit_summarize() — async summarization + prefetch
 - planner.process_agent_questions() — Q&A relay (every loop iteration)
 - planner.set_world_state() — world state for Q&A context
 - planner.decide_next_step() — termination check
 - Task distribution now via WebSocket send_task_assignment instead of agent.set_current_task()
 - Manual plans (manual_plans_file) work unchanged via planner._build_manual_tasks()

 ---
 Step 4: Create server/game_server.py

 Main server that ties everything together.

 class GameServer:
     """Minecraft-style game server for SAR simulation."""

     def __init__(self, grid_world, planner=None, agents_meta=None,
                  ws_port=3002, ticks_per_iteration=1200):
         self._gw = grid_world
         self._action_queue = ActionQueue()
         self._ws_handler = WSHandler(self._action_queue, planner_channel)
         self._planner_runner = PlannerRunner(planner, agents_meta,
             grid_world, self._ws_handler, ticks_per_iteration) if planner else None
         self._viz_timer = VizTimer(grid_world)  # pushes to existing Flask API
         self._ws_port = ws_port
         self._stop = Event()

     def run(self):
         """Start all server components."""
         # Start WebSocket server in its own thread
         ws_thread = Thread(target=self._run_ws_server, daemon=True)
         ws_thread.start()

         # Start viz timer
         self._viz_timer.start()

         # Main loop: process actions + advance planner
         self._world_executor_loop()

         # Cleanup
         self._viz_timer.stop()

     def _world_executor_loop(self):
         """Process action queue + advance planner state machine."""
         while not self._stop.is_set():
             # Advance planner (non-blocking)
             if self._planner_runner:
                 self._planner_runner.advance()

             # Process ready actions
             ready = self._action_queue.pop_all()
             if not ready:
                 time.sleep(0.02)
                 continue

             for agent_id, (action_name, action_kwargs) in ready.items():
                 # Run agent's brain-level preprocessing
                 # (ArtificialBrain.decide_on_action wraps with duration/capability)
                 # In client-server model, this happens client-side.
                 # Server only does: validate + mutate.

                 world_state = self._gw.get_complete_state()
                 result = self._gw.perform_action(agent_id, action_name, action_kwargs)

                 # Push result to agent
                 self._ws_handler.send_action_result(agent_id, action_name, result)

                 # Push fresh state to this agent
                 agent_obj = self._gw.get_agent(agent_id)
                 new_state = self._gw.get_agent_state(agent_obj)
                 filtered = agent_obj.filter_observations(new_state)
                 self._ws_handler.send_state_update(agent_id, filtered, ...)

                 # Track for planner
                 if self._planner_runner:
                     self._planner_runner.on_action_executed()

             # Deliver inter-agent messages (from queue)
             self._process_messages()

             # Check goal
             if self._gw.check_goal():
                 self._ws_handler.broadcast({'type': 'simulation_end', 'reason': 'goal_reached'})
                 self._stop.set()

 ---
 Step 5: Create agents1/agent_client.py

 WebSocket client base class — replaces the MATRX callback-based agent lifecycle.

 class AgentClient:
     """Base class for WebSocket-connected agent clients."""

     def __init__(self, agent_id, ws_url='ws://localhost:3002',
                  llm_model='ollama/llama3', api_base=None):
         self.agent_id = agent_id
         self._ws_url = ws_url
         self._ws = None  # WebSocket connection
         self._current_task = None
         self._latest_state = None
         self._action_result = None
         self._received_messages = []

         # Reuse existing infrastructure
         self.memory = BaseMemory()
         self.planner = Planning(mode='simple')
         self._llm_model = llm_model
         self._api_base = api_base

     async def run(self):
         """Main agent loop — connect and run indefinitely."""
         async with websockets.connect(self._ws_url) as ws:
             self._ws = ws

             # Register
             await ws.send(json.dumps({
                 'type': 'register', 'agent_id': self.agent_id}))

             # Start listener in background
             listener = asyncio.create_task(self._listen())

             # Main decision loop
             while True:
                 # Wait for state update
                 state = await self._wait_for_state()
                 if state is None:
                     break  # Simulation ended

                 # Decide (may involve multiple LLM calls — takes as long as needed)
                 action_name, action_kwargs = await self.decide(state)

                 # Submit action
                 await ws.send(json.dumps({
                     'type': 'action',
                     'action_name': action_name,
                     'action_kwargs': action_kwargs,
                 }))

                 # Wait for action result + new state (server pushes both)

     async def _listen(self):
         """Background task: route incoming server messages."""
         async for raw in self._ws:
             msg = json.loads(raw)
             if msg['type'] == 'state_update':
                 self._latest_state = msg['state']
                 self._state_event.set()
             elif msg['type'] == 'action_result':
                 self._action_result = msg
             elif msg['type'] == 'task_assignment':
                 self._current_task = msg['task']
             elif msg['type'] == 'message':
                 self._received_messages.append(msg)
             elif msg['type'] == 'simulation_end':
                 self._stop = True
                 self._state_event.set()

     async def decide(self, state) -> Tuple[str, Dict]:
         """Override in subclass. Run full reasoning pipeline."""
         raise NotImplementedError

 ---
 Step 6: Create agents1/ws_search_rescue_agent.py

 Adapts the existing SearchRescueAgent pipeline to the WebSocket client model.

 class WSSearchRescueAgent(AgentClient):
     """WebSocket-based SearchRescueAgent with multi-stage LLM pipeline."""

     def __init__(self, agent_id, capabilities=None, **kwargs):
         super().__init__(agent_id, **kwargs)
         self._capabilities = capabilities
         # Reuse existing modules
         self._validator = ActionValidator(capabilities, 'informed')
         # ... tool registry, perception, etc.

     async def decide(self, state) -> Tuple[str, Dict]:
         """Run the full CRITIC→PLANNING→COMM→REASONING→EXECUTE pipeline.

         Unlike the tick-based version, this runs all stages sequentially
         in one call — no need to return Idle between stages.
         """
         # Stage 1: Critic (evaluate previous action)
         critic_result = await self._run_llm_stage('critic', state)

         # Stage 2: Planning (decompose task)
         plan_result = await self._run_llm_stage('planning', state)

         # Stage 3: Communication (check messages, negotiate)
         comm_result = await self._run_llm_stage('communication', state)

         # Stage 4: Reasoning (choose action)
         action_name, action_kwargs = await self._run_llm_stage('reasoning', state)

         # Stage 5: Validate
         check = self._validator.validate(action_name, action_kwargs, state, ...)
         if not check.valid:
             self.memory.update('action_failure', check.feedback)
             return 'Idle', {'duration_in_ticks': 1}

         return action_name, action_kwargs

     async def _run_llm_stage(self, stage, state):
         """Submit LLM call and await result (blocking for this agent, not the server)."""
         messages = self._build_prompt(stage, state)
         future = submit_llm_call(self._llm_model, messages, ...)
         # Block until LLM responds (this is fine — we're in our own thread/process)
         while not future.done():
             await asyncio.sleep(0.05)
         return future.result()

 Key advantage: The pipeline no longer needs the idle-and-poll pattern. Each LLM call can simply await its result since the agent is independent
  from the server.

 ---
 Step 7: Modify matrx/grid_world.py — Expose Public Wrappers

 # Add public methods that WorldExecutor/GameServer can call:

 def perform_action(self, agent_id, action_name, action_kwargs):
     world_state = self.__get_complete_state()
     return self.__perform_action(agent_id, action_name, action_kwargs, world_state)

 def get_agent_state(self, agent_obj):
     return self.__get_agent_state(agent_obj)

 def get_complete_state(self):
     return self.__get_complete_state()

 def check_goal(self):
     world_state = self.__get_complete_state()
     return self.__check_simulation_goal(world_state)

 def update_grid(self):
     return self.__update_grid()

 def get_agent(self, agent_id):
     return self.__registered_agents.get(agent_id)

 @property
 def registered_agents(self):
     return self.__registered_agents

 Keep __step(), run(), run_with_planner() intact as legacy sync path.

 ---
 Step 8: Modify main.py

 # New config
 async_mode = True  # Enable client-server architecture

 if async_mode:
     from server.game_server import GameServer

     # Build world (same as before)
     builder = create_builder(...)
     grid_world = builder.build()

     # Create server
     server = GameServer(
         grid_world=grid_world,
         planner=planner,
         agents_meta=agent_configs,
         ws_port=3002,
         ticks_per_iteration=ticks_per_iteration,
     )

     # Launch agent clients (each in own thread or subprocess)
     for i, config in enumerate(agent_configs):
         agent = WSSearchRescueAgent(
             agent_id=config['agent_id'],
             capabilities=config['capabilities'],
             llm_model=config['llm_model'],
             api_base=f'http://localhost:{ollama_base_port + i}',
             ws_url='ws://localhost:3002',
         )
         Thread(target=asyncio.run, args=(agent.run(),), daemon=True).start()

     # Start server (blocks until simulation ends)
     server.run()
 else:
     # Legacy synchronous path
     builder.run_with_planner(...)

 ---
 Step 9: Action Duration in Client-Server Model

 In the sync model, ArtificialBrain.decide_on_action() sets action_duration in kwargs, and the tick loop marks the agent "busy" for N ticks.

 In the client-server model, action duration is handled server-side:
 - After executing an action, the server checks action_duration
 - If duration > 0: delays the state push to the agent by duration * tick_interval seconds
 - The agent blocks on _wait_for_state() during this time — naturally simulating the "busy" period
 - Duration values (RemoveObject stone=200, tree=100, CarryObject mild=150) are set in ArtificialBrain.decide_on_action() — this logic moves to
 server-side or the agent sends duration hints with the action

 ---
 Step 10: Cooperative Actions

 CarryObjectTogether / RemoveObjectTogether flow:
 1. Agent A submits action: CarryObjectTogether(object_id) via WebSocket
 2. Server's WorldExecutor runs is_possible() — checks if partner is adjacent
 3. If partner NOT adjacent: Server sends action_result: {succeeded: false, reason: "partner not adjacent"}
 4. Agent A receives failure, sends message: {send_to: "AgentB", type: "ask_help", content: "help carry victim_1"}
 5. Server relays message to Agent B
 6. Agent B navigates to victim, then submits CarryObjectTogether(object_id)
 7. Server processes Agent B's action — partner (A) is now adjacent → succeeds
 8. Server triggers carry autopilot (both agents navigate to drop zone)

 The existing SharedMemory rendezvous mechanism needs to work through WebSocket messages instead of direct memory access. Agent messages
 (ask_help, help) already exist in the protocol.

 ---
 Step 11: Thread Safety

 ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────┬────────────────────────────────┐
 │         Resource         │                                Thread                                │             Safety             │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ World grid state         │ WorldExecutor (sole writer)                                          │ Safe — single thread           │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ ActionQueue              │ WS handler (write), WorldExecutor (read)                             │ Lock-protected                 │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ Agent state (per client) │ Own thread/process                                                   │ Safe — isolated                │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ SharedMemory             │ Not needed — agents communicate via WebSocket messages               │ N/A                            │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ Flask API state          │ VizTimer (write), Flask (read)                                       │ Same as current                │
 ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┼────────────────────────────────┤
 │ EnginePlanner            │ PlannerRunner calls (single thread) + ThreadPoolExecutor (LLM calls) │ Safe — same pattern as current │
 └──────────────────────────┴──────────────────────────────────────────────────────────────────────┴────────────────────────────────┘

 ---
 Step 12: Verification

 1. Regression: async_mode=False → legacy sync path, behavior unchanged
 2. 1 agent: Single agent connects, receives state, submits actions, sees results
 3. 2 agents: Both connect independently, act at own pace, don't block each other
 4. Planner: Tasks generated and distributed via WebSocket task_assignment
 5. Cooperative carry: Two agents coordinate via messages, CarryObjectTogether succeeds
 6. Manual plans: manual_plans_file distributes tasks correctly via WebSocket
 7. Visualization: http://localhost:3000 shows smooth updates via VizTimer
 8. Agent disconnect/reconnect: Server handles gracefully
 9. Planner Q&A: Agent asks question, gets answer via planner_question/planner_response

 ---
 Implementation Order

 1. Phase 1 — Server foundation: action_queue.py, public wrappers in grid_world.py
 2. Phase 2 — WebSocket: ws_handler.py with websockets library
 3. Phase 3 — Planner extraction: planner_runner.py (extract from run_with_planner)
 4. Phase 4 — Game server: game_server.py ties it all together
 5. Phase 5 — Agent client: agent_client.py base + ws_search_rescue_agent.py
 6. Phase 6 — Integration: Update main.py, test with 1 agent
 7. Phase 7 — Multi-agent: Test cooperative actions, planner, messages

 Dependencies

 - websockets library (add to requirements.txt) — async WebSocket server/client
 - Everything else already exists in the project

 ---
 How to Host & Run

 Option A: Everything on one machine (development)

 Same as today but with a different startup flow:

 # Terminal 1: Start Ollama instances (same as before)
 ollama serve                                     # port 11434
 OLLAMA_HOST=0.0.0.0:11435 ollama serve           # port 11435

 # Terminal 2: Start the game server
 python main.py --mode server
 # This starts:
 #   - World simulation (WorldExecutor)
 #   - WebSocket server on ws://localhost:3002
 #   - EnginePlanner (task generation, summarization)
 #   - VizTimer → Flask API on http://localhost:3001
 #   - Visualizer on http://localhost:3000

 # Terminal 3: Start Agent 0
 python main.py --mode agent --agent-id RescueAgent0 --ollama-port 11434 --ws ws://localhost:3002

 # Terminal 4: Start Agent 1
 python main.py --mode agent --agent-id RescueAgent1 --ollama-port 11435 --ws ws://localhost:3002

 What changes vs today:
 - Instead of python main.py (one command), you start server + agents separately
 - Each agent is its own process with its own Ollama port
 - Agents can be started/stopped independently without restarting the server
 - The server keeps running even if an agent disconnects

 Option B: Server on a remote machine (distributed)

 # On the server machine (has good CPU, runs world sim):
 python main.py --mode server --ws-host 0.0.0.0 --ws-port 3002

 # On machine with GPU 1 (runs Ollama + Agent 0):
 ollama serve
 python main.py --mode agent --agent-id RescueAgent0 \
     --ollama-port 11434 \
     --ws ws://server-ip:3002

 # On machine with GPU 2 (runs Ollama + Agent 1):
 ollama serve
 python main.py --mode agent --agent-id RescueAgent1 \
     --ollama-port 11434 \
     --ws ws://server-ip:3002

 What this enables:
 - Each agent runs on its own machine with its own GPU
 - LLM inference is fully parallel (no port juggling)
 - Server machine only runs the lightweight world simulation
 - Visualization accessible at http://server-ip:3000
 - Agents can be on different networks (as long as they can reach the server WebSocket)

 Option C: Docker Compose (production-like)

 # docker-compose.yml
 services:
   game-server:
     build: .
     command: python main.py --mode server --ws-host 0.0.0.0
     ports:
       - "3000:3000"   # Visualizer
       - "3001:3001"   # Flask API
       - "3002:3002"   # WebSocket

   agent-0:
     build: .
     command: python main.py --mode agent --agent-id RescueAgent0 --ws ws://game-server:3002
     depends_on: [game-server, ollama-0]

   agent-1:
     build: .
     command: python main.py --mode agent --agent-id RescueAgent1 --ws ws://game-server:3002
     depends_on: [game-server, ollama-1]

   ollama-0:
     image: ollama/ollama
     ports: ["11434:11434"]

   ollama-1:
     image: ollama/ollama
     ports: ["11435:11434"]

 Config changes in main.py

 # Top-level config (replaces current agent_type, num_rescue_agents, etc.)
 mode = 'server'          # 'server' | 'agent' | 'legacy' (sync, same as today)

 # Server mode config
 ws_host = '0.0.0.0'      # Bind address for WebSocket
 ws_port = 3002            # WebSocket port
 # All existing planner config stays the same

 # Agent mode config
 agent_id = 'RescueAgent0' # Which agent this process is
 ws_url = 'ws://localhost:3002'
 ollama_port = 11434        # This agent's Ollama port (no more port arithmetic)
 agent_preset = 'generalist'

 Key difference from today: Instead of num_rescue_agents = 3 spawning 3 agents inside one process, you start 3 separate agent processes. The
 server doesn't need to know how many agents will connect — they join dynamically like Minecraft players.

 Backward compatibility

 mode = 'legacy' (or just python main.py without flags) runs the existing synchronous tick loop via run_with_planner(). Nothing changes for anyone who doesn't opt in.