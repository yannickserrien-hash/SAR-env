 Plan: Integrate CrewAI as the Multi-Agent System                                                                                          
                                                                  
 Context                             

 The current SAR-env uses a two-tier LLM system: EnginePlanner (task coordination) + SearchRescueAgent (per-tick 4-stage cognitive
 pipeline), both tightly coupled to MATRX's synchronous tick loop. The goal is to replace both tiers with CrewAI, using a client-server
 architecture where MATRX becomes a pure game server and CrewAI agents connect as clients. LLM backend must be configurable (Ollama +
 cloud APIs).

 Architecture

 MATRX Game Server (process 1)              CrewAI Orchestration (process 2)
 ┌──────────────────────────────┐           ┌──────────────────────────────────┐
 │ WorldExecutor                │           │ CrewAI Crew (hierarchical)       │
 │   - action queue             │           │   Manager Agent (= planner)      │
 │   - validate + mutate grid   │    WS     │   Worker Agent 0 (scout)         │
 │   - capability enforcement   │◄────────►│   Worker Agent 1 (medic)         │
 │                              │           │   ...                            │
 │ WebSocket Server (:3002)     │           │                                  │
 │ Flask Visualizer (:3000)     │           │ GameBridge (WS client)           │
 └──────────────────────────────┘           │ CrewAI Tools → bridge.execute()  │
                                            └──────────────────────────────────┘

 Key decisions:
 - Single CrewAI process manages all agents (not one process per agent) — simplifies cooperative coordination
 - GameBridge = shared WebSocket client connecting CrewAI tools to the game server
 - Navigation is server-side — MoveTo(x,y) is a macro-action; server does A* pathfinding and returns when done
 - Cooperative actions use server-side wait/retry (mirrors existing _handle_carry_retry pattern)
 - Iteration lifecycle = simple Python loop calling crew.kickoff() per iteration (can evolve to CrewAI Flow later)

 Implementation Phases

 Phase 1: Game Server Infrastructure

 Create the server that runs MATRX independently and accepts WebSocket connections.

 New files:
 - server/__init__.py
 - server/action_queue.py — Thread-safe OrderedDict with Lock (submit(), pop_all())
 - server/ws_handler.py — WebSocket connection manager using websockets library. Per-agent channels, message routing (register, action,
 request_state, message). Push methods (send_state_update, send_action_result, broadcast)
 - server/game_server.py — Ties together: GridWorld, ActionQueue, WSHandler, VizTimer. Main loop: pop actions → validate → execute → push
 results. Handles action duration (busy timers), capability enforcement (reuse logic from brains1/ArtificialBrain.py:629-677), and
 server-side A* navigation (reuse from agents1/llm_agent_base.py Navigator usage)

 Modify:
 - matrx/grid_world.py — Add public wrappers: perform_action(), get_complete_state(), get_agent_state(), check_goal(), registered_agents
 property. Existing sync path unchanged.

 Protocol: Follow spec from .claude/docs/async_resctructure.md (lines 53-89)

 Phase 2: GameBridge + CrewAI Tools

 Bridge layer connecting CrewAI to the game server.

 New files:
 - crewai_sar/__init__.py
 - crewai_sar/game_bridge.py — WebSocket client singleton. Background asyncio loop listens for messages, dispatches to per-agent state
 buffers. Synchronous execute_action(agent_id, action_name, kwargs) → dict for tools to call. Also get_state(), send_agent_message(),
 get_messages()
 - crewai_sar/tools.py — Per-agent tool instances (agent_id baked in via closure). High-level tools:
   - observe_world() — returns formatted state
   - navigate_to(x, y), navigate_to_area(area), navigate_to_drop_zone(), enter_area(area)
   - carry_victim(object_id), carry_victim_together(object_id), drop_victim()
   - remove_obstacle(object_id), remove_obstacle_together(object_id)
   - send_message(message, send_to, message_type), check_messages()
   - check_score(), idle(ticks)

 Reuse: Tool names/descriptions from agents1/tool_registry.py. State formatting from agents1/modules/perception_module.py.

 Phase 3: CrewAI Agents + Crew

 Define the CrewAI agents and crew structure.

 New files:
 - crewai_sar/llm_config.py — LLMConfig dataclass + create_llm() factory supporting ollama/openai/anthropic via CrewAI's LLM class (which
 uses LiteLLM internally)
 - crewai_sar/agents.py — create_manager_agent() (replaces EnginePlanner, uses env info from worlds1/environment_info.py),
 create_worker_agent() (per rescue agent, backstory built from agents1/capabilities.py:get_capability_prompt() + get_game_rules())
 - crewai_sar/tasks.py — Task templates: create_exploration_task(), create_rescue_task(), create_planning_task(), create_summary_task()
 - crewai_sar/crew.py — SARCrew class: builds crew, connects bridge, registers agents, runs iteration loop:
 while not done:
     manager plans → workers execute (crew.kickoff) → check score → repeat

 Reuse directly:
 - agents1/capabilities.py — CAPABILITY_PRESETS, resolve_capabilities(), get_capability_prompt(), get_game_rules()
 - worlds1/environment_info.py — EnvironmentInformation for area descriptions in agent backstories

 Phase 4: Cooperative Actions

 Server-side wait/retry for CarryObjectTogether and RemoveObjectTogether.

 - When tool calls carry_victim_together, bridge sends action to server
 - Server checks if partner adjacent → if not, returns failure with reason
 - Tool returns descriptive failure to CrewAI agent → agent uses send_message(type='ask_help')
 - Manager delegates help task to another agent → partner navigates and calls carry_victim_together
 - Server detects both adjacent → executes cooperative action → both agents get success result
 - Server handles carry autopilot (both navigate to drop zone)

 Phase 5: Entry Point Integration

 Modify main.py:
 - Add mode = 'crewai' config option ('legacy' | 'crewai')
 - 'crewai': Build world → start GameServer in thread → build SARCrew → crew.run()
 - 'legacy': Existing sync path (unchanged)
 - Add LLM config options: planner_llm_config, agent_llm_config with provider/model/api_key fields

 New dependency in requirements.txt:
 crewai>=0.80.0
 crewai-tools>=0.14.0
 websockets>=12.0

 Files Summary

 ┌────────┬───────────────────────────┬────────────────────────────────┐
 │ Action │           File            │            Purpose             │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ server/__init__.py        │ Package                        │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ server/action_queue.py    │ Thread-safe action queue       │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ server/ws_handler.py      │ WebSocket connection manager   │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ server/game_server.py     │ Main game server               │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/__init__.py    │ Package                        │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/game_bridge.py │ WS client bridge for CrewAI    │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/tools.py       │ CrewAI tool definitions        │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/llm_config.py  │ LLM backend configuration      │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/agents.py      │ CrewAI agent factories         │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/tasks.py       │ Task templates                 │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Create │ crewai_sar/crew.py        │ Crew assembly + iteration loop │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Modify │ matrx/grid_world.py       │ Add public wrappers            │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Modify │ main.py                   │ Add mode='crewai' path         │
 ├────────┼───────────────────────────┼────────────────────────────────┤
 │ Modify │ requirements.txt          │ Add crewai, websockets         │
 └────────┴───────────────────────────┴────────────────────────────────┘

 Reusable Existing Code

 - agents1/capabilities.py — Presets, prompt gen, game rules (used directly)
 - worlds1/environment_info.py — Area descriptions (used directly)
 - worlds1/WorldBuilder.py — World building (unchanged)
 - agents1/tool_registry.py — Tool names/descriptions (reference for CrewAI tools)
 - agents1/modules/perception_module.py — State formatting logic (adapted)
 - agents1/llm_agent_base.py — A* nav logic (extracted to server)
 - brains1/ArtificialBrain.py — Capability enforcement logic (extracted to server)
 - actions1/CustomActions.py — MATRX action classes (unchanged)
 - SaR_gui/ — Visualizer (unchanged)

 Verification

 1. Server standalone: WS client script → register → send action → verify state_update + action_result
 2. Single tool: CrewAI observe_world() returns formatted state via bridge
 3. Single agent: One worker explores area 1, reports findings
 4. Multi-iteration: Manager plans → workers execute → re-plan cycle works
 5. Cooperative carry: Two agents coordinate critical victim rescue
 6. Full mission: All 8 victims rescued, score reaches 1.0
 7. Regression: mode='legacy' runs unchanged sync path
 8. Cloud LLM: Test with OpenAI/Anthropic API key to verify flexible backend