# System Architecture & Initialization Flow

## Overview

SAR-env is a search-and-rescue simulation combining MATRX's grid-world environment with LLM-powered agents. The system operates in discrete **iterations** (each 1200 ticks / 2 minutes), with an overarching **EnginePlanner** coordinating multi-agent task assignments via background LLM calls that never block the simulation loop. All LLM calls route through a unified async path (`agents1/async_model_prompting.py`) using `litellm.completion()` with per-agent Ollama port routing.

## Component Relationships

**Core Components:**
- **main.py**: Entry point, configuration hub, initialization orchestrator
- **WorldBuilder** (worlds1/WorldBuilder.py): MATRX world setup, agent/object placement
- **GridWorld** (matrx/grid_world.py): Tick loop executor, state manager, simulation runtime with 5-phase state machine
- **EnginePlanner** (engine/engine_planner.py): Task coordinator via LLM (Ollama), termination controller
- **SearchRescueAgent** (agents1/search_rescue_agent.py): Individual agent brain with per-tick LLM reasoning

**Supporting Systems:**
- **async_model_prompting** (agents1/async_model_prompting.py): Unified LLM interface, ThreadPoolExecutor, async/sync APIs
- **toon_utils** (engine/toon_utils.py): TOON format encoder for 30-60% token reduction vs JSON
- **SharedMemory** (memory/shared_memory.py): Thread-safe cross-agent communication store
- **PlannerChannel** (engine/planner_channel.py): Bidirectional agent ↔ planner Q&A messaging

## Initialization Sequence

1. **Configuration (main.py)**: Load parameters (agent_type='marble', ticks_per_iteration=1200, num_rescue_agents=2, planner_model='qwen3:8b', planning_mode='dag', manual_plans_file)
2. **LLM Pool Init**: `init_marble_pool(num_rescue_agents)` creates ThreadPoolExecutor with `max(8, num_agents*3)` workers for non-blocking LLM calls
3. **World Creation**: `create_builder()` returns WorldBuilder + agent list; builder configures 25×24 grid with 7 rooms, victims, obstacles, drop zone (x=23, y=8-15)
4. **World Startup**: `builder.startup()` + `visualization_server.run_matrx_visualizer()` launches MATRX API and custom Flask UI
5. **EnginePlanner Init**: Creates planner with LLM model, score tracking (logs/score.json), 4-worker ThreadPoolExecutor for async task generation/summarization/Q&A, loads prompts from `prompts_engine_planner.yaml` and `few_shot_examples.yaml`
6. **Runtime Loop**: `world.run_with_planner(api_info, planner, agents, ticks_per_iteration, include_human)` enters main execution

## Runtime Architecture (Tick/Iteration Loop)

**State Machine (matrx/grid_world.py:run_with_planner)**

The system operates as a state machine with five phases per iteration:

1. **NEEDS_PLANNING**: Capture world state via `__get_complete_state()`, serialize to dict via `process_map_to_dict()`, send to `planner.set_world_state()` (TOON-compressed), submit `planner.submit_generate_tasks(agents)` → returns Future immediately (non-blocking)
2. **PLANNING_IN_PROGRESS**: Poll `planning_future.done()` each tick. When ready, extract task dict, distribute via `agent.set_current_task(task)`. Inject manual plans via `agent.set_manual_task_decomposition()` if configured. Wire mid-iteration re-task callbacks so agents can request new tasks when they finish early. Send human's suggested task as MATRX message.
3. **EXECUTING**: Run ticks via `__step()` until `ticks_in_iteration >= ticks_per_iteration`. Every tick calls `planner.process_agent_questions()` to handle agent Q&A asynchronously.
4. **NEEDS_SUMMARIZATION**: Update task results, submit `planner.submit_summarize(iteration_data, world_state)` → returns Future immediately
5. **SUMMARIZING**: Poll `summary_future.done()` each tick. When ready, call `planner.decide_next_step()` to check termination (block_hit_rate >= 1.0 or max_iterations reached). If continuing, increment iteration and return to NEEDS_PLANNING.

**Key Invariant**: Simulation never blocks on LLM calls. All LLM operations return Futures immediately and complete in background threads.

## Data Flow: World State → EnginePlanner → Agent Tasks

1. **State Capture**: GridWorld calls `__get_complete_state()` → returns full dictionary of all objects/agents with properties
2. **Serialization**: `process_map_to_dict(state)` transforms raw state into structured dict with keys: `victims`, `obstacles`, `doors`, `team_positions`. TOON-compressed via `to_toon()` for token efficiency (35-50% reduction on task dicts, 10-30% on observations vs JSON).
3. **Planner Context**: EnginePlanner receives serialized world via `set_world_state(map)`, uses it to answer agent questions and generate task assignments
4. **Task Generation**: LLM receives world state + previous iteration summary, returns JSON: `{"tasks": {"rescuebot0": "search area 2 for victims", ...}, "human_task": "..."}` (prompts: engine/prompts_engine_planner.yaml, few-shot: few_shot_examples.yaml)
5. **Task Distribution**: GridWorld extracts `task_assignments.get('tasks', {})` and calls `agent.set_current_task(task)` for each agent. Optional: `agent_plans` dict injected via `set_manual_task_decomposition()` for manual plan override.
6. **Agent Execution**: Agents receive tasks, decompose into subtask plans (DAG or simple list via Planning module), execute actions each tick via `decide_on_actions()`

## Async LLM Infrastructure

**Unified Path (agents1/async_model_prompting.py)**

All LLM calls (EnginePlanner + SearchRescueAgent + memory modules) route through `litellm.completion()` via two APIs:

- **Async API** (`submit_llm_call()`, `get_llm_result()`): For agents — submit call, return Future immediately, poll each tick for result. Never blocks.
- **Sync API** (`call_llm_sync()`): For EnginePlanner / memory — blocks caller but runs on planner's ThreadPoolExecutor thread (not the MATRX tick thread).

**Per-Agent Ollama Routing**: Each agent gets dedicated Ollama instance. Agent N → port 11434+N. Routing via `api_base` parameter (e.g. `http://localhost:11435`).

**Retry Logic**: 5 attempts with exponential backoff (1s base wait). Wraps `litellm.completion()` with `_retry_with_backoff()` decorator.

**Token Compression (engine/toon_utils.py)**: TOON (Token-Oriented Object Notation) encoder for all world state / task / observation serialization. Format uses indented `key: value` pairs and array notation `arr[N]: v1,v2,v3`. Self-contained implementation (no external deps).

## Prefetching Strategy (Non-Blocking LLM Calls)

**Mechanism (engine/engine_planner.py):**

- EnginePlanner maintains `_prefetch_future` storing the next iteration's task generation call
- At end of iteration N, `submit_summarize()` launches two parallel futures: summarization for iteration N AND task generation for iteration N+1
- When iteration N+1 starts, `submit_generate_tasks()` checks if `_prefetch_future` exists and is `.done()` → if yes, wraps cached result in resolved Future and returns immediately; if no, returns the Future (already running in background)
- Result: First iteration blocks on LLM, but every subsequent iteration receives tasks instantly (prefetch completed during prior iteration's ticks)

**Callback Pattern**: `submit_summarize()` attaches `_on_summary_done(fut)` callback to update `_last_summary`, which feeds context into next iteration's task generation.

## Agent ↔ Planner Communication (PlannerChannel)

**Bidirectional Q&A Flow:**

1. Agent calls `planner_channel.ask_planner(question)` → posts PlannerMessage to queue
2. Every tick, `planner.process_agent_questions()` drains queue, submits LLM answers via ThreadPoolExecutor, stores (question, Future) pairs
3. When Future completes, planner posts PlannerResponse to channel
4. Agent polls `planner_channel.get_response(msg_id)` each tick until response available

**Non-Blocking**: Q&A never blocks tick loop. Agents continue acting while waiting for planner response.

## Configuration & Termination

**Manual Plans Override**: Set `manual_plans_file="manual_plans.yaml"` in main.py to bypass LLM task generation. EnginePlanner reads YAML file with per-iteration task/plan mappings for each agent.

**Termination**: EnginePlanner reads logs/score.json every iteration. Simulation stops if `block_hit_rate >= 1.0` (all 8 victims rescued) OR `iteration >= max_iterations` (default 50). Score tracking happens in WorldBuilder's CollectionGoal class, which updates score.json when victims enter drop zone.

**Scoring**: Critical victims = 6 points, mild = 3 points, healthy = 0 points. Only injured victims count toward goal.

## Performance Optimizations

- **Prefetching**: Next iteration's tasks generated in background during current iteration (eliminates planning latency after iteration 0)
- **Thread pooling**: Sized to `num_agents*3` so each agent gets ~3 concurrent LLM slots (reasoning + Q&A + prefetch)
- **TOON compression**: 30-60% token reduction shrinks prompt size and response time
- **Non-blocking tick loop**: All LLM calls return Futures immediately; MATRX tick rate unaffected by LLM latency
