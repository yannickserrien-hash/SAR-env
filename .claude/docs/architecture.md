# System Architecture & Initialization Flow

## Overview

SAR-env is a search-and-rescue simulation combining MATRX's grid-world environment with LLM-powered agents using MARBLE-style planning. The system operates in discrete **iterations** (each 1200 ticks / 2 minutes), with an overarching **EnginePlanner** coordinating multi-agent task assignments via background LLM calls that never block the simulation loop.

## Component Relationships

**Core Components:**
- **main.py**: Entry point, configuration hub, initialization orchestrator
- **WorldBuilder** (worlds1/WorldBuilder.py): MATRX world setup, agent/object placement
- **GridWorld** (matrx/grid_world.py): Tick loop executor, state manager, simulation runtime
- **EnginePlanner** (engine/engine_planner.py): LLM-based task coordinator, termination controller
- **SearchRescueAgent** (agents1/search_rescue_agent.py): Individual agent brain implementing MARBLE patterns

**Supporting Systems:**
- **SharedMemory** (memory/shared_memory.py): Thread-safe cross-agent communication store
- **PlannerChannel** (engine/planner_channel.py): Bidirectional agent ↔ planner messaging
- **LLM Executor** (agents1/async_model_prompting.py): Unified LLM interface, ThreadPoolExecutor pool, dual backend support (ollama_sdk/requests)
- **Parsing Utils** (engine/parsing_utils.py): JSON response parsing, few-shot example loader

## Initialization Sequence

1. **Configuration (main.py)**: Load parameters (agent_type='marble', ticks_per_iteration=1200, num_rescue_agents=2, planner_model='qwen3:8b', planning_mode='dag')
2. **LLM Pool Init**: `init_marble_pool(num_rescue_agents)` creates background thread pool for non-blocking LLM calls via `async_model_prompting.py`
3. **World Creation**: `create_builder()` returns WorldBuilder + agent list; builder configures 25×24 grid with rooms, victims, obstacles, drop zones
4. **World Startup**: `builder.startup()` + `visualization_server.run_matrx_visualizer()` launches MATRX API and custom UI
5. **EnginePlanner Init**: Creates planner with LLM model, score tracking, 4-worker ThreadPoolExecutor for async task generation/summarization
6. **Runtime Loop**: `world.run_with_planner(api_info, planner, agents, ticks_per_iteration, include_human)` enters main execution

## Runtime Architecture (Tick/Iteration Loop)

**State Machine (matrx/grid_world.py:run_with_planner)**

The system operates as a state machine with five phases per iteration:

1. **NEEDS_PLANNING**: Capture world state via `__get_complete_state()`, serialize to dict, submit `planner.submit_generate_tasks(agents)` → returns Future immediately (non-blocking)
2. **PLANNING_IN_PROGRESS**: Poll `planning_future.done()` each tick. When ready, extract task assignments and distribute to agents via `agent.set_current_task(task)`. Store assignments in SharedMemory for cross-agent visibility.
3. **EXECUTING**: Run ticks via `__step()` until `ticks_in_iteration >= ticks_per_iteration`. Every tick calls `planner.process_agent_questions()` to handle agent queries asynchronously.
4. **NEEDS_SUMMARIZATION**: Update task results, submit `planner.submit_summarize(iteration_data, world_state)` → returns Future immediately. **Prefetch launched here**: submit_summarize also triggers next iteration's task generation in background.
5. **SUMMARIZING**: Poll `summary_future.done()` each tick. When ready, call `planner.decide_next_step()` to check termination (victims_rescued == 8 or max_iterations reached). If continuing, increment iteration and return to NEEDS_PLANNING.

**Key Invariant**: Simulation never blocks on LLM calls. All LLM operations (`submit_generate_tasks`, `submit_summarize`, `process_agent_questions`) return Futures immediately and complete in background threads.

**LLM Backend Architecture**: All LLM calls route through `agents1/async_model_prompting.py`, which supports two Ollama backends: `ollama_sdk` (official Python client) and `requests` (direct HTTP). Backend is configurable via `set_backend()` or `LLM_BACKEND` environment variable. EnginePlanner uses synchronous wrapper `call_llm_sync()`, agents use async `submit_llm_call()`.

## Data Flow: World State → EnginePlanner → Agent Tasks

1. **State Capture**: GridWorld calls `__get_complete_state()` → returns full dictionary of all objects/agents with properties
2. **Serialization**: State is structured with keys: `victims`, `obstacles`, `doors`, `team_positions`
3. **Planner Context**: EnginePlanner receives serialized world via `set_world_state(map)`, uses it to answer agent questions and generate task assignments
4. **Task Generation**: LLM receives world state summary + previous iteration summary, returns JSON: `{"tasks": {"rescuebot0": "search area 2 for victims", ...}, "human_task": "..."}` (engine/prompts_engine_planner.yaml defines prompts)
5. **Task Distribution**: GridWorld extracts `task_assignments.get('tasks', {})` and calls `agent.set_current_task(task)` for each agent
6. **Agent Execution**: Agents receive tasks, decompose into subtask plans (DAG or simple list), execute actions each tick via `decide_on_actions()`

## Prefetching Strategy (Non-Blocking LLM Calls)

**Mechanism (engine/engine_planner.py):**

- EnginePlanner maintains `_prefetch_future` storing the next iteration's task generation call
- At end of iteration N, `submit_summarize()` launches two parallel futures: summarization for iteration N AND task generation for iteration N+1
- When iteration N+1 starts, `submit_generate_tasks()` checks if `_prefetch_future` exists and is `.done()` → if yes, returns cached result immediately; if no, waits for it (but it's already running in background)
- Result: First iteration blocks on LLM, but every subsequent iteration receives tasks instantly (prefetch completed during prior iteration's ticks)

**Callback Pattern**: `submit_summarize()` attaches `_on_summary_done(fut)` callback to update `_last_summary`, which feeds context into next iteration's task generation.

## Configuration & Termination

**Configuration**: main.py exposes `manual_plans_file` to override LLM task generation with YAML-defined tasks/plans (see manual_plans.yaml). YAML format includes `iterations` with per-agent `tasks` and optional `plans` (task decompositions).

**Termination**: EnginePlanner reads logs/score.json every iteration. Simulation stops if `block_hit_rate >= 1.0` (all 8 victims rescued) OR `iteration >= max_iterations` (default 50). Score tracking happens in WorldBuilder's CollectionGoal class, which updates score.json when victims enter drop zone (x=23, y=8-15).

**Scoring**: Critical victims = 6 points, mild = 3 points. Goal: maximize score + speed (fewer iterations).
