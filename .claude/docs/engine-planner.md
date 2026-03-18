# EnginePlanner: Multi-Agent Task Coordination

## Overview

EnginePlanner is the central coordinator in this MARBLE-style orchestration system, responsible for generating task assignments via LLM, tracking iteration progress, and making termination decisions. It operates entirely asynchronously — all LLM calls run in background threads so the MATRX simulation never blocks.

## Architecture Pattern

**Async Prefetch Strategy**: The planner uses a ThreadPoolExecutor (4 workers) to dispatch LLM calls to background threads. Task generation for iteration N+1 is prefetched while iteration N executes, making subsequent `submit_generate_tasks()` calls instantaneous after the first iteration.

**Manual Override Mode**: An optional YAML file can replace LLM task generation entirely with manually scripted plans (controlled via `manual_plans_file` parameter).

## Core Responsibilities

### 1. Task Generation

`submit_generate_tasks(agents)` dispatches task generation to a background thread and returns a `Future` immediately. The LLM receives world state, previous iteration summary, and agent list, then generates per-agent task assignments using prompts from `prompts_engine_planner.yaml`.

Task generation supports two modes:
- **LLM mode**: Queries Ollama model (via `llm_utils.query_llm`) with system/user prompts and few-shot examples
- **Manual mode**: Loads per-iteration task/plan mappings from a YAML file

### 2. Iteration Summarization

`submit_summarize(iteration_data, world_state)` runs LLM summarization in a background thread while simultaneously prefetching the next iteration's tasks. The summary is cached in `_last_summary` and passed to subsequent task generation calls to maintain planning continuity.

### 3. Agent Communication

EnginePlanner processes agent questions via a `PlannerChannel` (thread-safe queue). `process_agent_questions()` is called every tick to:
- Drain new questions from the channel
- Submit each to a background LLM thread
- Post completed answers back to the channel

Agents poll their response slots asynchronously without blocking the main loop.

### 4. Termination Logic

`decide_next_step(iteration_data)` implements rule-based termination (no LLM):
- Reads `logs/score.json` for `block_hit_rate` (0.0-1.0)
- Terminates if `block_hit_rate >= 1.0` (all victims rescued)
- Terminates if `iteration >= max_iterations`

Returns `False` to stop simulation, `True` to continue.

## Data Flow

1. **Planning Phase**: `run_with_planner()` calls `submit_generate_tasks(agents)` → returns Future
2. **Execution Phase**: World ticks N times (`ticks_per_iteration`), polling the Future until ready
3. **Summarization Phase**: Calls `submit_summarize()` → starts prefetch for next iteration
4. **Termination Check**: `decide_next_step()` reads score file and decides whether to continue

## Key Files

- `engine/engine_planner.py` - Core coordinator with async task/summary pipelines
- `engine/llm_utils.py` - LLM querying via Ollama, shared thread pool (`init_llm_pool`)
- `engine/iteration_data.py` - IterationData dataclass tracking assignments/results/score
- `engine/prompts_engine_planner.yaml` - LLM prompt templates for tasks, summaries, Q&A
- `engine/planner_channel.py` - Thread-safe agent ↔ planner communication channel
- `matrx/grid_world.py` - `run_with_planner()` integrates planner into tick loop
- `main.py` - Initializes planner with config and calls `run_with_planner()`

## Configuration

Configured in `main.py`:
- `max_iterations`: Maximum planning cycles (default 50)
- `llm_model`: Ollama model for planner LLM calls (e.g., `qwen3:8b`)
- `ticks_per_iteration`: MATRX ticks per iteration (e.g., 1200 = 2 minutes)
- `manual_plans_file`: Optional YAML file to override LLM task generation
- `score_file`: Path to score.json for termination checks

## Dependencies

- **Ollama**: LLM backend via HTTP API (`localhost:11434`)
- **concurrent.futures**: ThreadPoolExecutor for async LLM calls
- **PyYAML**: Prompt template and manual plan loading
