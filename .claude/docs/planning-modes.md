# Communication & Planning Modes

## Overview

The system supports two planning modes (simple and DAG) and provides bidirectional agent-planner communication via `PlannerChannel`. Planning can be driven by LLM or overridden with manual YAML plans.

## Planning Modes

### Simple Mode (Flat Task List)

In `simple` mode, the agent maintains a flat list of subtasks (`task_decomposition`). Tasks are consumed from the end of the list using `get_tasks_for_reasoning(task_num)`, which returns the last N entries for LLM context.

### DAG Mode (Conditional Graph)

In `dag` mode, tasks are structured as a directed acyclic graph (`TaskGraph`) with conditional branching support. Task descriptions are parsed for conditionals using regex patterns:

- Inline: `"Check for X. If present, Y"` (detected by `_COND_INLINE`)
- Sub-bullet: `"Check for X\n  - If present, Y"` (detected by `_COND_SUBBULLET`)

Conditional nodes store both a `description` and a `condition_action`. The LLM decides whether to execute the conditional action or skip it (by choosing `Idle`). The graph always advances linearly—branching is implicit in the LLM's action choice, not graph structure.

**Key methods:**
- `TaskGraph.from_task_list()`: Parses flat task list into linked nodes
- `get_current_task()`: Returns the active node (marked `TaskStatus.ACTIVE`)
- `advance(action_name)`: Marks current task complete, removes it, activates next
- `get_tasks_for_prompt()`: Returns current task + up to 2 upcoming tasks for LLM context

**Implementation:** `agents1/modules/planning_module.py`

## Agent ↔ Planner Communication

### PlannerChannel

Thread-safe bidirectional message queue between agents and `EnginePlanner`. Agents push questions; planner drains them, generates LLM answers in background threads, and posts responses to per-agent slots.

**Message types:**
- `PlannerMessage`: Agent question with `msg_id`, `agent_id`, `content`, `tick`, `context`
- `PlannerResponse`: Planner answer with matching `msg_id` and `agent_id`

**Agent API:**
- `submit_question(agent_id, content, tick, context)`: Push question, returns `msg_id`
- `poll_responses(agent_id)`: Drain responses for this agent (clears after read)

**Planner API:**
- `drain_questions()`: Pull all pending questions from inbound queue
- `post_response(response)`: Push answer to agent's response slot

**Async Q&A pipeline:** `EnginePlanner.process_agent_questions()` is called every tick. It drains new questions, submits them to a thread pool (`_executor.submit(_answer_question_sync)`), and harvests completed futures. Answers use the planner's global context (world state, current tasks) via the `answer_question_system` prompt in `prompts_engine_planner.yaml`.

**Implementation:** `engine/planner_channel.py`

## Response Validation

LLM responses are parsed and validated in `engine/llm_utils.py`:

**`parse_json_response(text)`**: Extracts JSON from LLM text using three fallback strategies:
1. Fenced code block (````json ... ````)
2. Strict `json.loads()` (double-quoted)
3. `ast.literal_eval()` (handles single-quoted keys/values)

Returns `None` on failure. Used by `EnginePlanner._generate_tasks_sync()` to validate task assignments. If parsing fails, the planner logs a warning and falls back to default tasks (`explore area 1 and find victims`).

**Target ID sanitization:** `_strip_location_from_id()` removes `@location` suffixes that LLMs sometimes append despite instructions (e.g., `victim_1@[3,5]` → `victim_1`).

## Manual Plans (YAML Override)

`manual_plans.yaml` bypasses LLM for task/plan generation. Two independent sections:

1. **`iterations`**: List of per-agent `{task, plan?}` entries. Each list item = one planning iteration. If plan is omitted, agent's `PlanningModule` still uses LLM for decomposition.

2. **`agent_plans`**: Fallback map of `agent_id → plan text`. Used when `iterations` lacks a plan for an agent, or as standalone override when `iterations` is absent.

**Activation:** Set `manual_plans_file="manual_plans.yaml"` in `main.py`. Set to `None` to revert to full LLM planning.

**Loading:** `EnginePlanner._build_manual_tasks()` reads YAML, increments `_manual_iteration` counter, and returns `{tasks: {agent_id: task}, plans?: {agent_id: plan}}`.

**Implementation:** `engine/engine_planner.py`, `manual_plans.yaml`

## Prompt Configuration

All planner prompts live in `engine/prompts_engine_planner.yaml`:

- `generate_tasks_system/user`: Task assignment generation
- `summarize_system/user`: Iteration summary
- `answer_question_system/user`: Agent Q&A responses

User prompts use Python `str.format()` placeholders for dynamic data (world state, agent list, previous summary). World state is serialized to TOON notation (indented key-value pairs) via `to_toon()`.
