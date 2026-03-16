# Planning Modes & Task Decomposition

## Overview

The SAR-env system supports two planning modes that control how agents track and execute multi-step tasks: **simple** (flat countdown) and **dag** (conditional task graphs). Both modes receive high-level task assignments from EnginePlanner and maintain task decompositions locally within each agent.

## Planning Modes

### Simple Mode (Default)

Flat task list with countdown tracking. The agent receives a numbered list of subtasks and executes them sequentially, consuming entries from the end of the list. Task completion is implicit — each action advances the agent's reasoning state, and the prompt includes the last `task_num` entries.

- **Task structure**: Python list of strings (`self.task_decomposition`)
- **Progress tracking**: `get_tasks_for_reasoning(task_num)` returns last N entries
- **No explicit advancement**: Tasks remain in the list; countdown approach

### DAG Mode

Graph-based planning with support for conditional branching. The `TaskGraph` class parses task descriptions for conditional patterns and creates nodes with `is_condition` flags. Nodes are linked sequentially with `next_id` pointers.

**Conditional patterns detected**:
- Inline: `"Check for victim. If present, carry victim"`
- Sub-bullet (multiline): `"Check for victim\n  - If present, carry victim"`

The graph advances explicitly via `advance(action_name)` after each action. Conditional nodes don't branch the graph structure — the LLM implicitly branches by choosing whether to execute the conditional action or Idle. The graph always moves forward.

**Implementation**: `agents1/modules/planning_module.py` (lines 53-184)

## Task Decomposition Pipeline

### 1. EnginePlanner Task Generation

EnginePlanner generates high-level task assignments per agent each iteration using LLM prompts (`engine/prompts_engine_planner.yaml`). Tasks are injected via `agent.set_current_task(task)`, which resets navigation and calls `planner.update_current_task(task)`.

**Manual override**: `manual_plans_file` in `main.py` bypasses LLM task generation. `manual_plans.yaml` maps iterations to `{agent_id: {task, plan}}` entries. The `plan` field (optional) injects a pre-written task decomposition.

### 2. Agent-Level Decomposition

If no manual plan is provided, agents decompose high-level tasks locally (legacy RescueAgent behavior in `agents_graveyard/RescueAgent.py`). Current SearchRescueAgent uses `set_manual_task_decomposition()` to inject plans from manual_plans.yaml.

**Format**: Numbered list with optional sub-bullets:
```
1. Move to area_1 door at [3, 4]
2. Check for victim
   - If present, carry victim
3. Navigate to Drop Zone
```

Lines are parsed in `llm_agent_base.py` (lines 224-239) and passed to `Planning.set_manual_task_decomposition()`, which splits on `\d+\.` and merges sub-bullets.

### 3. Task Graph Construction

In DAG mode, `TaskGraph.from_task_list(tasks)` builds the graph:
- Regex patterns (`_COND_INLINE`, `_COND_SUBBULLET`) detect conditionals
- Creates `TaskNode` objects with `is_condition=True` and `condition_action` field
- Links nodes sequentially via `next_id`
- Activates first node (sets `status = ACTIVE`)

### 4. Task Advancement

After each action execution, `llm_agent_base.py` calls `planner.advance_task(action_name)` (lines 425, 445). In DAG mode this:
- Marks current node `COMPLETED`
- Removes node from graph
- Advances `_head_id` to `next_id`
- Sets next node `ACTIVE`

Simple mode: no-op.

### 5. Prompt Context

`get_tasks_for_reasoning()` provides task context for LLM reasoning prompts:
- **DAG**: Returns current task + up to 2 upcoming tasks (using `get_tasks_for_prompt()`)
- **Simple**: Returns last `task_num` entries from flat list

Conditional tasks include the action via `full_description()`: `"Check for victim. If so, carry victim"`.

## PlannerChannel Q&A

Thread-safe bidirectional communication between agents and EnginePlanner for mid-iteration clarifications (legacy AskPlanner feature, currently unused by SearchRescueAgent).

**Architecture**: `engine/planner_channel.py`
- Agents push `PlannerMessage` into inbound queue via `submit_question()`
- EnginePlanner drains queue each tick via `drain_questions()`
- Answers generated async in ThreadPoolExecutor using LLM with world state context
- Responses posted to per-agent slots via `post_response()`
- Agents poll via `poll_responses()`

**Implementation**: `EnginePlanner.process_agent_questions()` runs every tick from `matrx/grid_world.py` run_with_planner loop. Uses `prompts_engine_planner.yaml` answer_question prompts.

## Configuration

**main.py**: `planning_mode` ('simple' | 'dag'), `manual_plans_file` (path or None)

**Task tracking**: Each Planning instance tracks `current_task` (string), `task_decomposition` (list), `task_graph` (TaskGraph or None)

## Key Files

- `agents1/modules/planning_module.py` - Planning class, TaskGraph, TaskNode, conditional parsing
- `engine/engine_planner.py` - Task generation, manual plan loading, Q&A processing
- `engine/planner_channel.py` - Thread-safe agent-planner communication
- `agents1/llm_agent_base.py` - Task injection, decomposition parsing, advancement hooks
- `manual_plans.yaml` - Manual task/plan override format
