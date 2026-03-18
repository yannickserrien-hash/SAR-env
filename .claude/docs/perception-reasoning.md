# Perception & Reasoning Modules

## Overview

The perception-reasoning pipeline filters raw game state into LLM-digestible observations, selects reasoning strategies, constructs per-tick prompts, and dispatches tool calls to MATRX actions. The architecture separates state compression (perception), task management (planning), action selection (reasoning), and action dispatch (execution).

## State Compression Pipeline

**Perception Module** (`agents1/modules/perception_module.py`)

The `Perception` class implements two-stage state filtering:

1. **Observable radius filter** (`filter_observations` in `llm_agent_base.py`): Restricts raw MATRX state to 1-block Chebyshev radius around the agent, plus doors and teammates. Saves unfiltered state for A* pathfinding.

2. **TOON format compression** (`percept_state`): Converts filtered state into token-efficient structure with three sections:
   - `agent`: self-location and carrying status
   - `nearby`: visible objects (victims, obstacles, doors, walls) with type classification and severity metadata
   - `teammates`: known teammate positions

The `process_observations` method maintains a **global persistent state** (`WORLD_STATE_GLOBAL`) that accumulates all objects ever seen during the episode, surviving beyond the 1-block perception radius. This global knowledge merges with local observations in the reasoning prompt.

**TOON Encoding** (`agents1/modules/utils_prompting.py`)

Token-Oriented Object Notation achieves 30-60% token reduction vs JSON by using indented key-value pairs and tabular array notation. Example:

```
nearby[3]{id,type,location}:
  victim_mild_11,victim,5,8
  rock03,rock,6,9
  area_2,door,7,3
```

## Reasoning Strategies

**Strategy Selection** (`agents1/tool_registry.py`)

Three reasoning strategies control LLM prompt framing:

- **CoT** (chain-of-thought): "Think step-by-step about your goal and current situation"
- **ReAct**: "Thought: <reason about goal, observations, and constraints>. Then call the best action tool"
- **Reflexion**: "Reflect on what you have done and what failed. If a previous action failed, try a completely different approach"

Selected strategy becomes the system prompt. The agent class (`SearchRescueAgent`) stores the chosen strategy in `_strategy` and passes it to the LLM call.

## Per-Tick Reasoning Loop

**Reasoning Module** (`agents1/modules/reasoning_module.py`)

`ReasoningIO.get_reasoning_prompt` constructs the LLM prompt from:
- `observation`: Current percept + globally known objects from `WORLD_STATE_GLOBAL`
- `task_decomposition`: Current subtask(s) from the planning module
- `feedback`: Action validation failures from previous tick
- `memory`: Last 15 entries from agent's BaseMemory (action history)
- `previous_action`: Optional context (unused in current implementation)

All fields are TOON-encoded for token efficiency.

**Prompt Flow** (`SearchRescueAgent.decide_on_actions`)

Each tick follows this sequence:
1. **Perception**: Update `WORLD_STATE` and `WORLD_STATE_GLOBAL`
2. **Infrastructure checks**: Carry retry, navigation, rendezvous, LLM polling (handled by `LLMAgentBase._run_preamble`)
3. **Reasoning step**: Build prompt, submit async LLM call, return Idle
4. **Next tick**: Poll LLM future, validate result, dispatch action

The `_reasoning_step` flag controls whether to submit a new LLM call or poll an existing one.

## Tool Calling & Validation

**Tool Registry** (`agents1/tool_registry.py`)

Defines 14 LangChain `@tool` functions (MoveNorth, MoveTo, CarryObject, etc.) that return `(action_name, args_dict, metadata)` tuples. The `build_tool_schemas` function converts these to OpenAI-compatible tool schemas for LiteLLM.

**Action Dispatch** (`agents1/modules/execution_module.py`)

The `execute_action` function maps tool names to MATRX action classes and enriches cooperative actions with `partner_name` (hidden from LLM). Example: `CarryObjectTogether` receives the victim ID from the LLM and adds the partner agent ID at runtime.

**Validation Chain** (`LLMAgentBase._handle_llm_result`)

1. **Parse tool call or text fallback**: Extract action name and args from LLM response
2. **Agent-level validation** (`_validate_action`): Check object_id against `WORLD_STATE.nearby`; populate `_action_feedback` on failure
3. **MATRX feasibility check** (`_check_matrx_action`): Ask MATRX if action is possible before dispatch
4. **Task advancement**: Update task graph or decrement task counter
5. **Memory update**: Record action to BaseMemory
6. **Navigation setup**: Convert MoveTo/NavigateToDropZone to A* waypoints

Failed validations return Idle and inject feedback into the next reasoning prompt.

## Planning vs. Execution Separation

**Planning Module** (`agents1/modules/planning_module.py`)

Two planning modes:
- **Simple mode**: Flat list of subtasks; reasoning prompt shows last N entries
- **DAG mode**: `TaskGraph` with conditional branching (e.g., "Check for victim. If present, carry to drop zone"). The graph advances automatically after each action; reasoning prompt shows current task + 2 upcoming for context.

The planner never executes actions—it only provides task context to the reasoning module. Task advancement happens in `LLMAgentBase` after successful action dispatch.

## Feedback Loop

Feedback flows through three channels:

1. **Action feedback** (`_action_feedback`): Validation failures injected into next reasoning prompt, then cleared
2. **Memory** (`BaseMemory`): Action history appended to storage; last 15 entries included in every prompt
3. **Shared memory** (`SharedMemory`): Cross-agent state (carry rendezvous, victim locations) for coordination

The reasoning module includes feedback as a dedicated field in the TOON-encoded prompt, enabling the LLM to adapt strategy based on previous failures.
