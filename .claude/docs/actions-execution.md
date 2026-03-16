# Actions & Execution

## Overview

The system implements 14 custom action tools that agents can call via LLM tool calling. Actions are validated through a two-tier pipeline before execution, and cooperative actions (CarryTogether, RemoveTogether) use a retry-loop mechanism to synchronize multiple agents.

## Action Registration & Schema

All 14 actions are defined in `agents1/tool_registry.py` as `@tool`-decorated LangChain functions. Each tool returns a `(action_name, args_dict, metadata)` tuple. The `build_tool_schemas()` function converts these to OpenAI-compatible JSON schemas for LiteLLM.

**Action categories:**
- **Movement**: MoveNorth/South/East/West (single step), MoveTo (A* navigation), NavigateToDropZone
- **Solo carry**: CarryObject, Drop
- **Cooperative carry**: CarryObjectTogether, DropObjectTogether (requires 2 agents adjacent)
- **Solo obstacle**: RemoveObject (small stones, trees)
- **Cooperative obstacle**: RemoveObjectTogether (big grey rocks, requires 2 agents)
- **Utility**: Idle, SendMessage

Tool schemas define strict argument types (e.g., `MoveTo` requires `x: int, y: int`). The `GAME_RULES` constant encodes domain constraints (e.g., "Critically injured victims require CarryObjectTogether").

## Action Dispatch

`agents1/modules/execution_module.py` (`execute_action()`) maps LLM-returned tool names + args to MATRX-ready `(action_class_name, kwargs)` pairs. Key responsibilities:

- **Parameter validation**: Missing `object_id` triggers Idle fallback
- **Partner enrichment**: Cooperative actions inject `partner_name` kwarg (not exposed to LLM) by finding nearest agent
- **Fallback handling**: Unknown actions default to Idle(1)

## Two-Tier Validation Pipeline

Before dispatching to MATRX, `LLMAgentBase` runs two sequential checks in `_handle_llm_result()`:

### 1. World-State Validation (`_validate_action()`)

Validates object-based actions (Carry*, Remove*) against current `WORLD_STATE['nearby']`:

- **Missing object_id**: Returns Idle, sets `_action_feedback` with nearby objects summary
- **Object not in range**: Checks if `object_id` exists in nearby set (1-block Chebyshev radius), fails if absent
- **Feedback loop**: Rejection messages include agent location + nearby actionable objects (type + severity + location) for next LLM prompt

Skips validation for movement and utility actions.

### 2. MATRX Feasibility Check (`_check_matrx_action()`)

Calls `is_action_possible()` on the MATRX action class to verify world-state constraints:

- **Non-victim filter**: Rejects carry actions on non-victim objects (rocks, stones)
- **MATRX rules**: Enforces grab_range, max_objects, inventory checks, mutual exclusivity (can't carry while already carrying)
- **Result feedback**: Populates `_action_feedback` with MATRX error message + nearby objects

Actions like Idle, movement, and message-sending skip this check.

## Cooperative Action Execution

### CarryObjectTogether / RemoveObjectTogether

Defined in `actions1/CustomActions.py`. Both require:

1. **Partner discovery**: `_find_partner_agent()` finds nearest agent in world_state by distance
2. **Dual adjacency**: Both agents must be within `remove_range=1` of target object
3. **is_possible()**: Checks object exists in infinite range (actual range validated in mutate())
4. **mutate()**: Verifies both agents adjacent, executes removal/carry

### Cooperative Carry Retry Loop

Handled by `LLMAgentBase._handle_carry_retry()` (runs every tick):

- **Trigger**: When CarryObjectTogether is requested, `_pending_carry_kwargs` is set
- **Rendezvous**: Publishing agent writes `{agent, victim_id, location, status: 'waiting_for_partner'}` to SharedMemory
- **Partner response**: Other agent reads rendezvous, navigates to location via `_handle_rendezvous()`
- **Retry logic**: Re-submits CarryObjectTogether every tick for up to `CARRY_WAIT_TIMEOUT_TICKS` (default 50)
- **Completion**: Exits when `object_id` disappears from nearby (victim carried/delivered)
- **Timeout**: After 50 ticks, clears rendezvous, logs failure to memory, returns to reasoning

During carry, one agent is set to `opacity=0` (invisible). `DropObjectTogether` uses `_find_invisible_partner()` to restore visibility.

## Object Type Checking

Validation logic in `_validate_action()` and `_check_matrx_action()` filters nearby objects by type:

```python
actionable_types = {'victim', 'rock', 'stone', 'tree'}
```

Victim severity (`'critically_injured'`, `'mildly_injured'`) is extracted from object properties and included in feedback. MATRX actions use `'injured'` substring check to distinguish victims from obstacles.

## Future: Capability-Based Filtering

The project specification mentions capability-based action filtering (not yet implemented). Would restrict tool schemas per agent (e.g., only `rescuebot0` can remove trees). Current system sends all 14 tools to every agent.

## Key Files

- `agents1/tool_registry.py` - Tool definitions, schemas, game rules
- `agents1/modules/execution_module.py` - Action dispatch, partner enrichment
- `agents1/llm_agent_base.py` - Validation pipeline, carry retry loop
- `actions1/CustomActions.py` - MATRX action classes (Carry*, Remove*, Drop*, Idle)
