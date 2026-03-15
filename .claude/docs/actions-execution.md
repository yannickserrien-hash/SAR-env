# Custom Actions & Execution

## Overview

This system extends MATRX with cooperative multi-agent actions for search-and-rescue scenarios. Three collaborative actions enable agents (AI or human) to coordinate on tasks requiring multiple participants: carrying victims together, dropping carried objects cooperatively, and jointly removing obstacles.

## Cooperative Actions

### CarryObjectTogether

Enables two agents to jointly carry heavy victims (healthy/mild severity) that cannot be carried solo. Located in `actions1/CustomActions.py`.

**Validation (is_possible)**:
- Finds nearest partner agent via `_find_partner_agent()` (searches all SearchRescueAgent instances by proximity)
- Returns `NOT_IN_RANGE` failure if no partner found or partner too far from target object
- Delegates to standard `_is_possible_grab()` for inventory/movability checks

**Execution (mutate)**:
- Adds victim to agent's inventory temporarily
- Removes victim from grid (not from carrier)
- **Atomic delivery**: Immediately teleports victim to `CARRY_DROP_ZONE` (23, 8) without agent movement
- Duration: `CARRY_TOGETHER_DURATION` ticks (10) simulates "in progress" state
- Partner agent set to `opacity=0` (invisible) during carry to signal cooperative state

### DropObjectTogether

Cooperative drop action that restores partner visibility and completes victim delivery. Located in `actions1/CustomActions.py`.

**Validation**:
- Checks for invisible partner via `_find_invisible_partner()` (searches for `opacity=0` agents)
- Blocks drop of healthy/mild victims unless cooperative carry is active (invisible partner exists)
- Falls back to standard `_possible_drop()` validation

**Execution**:
- Restores partner agent visibility (`opacity=1`)
- Resets agent avatar image to default (`/images/rescue-man-final3.svg`)
- Performs standard drop via `_act_drop()`

### RemoveObjectTogether

Collaborative obstacle removal (stones, rocks, trees) requiring two agents. Located in `actions1/CustomActions.py`.

**Validation**:
- Inherits from standard MATRX `RemoveObject`
- Checks object exists and is in range
- Partner validation happens during execution via `_find_partner_agent()`

**Execution**:
- Finds nearest partner agent
- Removes object from world if partner available
- Returns failure if no partner in proximity

## Action Mapping & Dispatch

### ActionMapper (`agents1/action_mapper.py`)

Parses LLM JSON responses into MATRX action tuples. Handles multiple JSON formats:

```python
{"action": "CarryObjectTogether", "args": {"object_id": "victim_01"}}
```

**Extraction strategy** (in order):
1. Fenced code blocks: `` ```json ... ``` ``
2. First `{...}` span via `json.loads()`
3. Python dict literal via `ast.literal_eval()` (handles single-quoted LLM output)

Falls back to `Idle` action on parse failure.

### Action Dispatch (`agents1/modules/execution_module.py`)

Central dispatcher maps action names to MATRX-ready `(action_class_name, kwargs)` pairs. Key responsibility: **injecting partner_name** parameter.

**Partner injection**: Cooperative actions automatically receive `partner_name` kwarg (never exposed to LLM). Example:

```python
execute_action('CarryObjectTogether', {'object_id': 'v1'}, partner_name='humanagent')
# Returns: ('CarryObjectTogether', {'object_id': 'v1', 'partner_name': 'humanagent'})
```

**Action categories**:
- Movement: `MoveNorth/South/East/West`, `MoveTo`, `NavigateToDropZone`
- Solo object manipulation: `CarryObject`, `Drop`, `RemoveObject`
- Cooperative: `CarryObjectTogether`, `DropObjectTogether`, `RemoveObjectTogether`
- Communication: `SendMessage`
- Idle: `Idle` (default fallback)

Missing `object_id` parameters trigger `Idle` fallback with logged warning.

## State Constraints

**Cooperative carry state tracking**:
- Partner visibility (`opacity=0`) signals active cooperative carry
- Prevents solo drop of healthy/mild victims unless cooperative state active
- `_find_invisible_partner()` searches registered agents for `opacity=0`

**Partner discovery**:
- `_find_partner_agent()` finds nearest agent with `SearchRescueAgent` in class inheritance
- Returns `None` if no valid partners exist
- Distance calculated via MATRX `get_distance()` utility

**Atomic delivery**: `CarryObjectTogether` bypasses standard multi-tick movement by teleporting victim directly to drop zone. This simulates extended carry duration (10 ticks) while maintaining simple state management.

## Key Differences from Standard MATRX

1. **Partner-aware actions**: Custom actions require coordination between agents
2. **Atomic teleportation**: Cooperative carry delivers victims instantly vs. standard grab-move-drop flow
3. **Visibility as state signal**: Uses `opacity=0` to track active cooperative actions
4. **Centralized dispatch**: `execute_action()` enriches actions with internal parameters before MATRX execution
5. **Robust LLM parsing**: Multiple JSON extraction strategies handle inconsistent LLM output formats

## Configuration

- `CARRY_DROP_ZONE = (23, 8)`: Hardcoded destination matching WorldBuilder drop zone
- `CARRY_TOGETHER_DURATION = 10`: Simulated ticks for cooperative carry action
