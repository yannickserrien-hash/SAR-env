# MATRX Framework Integration

## Overview

This codebase extends the MATRX multi-agent simulation framework for search-and-rescue scenarios. Custom agent brains inherit from MATRX's `AgentBrain` through local bridge classes that add LLM-driven decision-making, cooperative action handling, and enhanced message tracking.

## Class Hierarchy

The integration follows a three-layer inheritance pattern:

1. **MATRX Core**: `matrx/agents/agent_brain.py` defines `AgentBrain` base class
2. **Local Bridge**: `brains1/ArtificialBrain.py` defines `ArtificialAgentBrain` → `ArtificialBrain`
3. **Agent Implementation**: `agents1/llm_agent_base.py` defines `LLMAgentBase(ArtificialBrain, Perception)`

The bridge layer (`ArtificialAgentBrain`) enhances MATRX's message handling by preserving sender information in `_set_messages()`. The `ArtificialBrain` subclass adds domain-specific action durations (e.g., 200 ticks for stone removal, 150 ticks for victim carry).

## MATRX Lifecycle Hooks

MATRX orchestrates agents through private callback methods that should not be overridden. Subclasses implement public hooks:

**Factory Initialization**: `_factory_initialise(agent_name, agent_id, action_set, sense_capability, agent_properties, rnd_seed, callback_is_action_possible)` called by WorldBuilder. Populates `agent_id`, `action_set`, `sense_capability`, and stores a callback reference for feasibility checks.

**World Start**: `initialize()` called once before simulation. `LLMAgentBase` creates `StateTracker` and `Navigator` instances here.

**Tick Cycle**: Each tick, MATRX calls `_get_action(state, agent_properties, agent_id)` which internally:
1. Updates agent state via `state.state_update(state.as_dict())`
2. Calls `filter_observations(state)` for perception
3. Calls `decide_on_action(state)` for action selection
4. Stores action in `previous_action`
5. Returns filtered state, properties, action name, and kwargs

**Action Result**: After action execution, `_set_action_result(action_result)` provides feedback in `previous_action_result`.

## State Management

MATRX provides `State` (in `matrx/agents/agent_utils/state.py`), a dict-like container with query methods and optional temporal memory decay. The brain's `state` property is updated each tick and supports advanced queries:

- `state["object_id"]` — retrieve single object
- `state[{"property": value}]` — filter by property
- `state[{"class_inheritance": ["Door", "Wall"]}]` — multi-value match

`LLMAgentBase.filter_observations()` restricts observations to 1-block Chebyshev radius plus doors and teammates, storing the unfiltered state in `state_for_navigation` for A* pathfinding.

## Navigator Delegation

`LLMAgentBase` delegates pathfinding to MATRX's `Navigator` (in `matrx/agents/agent_utils/navigator.py`). When an LLM returns `MoveTo` or `NavigateToDropZone`, `_apply_navigation()` configures the navigator:

```python
self._navigator.reset_full()
self._navigator.add_waypoints([coords])
self._nav_target = coords
move = self._navigator.get_move_action(self._state_tracker)
```

Subsequent ticks in `_handle_navigation_tick()` call `get_move_action()` until the destination is reached, automatically clearing `_nav_target`.

## Feasibility Checking

The `is_action_possible(action_name, action_kwargs)` method (inherited from `AgentBrain`) queries MATRX's internal simulation state via the `callback_is_action_possible` callback set during factory initialization. It returns `(bool, ActionResult)`.

`LLMAgentBase._check_matrx_action()` wraps this check, skipping it for navigation and messaging actions (defined in `_SKIP_MATRX_CHECK`). If MATRX rejects an action, the agent populates `_action_feedback` and returns `Idle`, injecting the error into the next LLM prompt.

## Message Handling

MATRX provides `Message` objects (in `matrx/messages/message.py`) with `content`, `from_id`, `to_id` fields. Agents call `send_message(msg)` to append to `messages_to_send`. MATRX calls `_get_messages(all_agent_ids)` to retrieve and clear the queue, then calls `_set_messages(messages)` on recipients.

`ArtificialAgentBrain` overrides `_set_messages()` to store both the full `Message` object in `received_messages` and its content in `received_messages_content`, enabling sender identification (the base MATRX implementation only stores content).

## Sense Capability Filtering

The `sense_capability` attribute (set via `_factory_initialise`) is a `SenseCapability` instance defining perception ranges per object type. MATRX uses this to pre-filter the state dictionary before passing it to `_get_action()`.

`LLMAgentBase.filter_observations()` applies additional filtering beyond MATRX's capability-based range limits, implementing domain logic (1-block radius for nearby objects, unlimited range for doors/teammates). This two-stage filtering (MATRX capability → agent-specific logic) ensures perception constraints are enforced before decision-making.

## Key Integration Points

- **Action Execution**: Return `(action_name: str, action_kwargs: dict)` from `decide_on_action()`
- **State Access**: Read filtered `state` parameter, not raw world state
- **Feasibility**: Call `is_action_possible()` before returning actions (wrapped in `_check_matrx_action()`)
- **Navigation**: Delegate to `Navigator` via `StateTracker` for multi-tick movement
- **Messaging**: Use `send_message(Message(...))` and read `received_messages`
- **Lifecycle**: Override `initialize()`, `filter_observations()`, `decide_on_action()` only

The framework enforces the agent-environment boundary through private callbacks, ensuring consistent simulation semantics across all agent implementations.
