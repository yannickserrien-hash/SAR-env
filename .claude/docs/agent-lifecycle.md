# Agent & Brain Lifecycle

## Overview

SAR-env uses a three-tier agent class hierarchy (`ArtificialBrain` → `LLMAgentBase` → `SearchRescueAgent`) where each layer has distinct lifecycle responsibilities. MATRX manages instantiation and tick execution through standardized hooks, while the framework provides separate extension points for infrastructure, perception, and reasoning.

## Class Hierarchy

**`ArtificialBrain`** (`brains1/ArtificialBrain.py`) - Base MATRX brain wrapper. Enforces speed delays and strength-based RemoveObject restrictions. All custom agents inherit from this via `ArtificialAgentBrain` (MATRX base class).

**`LLMAgentBase`** (`agents1/llm_agent_base.py`) - Infrastructure framework for LLM agents. Provides navigation (A* via Navigator), carry autopilot (auto-navigate to drop zone after cooperative carry), action validation (ActionValidator), SharedMemory integration, and async LLM polling. Subclasses implement `decide_on_actions(state)` and optionally `filter_observations(state)`.

**`SearchRescueAgent`** (`agents1/agent_sar.py`) - MARBLE-powered rescue agent. Implements multi-stage cognitive pipeline: CRITIC → PLANNING → REASONING → EXECUTE. Each stage submits async LLM call, returns Idle, polls next tick.

**`HumanBrain`** (`brains1/HumanBrain.py`) - Keyboard-controlled agent. Translates key presses to MATRX actions via `key_action_map`. Shares lifecycle hooks with AI agents but uses `decide_on_action(state, user_input)` instead of `decide_on_actions(state)`.

## Lifecycle Hooks (MATRX → Agent)

### 1. Instantiation (`main.py` → `WorldBuilder.add_agents`)

- Agent brain classes (not instances) passed to `builder.add_agent(location, BrainClass, ...)`
- MATRX calls `_factory_initialise(agent_name, agent_id, action_set, sense_capability, agent_properties, rnd_seed, callback_is_action_possible)`
- Sets `agent_id`, `agent_name`, `action_set`, `sense_capability`, `agent_properties`, `rnd_gen`, `keys_of_agent_writable_props`
- Never override `_factory_initialise` — it's reserved for MATRX

### 2. Pre-simulation setup (`initialize()`)

Called once before world starts. Use for:
- State tracker initialization (`LLMAgentBase`: creates `StateTracker` and `Navigator`)
- Memory allocation (`LLMAgentBase`: instantiates `BaseMemory`, initializes `CommunicationModule`)
- Resetting state between simulation runs (same brain instance may run multiple worlds)

**Example** (`LLMAgentBase.initialize`):
```python
def initialize(self) -> None:
    self._state_tracker = StateTracker(agent_id=self.agent_id)
    self._navigator = Navigator(self.agent_id, self.action_set, Navigator.A_STAR_ALGORITHM)
    self.init_global_state()
    self.communication = CommunicationModule(...)
```

### 3. Per-tick perception (`filter_observations(state)`)

Filters raw MATRX state to agent's perceived view. Called before `decide_on_action`.

**Default** (`LLMAgentBase`): 1-block Chebyshev radius + doors + self + teammates. Saves unfiltered state to `state_for_navigation` for A*.

**Capabilities-aware**: Vision range determined by `capabilities['vision']` (1-3 blocks) via `SenseCapability` set in WorldBuilder.

Override for custom sensing logic (e.g., ray casting, fog of war).

### 4. Per-tick decision (`decide_on_action(state)` / `decide_on_actions(state)`)

**MATRX standard**: `decide_on_action(state) → (action_name: str, action_kwargs: dict)`

**SAR-env pattern**: Subclasses implement `decide_on_actions(state)` (note plural). `ArtificialBrain.decide_on_action` wraps it to enforce action durations and capability restrictions.

**Example flow** (`SearchRescueAgent`):
1. `update_knowledge(state)`: Update state tracker, WORLD_STATE, area tracker
2. `_run_infra(state)`: Handle carry autopilot, navigation (returns action if active, else None)
3. Poll `_pending_future` for LLM result (non-blocking)
4. `_advance_pipeline()`: Submit next pipeline stage (CRITIC/PLANNING/REASONING/EXECUTE)
5. Return `Idle` if waiting for LLM, else return action tuple

### 5. Action result feedback (`_set_action_result(action_result)`)

MATRX calls after action execution. Sets `previous_action_result` (ActionResult object: `.succeeded`, `.action_name`, `.reason`).

**Used by**: `LLMAgentBase._check_carry_success()` to detect successful `CarryObjectTogether` and enter autopilot mode.

## Memory Initialization

**Per-agent memory** (`BaseMemory`): Instantiated in `LLMAgentBase.__init__`. Stores action history, critic feedback, planning outputs. Survives across ticks within a simulation, cleared on `initialize()`.

**Shared memory** (`SharedMemory`): Passed to constructor. Thread-safe (uses `threading.Lock`). Stores cross-agent coordination state (carry autopilot, rescued victims, task assignments). Survives entire simulation.

**Access pattern**:
```python
self.memory.update('action', {'action': 'MoveTo', 'result': 'success'})
recent = self.memory.retrieve_all()[-10:]  # Last 10 entries

self.shared_memory.update('rescued_victims', victim_list)
victims = self.shared_memory.retrieve('rescued_victims')
```

## Extending the Agent

**To create a custom brain**:
1. Inherit from `LLMAgentBase` (or `ArtificialBrain` for non-LLM agents)
2. Implement `decide_on_actions(state) → (action_name, action_kwargs)`
3. Optionally override `filter_observations(state)` for custom perception
4. Optionally override `initialize()` for setup (call `super().initialize()` first)
5. Pass class to `builder.add_agent(location, YourBrainClass, ...)`

**Key methods to override**:
- `decide_on_actions(state)`: Core reasoning loop
- `filter_observations(state)`: Perception filtering
- `initialize()`: One-time setup (e.g., load models, init data structures)

**Do not override**: `_factory_initialise`, `_get_action`, `_set_action_result`, `_set_messages` (MATRX internals)

## Capabilities Integration

Agent capabilities (`vision`, `strength`, `medical`, `speed`) flow through lifecycle:

1. **Instantiation**: `agent_properties['capabilities']` set in WorldBuilder
2. **Perception**: `SenseCapability` created with vision range, passed to `_factory_initialise`
3. **Filtering**: `LLMAgentBase.filter_observations` uses Chebyshev distance (base implementation ignores vision, subclasses can enforce it)
4. **Action enforcement**: `ArtificialBrain.decide_on_action` blocks low-strength agents from solo RemoveObject, applies speed delays
5. **Environment validation**: `CustomActions.is_possible()` checks strength/medical requirements before execution

## Human Agent Integration

`HumanBrain` shares lifecycle hooks but uses `decide_on_action(state, user_input)` with key press list. MATRX calls `_get_action(state, agent_properties, agent_id, user_input)` instead of standard `_get_action(state, ...)`.

Key differences:
- No async LLM calls (synchronous key → action mapping)
- `key_action_map` passed via `_factory_initialise` (e.g., `{'ArrowUp': 'MoveNorth'}`)
- Context menus via `create_context_menu_for_self` / `create_context_menu_for_other`

Add human via `builder.add_human_agent(location, HumanBrain, key_action_map=...)`
