# Agent Base Infrastructure (LLMAgentBase)

## Overview

LLMAgentBase is the framework foundation for all LLM-driven rescue agents. It handles perception filtering, action validation, navigation, cooperative carry coordination, SharedMemory rendezvous, async LLM polling, and task injection. Subclasses implement only `decide_on_actions()` and optionally `filter_observations()` — everything else is automatic.

## Approach & Patterns

**Multiple inheritance from MATRX + custom modules**: `LLMAgentBase` extends both `ArtificialBrain` (MATRX integration) and `Perception` (state serialization). `ArtificialBrain` wraps MATRX's `AgentBrain`, providing lifecycle hooks (`initialize`, `decide_on_action`) and feasibility checks (`is_action_possible`).

**Async LLM submission + polling**: Agents submit LLM calls via `_submit_llm()`, storing a `Future` in `_pending_future`. Each tick, `_poll_llm_future()` checks if the result is ready. If not, agent returns `Idle`. When ready, `_handle_llm_result()` parses tool calls or text fallback, validates actions, and dispatches them.

**Lifecycle orchestration**: Subclasses call `_tick_setup()` at the top of `decide_on_actions()` to update state tracking and perception. Then `_run_preamble()` runs infrastructure checks in strict order: carry retry → A* navigation → rendezvous → LLM poll. Returns `None` only when the agent should reason. This ensures infrastructure never deadlocks reasoning.

**Carry retry loop**: When `CarryObjectTogether` is issued, `_pending_carry_kwargs` is set and `_reasoning_step` disabled. Each subsequent tick, `_handle_carry_retry()` checks if the victim still exists nearby. If gone, carry is complete. If timeout (100 ticks) expires, agent logs failure and re-enables reasoning. SharedMemory is updated with `carry_rendezvous` so partners can navigate to the waiting agent.

**Action validation before dispatch**: `_validate_action()` checks that object-based actions (`CarryObject`, `RemoveObject`, etc.) target valid, in-range objects from `WORLD_STATE['nearby']`. Failures populate `_action_feedback` (shown to LLM next tick) and return `Idle`. After validation, `_check_matrx_action()` calls MATRX's `is_action_possible()` to verify feasibility (e.g., victim not already carried). Non-victims are rejected preemptively.

**Navigation delegation**: `MoveTo` and `NavigateToDropZone` actions trigger A* pathfinding via MATRX's `Navigator`. The target is stored in `_nav_target`; `_handle_navigation_tick()` steps the navigator each tick until the destination is reached, then re-enables reasoning.

**SharedMemory rendezvous**: When one agent waits for cooperative carry, it writes `carry_rendezvous` with location and victim ID. `_handle_rendezvous()` checks if another agent published a waiting status; if so, the current agent navigates there automatically, bypassing reasoning until arrival.

## Implementation Details

**Core files**:
- `agents1/llm_agent_base.py` — Base class containing all lifecycle logic
- `agents1/search_rescue_agent.py` — Concrete subclass building LLM prompts
- `brains1/ArtificialBrain.py` — MATRX integration layer (abstract base + wrapper)

**Key components**:
- **Perception** (`agents1/modules/perception_module.py`): `percept_state()` converts MATRX state to LLM-friendly dict; `init_global_state()` / `process_observations()` maintain persistent world knowledge across ticks
- **SharedMemory** (`memory/shared_memory.py`): Thread-safe key-value store for cross-agent coordination
- **ActionMapper** (`agents1/action_mapper.py`): Parses plain-text LLM responses as fallback when tool calls fail
- **Navigator** (MATRX built-in): A* pathfinding; agents call `reset_full()` → `add_waypoints()` → `get_move_action()`

**State tracking**:
- `WORLD_STATE` — per-tick filtered observation (nearby objects, agent location, carrying status)
- `WORLD_STATE_GLOBAL` — persistent knowledge of all victims/obstacles/doors seen so far
- `state_for_navigation` — unfiltered full state saved by `filter_observations()` for A* pathfinding

**Task injection**:
`set_current_task(task)` resets navigation, LLM state, and carry coordination. `set_manual_task_decomposition(text)` parses numbered lists into subtasks. The planner (`Planning` module) tracks progress; `planner.advance_task(action)` marks steps complete.

**Filter default**: `filter_observations()` restricts perception to 1-block Chebyshev radius + doors + self + teammates. Subclasses override for custom ranges.

## Configuration

**Constructor params**:
- `llm_model` — LiteLLM model string (e.g., `'ollama/llama3'`)
- `shared_memory` — Optional `SharedMemory` instance for multi-agent coordination
- `planning_mode` — `'simple'` (flat list) or `'dag'` (task graph with conditional branching)
- `include_human` — Whether to perceive human partner

**Constants** (`llm_agent_base.py`):
- `MAX_NR_TOKENS = 3000` — LLM response limit
- `TEMPERATURE = 0.3` — LLM sampling temperature
- `CARRY_WAIT_TIMEOUT_TICKS = 100` — Timeout for cooperative carry

**Action skip lists**:
- `_OBJECT_ACTIONS` — Actions requiring object validation (CarryObject, RemoveObject, etc.)
- `_SKIP_MATRX_CHECK` — Actions bypassing feasibility check (Idle, MoveTo, SendMessage, CarryObjectTogether)

## Dependencies

- **MATRX framework**: `matrx.agents.agent_utils.navigator.Navigator`, `matrx.agents.agent_utils.state.State`, `matrx.agents.agent_brain.AgentBrain`
- **LiteLLM**: Async LLM calls via `agents1/async_model_prompting.py` (`submit_llm_call`, `get_llm_result`)
- **Custom actions**: `actions1/CustomActions.py` (CarryObject, CarryObjectTogether, RemoveObjectTogether, Idle)
- **Modules**: `agents1/modules/execution_module.py` (action dispatch), `agents1/modules/perception_module.py` (state serialization), `agents1/modules/planning_module.py` (task decomposition)
