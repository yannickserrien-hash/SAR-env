# SAR-env Project Specification

This document is the canonical reference for the target design of the SAR-env system. It captures all requirements, design decisions, and maps features to existing code. Use this as context when implementing changes.

**Approach**: Incremental refactor of the existing MATRX/MARBLE codebase.

---

## 1. Core Architecture Invariant: Fully Async Execution

This is the most critical constraint in the entire system:

- **All LLM calls are non-blocking.** Multiple agents submit LLM calls simultaneously via `ThreadPoolExecutor`.
- **When an LLM response arrives, the action is executed immediately.** No batching. No waiting for other agents.
- **The environment continues evolving while agents are "thinking."** The observation snapshot sent to the LLM may be stale by the time the response returns. This is a design feature, not a bug.
- **Memory summarization, message replies, and reasoning all happen on background threads.** Nothing blocks the MATRX tick loop.

### 1.1 Single LLM Path: MARBLE Executor

**All LLM calls must go through MARBLE's `model_prompting()`** via `agents1/async_model_prompting.py`. This is the only LLM execution path.

- **API**: `submit_llm_call()` (non-blocking, returns `Future`) and `get_llm_result()` (polls without blocking)
- **Thread pool**: `init_marble_pool(num_agents)` — shared `ThreadPoolExecutor` sized to `max(8, num_agents * 3)`
- **Model format**: LiteLLM strings, e.g. `"ollama/qwen3:8b"`, `"ollama/llama3"`
- **Features**: OpenAI-style tool calling, few-shot messages, retry with exponential backoff

### 1.2 LLM Consolidation Refactoring (Required)

The codebase currently has a **second, legacy LLM path** in `engine/llm_utils.py` that calls Ollama directly via HTTP API and SDK. This must be eliminated — all callers migrated to the MARBLE executor.

**Current state (two paths)**:
| Path | File | Used By |
|------|------|---------|
| MARBLE executor (target) | `agents1/async_model_prompting.py` | `SearchRescueAgent` via `LLMAgentBase` |
| Ollama direct (to remove) | `engine/llm_utils.py` | `EnginePlanner`, `ShortTermMemory`, graveyard agents |

**Callers requiring migration**:

| Caller | Current Usage | Migration |
|--------|-------------|-----------|
| `engine/engine_planner.py` (lines 160, 303, 361) | `query_llm()` from `engine/llm_utils.py` | Switch to `submit_llm_call()` / `get_llm_result()`. Already async via its own ThreadPoolExecutor — replace the underlying call. Model string changes from `"qwen3:8b"` to `"ollama/qwen3:8b"`. |
| `memory/short_term_memory.py` (line 93) | `query_llm()` from `engine/llm_utils.py` | Switch to MARBLE `model_prompting()` or use `submit_llm_call()`. |
| `main.py` (line 31) | `init_llm_pool()` from `engine/llm_utils.py` | Switch to `init_marble_pool()` from `agents1/async_model_prompting.py`. |
| `agents1/agents_graveyard/*` | Various `engine/llm_utils.py` imports | Graveyard code — migrate or delete. |

**Utilities to preserve** (relocate to `engine/parsing_utils.py` or similar):
- `parse_json_response()` — 3-stage JSON/dict fallback parser. Used by `EnginePlanner` to parse LLM text responses.
- `load_few_shot()` — Loads few-shot examples from `few_shot_examples.yaml`. Used by `EnginePlanner` prompts.

**After migration**: Delete `engine/llm_utils.py` (except relocated utilities). Remove the separate Ollama thread pool (`_llm_executor`) and client cache (`_client_cache`).

---

## 2. Environment Overview

A search and rescue grid world where multiple LLM-powered agents (with optional human players) collaborate to find and rescue victims. The goal is to rescue as many victims as possible within the tick budget.

- **Grid**: 25x24 MATRX world with rooms, doors, obstacles, victims, and a drop zone at (23, 8)
- **Victims**: Critically injured (6 pts, require cooperative carry) and mildly injured (3 pts, solo carry possible depending on capabilities)
- **Obstacles**: Rocks (cooperative removal), stones (solo removal), trees (robot-only removal)
- **Framework**: MATRX (local copy in `matrx/`), agents use MARBLE/LiteLLM for LLM reasoning

**Existing code**: `worlds1/WorldBuilder.py`, `matrx/grid_world.py`

---

## 3. Agent Capabilities System

Each agent is created with a set of capabilities. **The environment enforces restrictions and adds delays based on capabilities. The agent layer does NOT know about capability enforcement** — the agent experiences restrictions through action feedback.

### 3.1 Capability Dimensions

| Capability | Values | Effect |
|-----------|--------|--------|
| **Vision** | 1 (low), 2 (medium), 3 (high) | Number of blocks around the agent that are visible. Baseline is 1. Maps to MATRX `SenseCapability` ranges. |
| **Strength** | low, medium, high | **Low**: cannot carry victims, can only report them. Cannot remove rocks/stones — only trees. **Medium**: can carry mildly injured alone. Can remove stones. **High**: can carry mildly injured alone, remove stones and rocks solo. |
| **Medical** | low, high | **High**: can carry critically injured victims alone. **Low**: can only carry mildly injured alone; critically injured require cooperative carry (`CarryTogether`). |
| **Speed** | slow, normal, fast | **Slow**: adds delay ticks to move actions. **Normal**: standard 1-tick movement. **Fast**: no delay (potential future: move 2 blocks per tick). |

### 3.2 Capability Knowledge Mode (Configurable)

A runtime parameter controls whether agents know their own capabilities:

- **`capability_knowledge = 'informed'`**: Capabilities are included in the agent's system prompt (e.g., "You have low strength — you cannot carry victims alone, only report them"). Agents can plan around limitations.
- **`capability_knowledge = 'discovery'`**: Agents are NOT told their capabilities. When they attempt an action they can't do, the environment returns a failure with feedback. Agents learn from trial and error.

### 3.3 Agent Presets

| Preset | Vision | Strength | Medical | Speed | Role |
|--------|--------|----------|---------|-------|------|
| **Scout** | 3 (high) | low | low | fast | Explores quickly, finds victims, reports locations. Cannot carry or remove obstacles. |
| **Medic** | 1 (low) | medium | high | slow | Carries critically injured alone. Slower movement. Limited vision. |
| **Heavy Lifter** | 1 (low) | high | low | normal | Removes rocks/stones solo. Carries mildly injured. Needs help for critical victims. |
| **Generalist** | 2 (medium) | medium | low | normal | Balanced. Can carry mild victims, remove stones/trees. Needs help for critical victims and rocks. |

Presets are configured at agent creation time in `main.py` or a config file. Custom capability combinations are also supported.

### 3.4 Existing Code to Extend

- `worlds1/WorldBuilder.py` → `add_agents()`: Pass capability dict per agent
- `matrx/agents/capabilities/capability.py` → `SenseCapability`: Already supports variable vision ranges
- `brains1/ArtificialBrain.py`: Hardcoded `grab_range=1`, action durations — make these capability-dependent
- `actions1/CustomActions.py`: Add capability checks to `is_possible()` methods

---

## 4. Agent Module Architecture (Strategy Pattern)

Each agent has 4 modules. Each module supports multiple interchangeable strategies, selected at agent creation time (strategy design pattern). When creating an agent, the strategy for each module is specified.

### 4.1 Modules

| Module | Responsibility | Current Strategies | Future Strategies |
|--------|---------------|-------------------|-------------------|
| **Perception** | Filter raw state into agent observations, maintain WORLD_STATE_GLOBAL | `StandardPerception` (1-block Chebyshev) | Vision-range-aware perception, priority-based filtering |
| **Planning** | Decompose high-level tasks into sub-tasks | `SimplePlanning` (flat list countdown), `DAGPlanning` (task graph with conditionals) | LLM-based replanning, hierarchical planning |
| **Reasoning** | Build prompts and decide on actions | `ReActReasoning`, `CoTReasoning`, `ReflexionReasoning` | Tool-use reasoning, multi-step reasoning |
| **Memory** | Store and retrieve past observations, actions, feedback | `ShortTermMemory` (capacity-based LLM compression) | `LongTermMemory` (embedding-based retrieval), `BaseMemory` (simple list) |

### 4.2 Strategy Selection at Creation

```python
agent = SearchRescueAgent(
    ...
    perception_strategy='standard',
    planning_strategy='dag',
    reasoning_strategy='react',
    memory_strategy='short_term',
)
```

### 4.3 Existing Code to Refactor

- `agents1/modules/perception_module.py` → Extract into strategy classes
- `agents1/modules/planning_module.py` → Already has simple/dag modes — formalize as strategies
- `agents1/modules/reasoning_module.py` → `ReasoningIO` is the only implementation — add strategy base class
- `memory/short_term_memory.py`, `memory/long_term_memory.py` → Exist but are NOT integrated. Wire them in as selectable strategies.
- `agents1/tool_registry.py` → `REASONING_STRATEGIES` dict already has cot/react/reflexion prompts

---

## 5. Communication System (Hybrid)

Agents communicate via MATRX Messages with LLM-generated content and typed message tags.

### 5.1 Message Types

| Tag | Purpose | Example |
|-----|---------|---------|
| `ask_help` | Request cooperative action from another agent | "I found a critically injured victim at (10,3). Can you come help carry?" |
| `share_info` | Share discovered information | "Area 2 has a critically injured girl at (10,3)" |
| `request_task` | Ask planner/agent for a new task | "My current task is complete. What should I do next?" |
| `task_update` | Report progress on current task | "Cleared obstacle at (9,4). Entering area 2 now." |
| `reply` | Response to a private message | "On my way to (10,3) to help carry." |

### 5.2 Addressing

- **Broadcast** (`send_to: "all"`): Message goes to all agents.
- **Direct/Private** (`send_to: "RescueBot1"`): Message goes to a specific agent.

### 5.3 Message Handling Rules

- **Private message received** → Triggers a priority LLM call to generate a reply. The agent responds before continuing its current task. The reply LLM call is async (non-blocking).
- **Broadcast message received** → Included in the agent's next observation context. The LLM decides whether to prioritize replying or continue with the current task. No forced interruption.

### 5.4 Existing Code

- `agents1/tool_registry.py` → `SendMessage` tool (lines 188-194): Already supports `send_to` targeting
- `agents1/modules/CommunicationModule.py` (in graveyard): Has templates and message log — can be revived and adapted
- `matrx/messages/message.py` → MATRX `Message` class with `to_id` field

---

## 6. Observation Space

Each agent partially observes the environment. Observations are stored internally as **Python dicts**. TOON conversion happens **ONLY at prompt construction time** to minimize token usage. Agent internals never use TOON format.

### 6.1 Observation Components

Every decision cycle, the agent receives:

1. **Own location**: `[x, y]` coordinates
2. **Teammate info**: Names and locations of all team members (always visible regardless of range)
3. **Nearby objects**: Objects within vision range (capability-dependent). Includes type, id, location, severity (for victims)
4. **History**: Past actions (succeeded + failed with reasons), completed sub-tasks. Stored in memory module. Auto-compressed by ShortTermMemory when capacity is reached.
5. **Current sub-task**: The active sub-task from the planning module
6. **Messages**: All received messages (past + new) and messages sent by the agent
7. **Doors/Entrances**: All known area entrances (discovered doors persist in WORLD_STATE_GLOBAL)

### 6.2 WORLD_STATE_GLOBAL

As agents explore, they accumulate a persistent world model:

```python
{
    'victims': [{'id': 'v1', 'severity': 'critical', 'location': [10, 3]}, ...],
    'obstacles': [{'id': 'rock_1', 'type': 'rock', 'location': [3, 4]}, ...],
    'doors': [{'area': 'area_1', 'location': [3, 4]}, ...],
    'teammate_positions': {'RescueBot1': [22, 11], ...}
}
```

- **Private by default**: Each agent has their own WORLD_STATE_GLOBAL. It is NOT shared automatically.
- **Shared on request**: Agents can share information via `share_info` messages when asked by other agents.

### 6.3 Async Staleness

Since LLM calls are non-blocking and the environment evolves during "thinking" time, the observation snapshot sent to the LLM may be outdated by the time the response arrives. The action validators (Section 7.2) catch cases where the world has changed (e.g., target object moved out of range).

### 6.4 Existing Code

- `agents1/modules/perception_module.py` → `percept_state()`, `process_observations()`, `WORLD_STATE_GLOBAL`
- `agents1/llm_agent_base.py` → `filter_observations()` (1-block Chebyshev), `_tick_setup()`
- `agents1/modules/utils_prompting.py` → `to_toon()` for TOON conversion at prompt time
- `engine/toon_utils.py` → Alternative TOON converter used by EnginePlanner

---

## 7. Action Space

Agents perform high-level actions. Each decision loop, the agent decides to either execute an action OR communicate with other agents.

### 7.1 Available Actions

| Action | Args | Solo/Coop | Description |
|--------|------|-----------|-------------|
| **MoveTo** | `x, y` | Solo | A* pathfinding to target coordinate |
| **Carry** | `object_id` | Solo | Pick up a mildly injured victim (capability-dependent). Must be adjacent. |
| **CarryTogether** | `object_id` | Cooperative | Cooperatively carry a critically injured victim. See Section 7.3. |
| **Drop** | — | Solo | Drop carried object at current location |
| **Remove** | `object_id` | Solo | Remove a stone or tree obstacle (capability-dependent). Must be adjacent. |
| **RemoveTogether** | `object_id` | Cooperative | Cooperatively remove a big rock. Both agents adjacent. |
| **SendMessage** | `message, send_to` | Solo | Send a message to a specific agent or broadcast to all |

### 7.2 Environment-Level Action Validators

Validators live in the **environment layer**, NOT the agent layer. They execute before every action:

| Check | Applies To | Failure Feedback |
|-------|-----------|-----------------|
| Valid arguments (correct types, non-empty) | All actions with args | "Action X requires argument Y but it was missing/invalid" |
| Target object in agent's current visible range | Carry, CarryTogether, Remove, RemoveTogether | "Object '{id}' is not within your visible range" |
| Object type matches action | Carry/CarryTogether | "Carry requires a victim, but '{id}' is a {type}" |
| Agent capability sufficient | Carry, Remove | "You lack the strength to carry victims" / "You cannot remove rocks alone" |
| Partner agent in range (1-2 blocks) | CarryTogether, RemoveTogether | "No partner agent within range for cooperative action" |

On validation failure:
1. Action is NOT executed
2. Failure feedback (action name + reason) is added to the agent's **memory**
3. This feedback is included in the next LLM prompt so the agent can self-correct

### 7.3 CarryTogether Mechanics

This is a key design that avoids coordinating two LLMs for movement:

1. **Initiation**: First agent sends an `ask_help` message to the best nearby agent, then dispatches `CarryTogether(object_id)`.
2. **Agreement**: Second agent must independently issue `CarryTogether` on the **same object_id**. Both must agree — the environment does NOT auto-recruit.
3. **Lock**: Once both agents have committed, **both are locked** — neither can perform any other action until the sequence completes.
4. **Atomic sequence**: Pick up victim → auto-navigate to drop zone (23, 8) → auto-drop victim → release both agents.
5. **Result**: Both agents end up at the drop zone location and can now take new actions.

### 7.4 Existing Code

- `agents1/tool_registry.py` → All 14 `@tool` functions with OpenAI-compatible schemas
- `actions1/CustomActions.py` → MATRX action implementations (CarryObject, CarryObjectTogether, RemoveObjectTogether, etc.)
- `agents1/llm_agent_base.py` → `_validate_action()`, `_check_matrx_action()` (partial validators — need to move to environment layer)

---

## 8. Engine Modes

The EnginePlanner orchestrates agent tasks. It supports 4 modes with different visibility and autonomy levels.

| Mode | Engine Visibility | Task Assignment | Use Case |
|------|------------------|----------------|----------|
| **Powerful** | Full world state (all victims, obstacles, agent positions) | LLM generates detailed task assignments per agent | Best task allocation, full information |
| **Partial** | Only what agents have collectively observed (union of WORLD_STATE_GLOBALs) | LLM generates tasks based on discovered information only | More realistic — planner doesn't have god-mode |
| **None** | No engine | Agents self-organize via broadcast messages. No initial task assignment. | Emergent collaboration research |
| **Manual** | N/A | Tasks loaded from a YAML file | Controlled experiments, reproducible baselines |

### 8.1 Mode: None (Self-Organizing)

- No `EnginePlanner` is instantiated
- Each agent starts with no assigned task
- Agents must communicate via broadcast messages to negotiate task allocation
- The LLM reasoning module handles both task selection and coordination

### 8.2 Existing Code

- `engine/engine_planner.py` → Currently only implements Powerful mode (sees full `world_state` via `set_world_state()`)
- `manual_plans.yaml` → Manual mode already partially works via `_build_manual_tasks()`
- `engine/planner_channel.py` → Agent ↔ Planner communication channel

---

## 9. Human Intervention

A human can interact with agents via the web GUI, even when no human player agent is in the simulation.

### 9.1 Features

- **Chat panel** in the Flask visualizer: Type messages that are delivered to agents as MATRX messages
- **Task override**: Directly assign or override an agent's high-level task mid-execution (bypasses EnginePlanner)
- **Agent selection**: Send messages to all agents or to a specific agent
- **Always available**: Chat works even when `include_human = False` (no human player in the grid)

### 9.2 Existing Code to Extend

- `SaR_gui/visualization_server.py` → Add chat API endpoint and UI panel
- `matrx/messages/message.py` → Use existing Message class
- `agents1/llm_agent_base.py` → `set_current_task()` already supports task injection

---

## 10. Logging

Configurable verbosity with 3 levels.

| Level | What's Logged |
|-------|--------------|
| **minimal** | Agent actions (name + args + success/failure), score changes, iteration transitions |
| **standard** | Everything in minimal + inter-agent messages, task assignments/completions, action feedback, memory updates |
| **debug** | Everything in standard + full LLM prompts, full LLM responses, tool call details, raw observation dicts, TOON-converted prompts |

- Log level is set in `main.py` as a configuration parameter
- Logs are written to `logs/` directory
- Console output respects the same level

### 10.1 Existing Code

- `loggers/ActionLogger.py` → Per-tick action/location CSV logging
- `loggers/OutputLogger.py` → Post-run summary generation
- Console `print()` statements scattered throughout (need to replace with proper logging)

---

## 11. Implementation Mapping

| Feature | Existing File | Status |
|---------|--------------|--------|
| Async LLM execution (MARBLE) | `agents1/async_model_prompting.py` | Done — **single LLM path** for all callers |
| LLM consolidation | `engine/llm_utils.py` | **Requires migration** — EnginePlanner, ShortTermMemory still use Ollama direct. Delete after migration. |
| LLM utility relocation | `engine/llm_utils.py` → new location | **Pending** — `parse_json_response()` and `load_few_shot()` need relocation before deletion |
| Agent base infrastructure | `agents1/llm_agent_base.py` | Done — needs refactoring for capabilities |
| SearchRescueAgent | `agents1/search_rescue_agent.py` | Done — needs strategy injection |
| EnginePlanner | `engine/engine_planner.py` | Done for Powerful mode. Partial/None modes missing |
| Manual plans | `manual_plans.yaml`, `engine/engine_planner.py` | Done |
| Perception module | `agents1/modules/perception_module.py` | Done — needs strategy pattern |
| Planning module | `agents1/modules/planning_module.py` | Partial (simple + dag modes) — needs strategy formalization |
| Reasoning module | `agents1/modules/reasoning_module.py` | Partial — single implementation |
| ShortTermMemory | `memory/short_term_memory.py` | Implemented but **NOT integrated** |
| LongTermMemory | `memory/long_term_memory.py` | Implemented but **NOT integrated** |
| SharedMemory | `memory/shared_memory.py` | Done |
| BaseMemory (simple list) | `memory/base_memory.py` | Done (currently used in production) |
| Tool registry | `agents1/tool_registry.py` | Done — needs capability-aware descriptions |
| Custom actions | `actions1/CustomActions.py` | Done — needs capability enforcement, CarryTogether redesign |
| Action validators | `agents1/llm_agent_base.py` | Partial — needs to move to environment layer |
| Communication module | `agents1/modules/CommunicationModule.py` | In graveyard — needs revival and integration |
| WORLD_STATE_GLOBAL | `agents1/modules/perception_module.py` | Done |
| Agent capability system | — | **Missing entirely** |
| Capability presets | — | **Missing entirely** |
| Engine Partial mode | — | **Missing** |
| Engine None mode | — | **Missing** |
| Human chat panel | — | **Missing** |
| Task override UI | — | **Missing** |
| Configurable logging | — | **Missing** (only ActionLogger CSV exists) |
| Per-agent vision range | `worlds1/WorldBuilder.py` | Hardcoded (agent_sense_range=2 for all) — needs to be per-agent |
| Speed delays | — | **Missing** |
| Strength/Medical enforcement | — | **Missing** |
| Private message priority reply | — | **Missing** |
| CarryTogether lock + auto-navigate | `actions1/CustomActions.py` | Partial — current impl has retry loop, not lock + auto-navigate |
| TOON at prompt time only | `agents1/modules/utils_prompting.py` | Done (already converts dicts at prompt construction) |
| Headless mode | `main.py` | Not implemented (visualizer always starts) |
