# SAR-env Project Specification

This document is the canonical reference for the target design of the SAR-env system. It captures all requirements, design decisions, and maps features to existing code. Use this as context when implementing changes.

**Approach**: Incremental refactor of the existing MATRX codebase (MARBLE dependency eliminated).

---

## 1. Core Architecture Invariant: Fully Async Execution

This is the most critical constraint in the entire system:

- **All LLM calls are non-blocking.** Multiple agents submit LLM calls simultaneously via `ThreadPoolExecutor`.
- **When an LLM response arrives, the action is executed immediately.** No batching. No waiting for other agents.
- **The environment continues evolving while agents are "thinking."** The observation snapshot sent to the LLM may be stale by the time the response returns. This is a design feature, not a bug.
- **Memory summarization, message replies, and reasoning all happen on background threads.** Nothing blocks the MATRX tick loop.

### 1.1 Single LLM Path: `agents1/async_model_prompting.py`

**All LLM calls go through `litellm.completion()` directly** via `agents1/async_model_prompting.py`. No external framework dependencies (MARBLE, Ollama SDK, etc.). This is the only LLM execution path.

- **Async API** (for agents): `submit_llm_call()` (non-blocking, returns `Future`) and `get_llm_result()` (polls without blocking)
- **Sync API** (for EnginePlanner, ShortTermMemory): `call_llm_sync()` — builds messages, calls litellm, returns `Optional[str]`
- **Thread pool**: `init_marble_pool(num_agents)` — shared `ThreadPoolExecutor` sized to `max(8, num_agents * 3)`
- **Per-agent routing**: `api_base` parameter routes each agent to its own Ollama instance (port `base + agent_nr`)
- **Model format**: LiteLLM strings, e.g. `"ollama/qwen3:8b"`, `"ollama/llama3"`
- **Retry**: Inlined exponential backoff decorator (5 retries, base 1s) — no MARBLE dependency
- **Features**: OpenAI-style tool calling, few-shot messages

### 1.2 LLM Consolidation Refactoring — COMPLETED

The legacy LLM path (`engine/llm_utils.py`) that called Ollama directly via HTTP API and SDK has been eliminated. All callers now use `agents1/async_model_prompting.py`.

**What was done**:

| Change | Detail |
|--------|--------|
| `engine/llm_utils.py` | **Deleted**. All `query_llm()`, `query_llm_async()`, `query_llm_with_tools()` removed. Ollama SDK client cache and thread pool removed. |
| `engine/parsing_utils.py` | **Created**. `parse_json_response()` and `load_few_shot()` relocated here. |
| `engine/engine_planner.py` | 3 `query_llm()` calls → `call_llm_sync()`. Model auto-normalized to `ollama/` prefix. `_api_url` → `_api_base`. |
| `memory/short_term_memory.py` | 1 `query_llm()` call → `call_llm_sync()`. Model auto-normalized. |
| `main.py` | `init_llm_pool()` → `init_marble_pool()`. |
| `agents1/llm_agent_base.py` | Added `api_base` parameter to `__init__` and `_submit_llm()`. |
| `agents1/search_rescue_agent.py` | Added `api_base` parameter, forwarded to super. |
| `worlds1/WorldBuilder.py` | Each MARBLE agent gets `api_base=f"http://localhost:{ollama_base_port + agent_nr}"`. |
| `agents1/agents_graveyard/*` | All 3 files (RescueAgent, PlanningModule, ReasoningModule) imports updated. |
| `memory/long_term_memory.py` | **Deleted**. Unused legacy file with MARBLE imports (`model_prompting`, `text_embedding`, `BaseMemory`). |
| `memory/__init__.py` | Removed `LongTermMemory` import. |

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
| **Strength** | low, medium, high | **Low**: can only remove trees solo. Cannot remove rocks or stones solo. **Medium**: can remove trees and stones solo. **High**: can remove trees, stones, and rocks solo. |
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

### 3.4 Implementation — COMPLETED

| Feature | Status | Code |
|---------|--------|------|
| Capability presets & resolver | Done | `agents1/capabilities.py` — `CAPABILITY_PRESETS`, `resolve_capabilities()` |
| Per-agent vision range | Done | `worlds1/WorldBuilder.py` — per-agent `SenseCapability` from `caps['vision']` |
| Capability-aware prompts | Done | `agents1/capabilities.py` — `get_capability_prompt()`, `get_game_rules(caps)` |
| Tool filtering by capabilities | Done | `agents1/capabilities.py` — `filter_tools_for_capabilities()` |
| Medical enforcement (CarryObject) | Done | `actions1/CustomActions.py` — `CarryObject.is_possible()` checks medical level |
| Joint actions bypass capabilities | Done | `CarryObjectTogether` and `RemoveObjectTogether` have no capability checks — always succeed if object exists, in range, and partner present |
| Strength enforcement (RemoveObject solo) | Done | `brains1/ArtificialBrain.py` — `decide_on_action()` blocks stones/rocks for low/medium strength |
| Speed delays | Done | `brains1/ArtificialBrain.py` — `decide_on_action()` adds `action_duration=3` for slow agents on move actions |
| Agent layer integration | Done | `agents1/llm_agent_base.py` + `search_rescue_agent.py` — accept capabilities, filter tools, inject prompts |
| Action validators cleaned up | Done | `agents1/llm_agent_base.py` — removed `'injured' not in` hack and `CarryObjectTogether` skip, let `is_possible()` enforce |
| Config in main.py | Done | `main.py` — `agent_presets`, `capability_knowledge` config variables |

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

## 5. Communication System

Agents communicate via MATRX Messages with LLM-generated content and typed message tags. Sending a message counts as the agent's action for that tick (the agent cannot also move or perform another action). The `CommunicationModule` follows the `Perception` module pattern — environment-sided, processing inbound messages into structured data for prompts.

### 5.1 Message Types

| Tag | Purpose | Example |
|-----|---------|---------|
| `ask_help` | Request cooperation, information, or anything that expects a reply | "I found a critically injured victim at (10,3). Can you come help carry?" |
| `help` | Response to an `ask_help` message | "On my way to (10,3) to help carry." |
| `message` | General purpose (sharing info, status updates, anything else) | "Area 2 has a critically injured girl at (10,3)" |

### 5.2 Message Content Format

Messages use MATRX's `Message(content, from_id, to_id)`. The `content` field is a structured dict:

```python
{"message_type": "ask_help", "text": "Need help carrying victim at (5,3)"}
```

MATRX provides `from_id`, `to_id`, and `message_id` natively on the `Message` object. No separate message_id tracking is needed — LLMs cannot reliably reference message IDs. Agents see the 10 most recent messages + an LLM-generated summary of older ones.

### 5.3 Addressing

- **Broadcast** (`send_to: "all"`): Message goes to all agents (`to_id=None` in MATRX).
- **Direct/Private** (`send_to: "RescueBot1"`): Message goes to a specific agent.

### 5.4 Message Handling

Messages are included in the agent's reasoning prompt alongside observations, tasks, feedback, and memory. The LLM decides whether to respond (via `SendMessage` tool) or continue its current task. No forced interruption or priority queue.

**Auto-announce**: When an agent sends a private `help` reply to someone who sent an `ask_help`, a broadcast announcement is automatically sent (e.g., "RescueBot0 is responding to RescueBot1 help request") so other agents don't redundantly help.

**NOT_SURE: Private message priority reply** — The spec previously described private messages triggering a priority LLM call that interrupts the current task. This is NOT implemented. All messages (private and broadcast) are simply included in the next reasoning prompt. The LLM decides naturally. Implementing priority interruption would require a separate LLM call path in `_run_preamble()` that checks for new private messages before the normal reasoning step.

### 5.5 Communication Strategies (Per-Agent, Configurable)

Strategies control which messages appear in the LLM prompt. Configured per-agent in `main.py` via `comm_strategies` list.

| Strategy | Behavior |
|----------|----------|
| `always_respond` (default) | All messages included in prompt. LLM decides what to do. |
| `busy_aware` | When agent is busy (navigating, carrying), only `ask_help` messages are shown. When idle, all messages shown. |

**NOT_SURE: BusyAware "busy" detection** — The `BusyAwareStrategy.filter_for_prompt()` accepts an `agent_busy` parameter, but `SearchRescueAgent` currently always passes `agent_busy=False` (the default). To fully enable this strategy, the agent needs to pass its actual busy state (e.g., `self._pending_carry_kwargs is not None or self._nav_target is not None`). This is a small wiring change but is not yet done.

### 5.6 Async Message History Summarization

When the message inbox exceeds a threshold (default: 10), the `CommunicationModule` submits an async LLM call via `submit_llm_call()` to summarize the older messages. This is non-blocking — the summary `Future` is polled each tick in `process_messages()`. Until the summary completes, only the most recent messages are shown. Once done, the summary text is prepended to the prompt as context.

### 5.7 Implementation — COMPLETED

| Feature | Status | Code |
|---------|--------|------|
| CommunicationModule (environment-sided) | Done | `agents1/modules/communication_module.py` — `CommunicationModule` class, processes inbound MATRX messages |
| Communication strategies | Done | `agents1/modules/communication_module.py` — `AlwaysRespondStrategy`, `BusyAwareStrategy`, strategy base class `CommStrategy` |
| SendMessage tool with message_type | Done | `agents1/tool_registry.py` — `SendMessage(message, send_to, message_type)` with 3 types: ask_help, help, message |
| Message dispatch | Done | `agents1/modules/execution_module.py` — passes `message_type` through |
| Messages in reasoning prompt | Done | `agents1/modules/reasoning_module.py` — `messages` key in `info_dict`, system prompt mentions communication |
| Separate `_apply_communication()` | Done | `agents1/llm_agent_base.py` — handles SendMessage actions, returns Idle (sending = 1 tick), auto-announce logic |
| SendMessage bug fix | Done | `agents1/llm_agent_base.py` — removed SendMessage from `_apply_navigation()` (was passing invalid action name to MATRX) |
| Inbound message processing | Done | `agents1/llm_agent_base.py` — `_tick_setup()` calls `self.comm.process_messages(self.received_messages)` |
| CommunicationModule init | Done | `agents1/llm_agent_base.py` — `initialize()` creates `CommunicationModule` with agent's LLM model/api_base |
| Async message summarization | Done | `agents1/modules/communication_module.py` — `_maybe_summarize()` + `_poll_summary()` via `submit_llm_call()`/`get_llm_result()` |
| Auto-announce commitments | Done | `agents1/llm_agent_base.py` — `_apply_communication()` broadcasts when private `help` reply to `ask_help` |
| Per-agent comm_strategy config | Done | `main.py` → `WorldBuilder.py` → `SearchRescueAgent` → `LLMAgentBase` → `CommunicationModule` |
| Module init fix | Done | `agents1/modules/__init__.py` — fixed dead import, now imports from `communication_module` |
| Old CommunicationModule replaced | Done | `agents1/modules/CommunicationModule.py` deleted, replaced by `communication_module.py` |

### 5.8 NOT_SURE Items (Not Yet Implemented)

| Item | Description | What's Needed |
|------|-------------|---------------|
| Private message priority reply | Spec Section 5.3 originally described private messages triggering an immediate priority LLM call. Currently all messages just appear in the next prompt. | Add a check in `_run_preamble()` for new private messages → submit a separate priority LLM call before normal reasoning. |
| BusyAware busy state wiring | `BusyAwareStrategy` exists but `agent_busy` is always `False`. | Pass actual busy state from `SearchRescueAgent` to `get_messages_for_prompt(agent_busy=...)` based on nav/carry state. |
| Event-driven scheduling (spec 5.4 pseudocode) | The original spec had tick-delayed response scheduling based on both agents' busy states. | Not implemented. Current approach: LLM sees messages and decides. Could add tick-delay buffer in `CommunicationModule` if needed. |
| Multi-turn conversation loop | Spec Section 5.4 `handle_message` described a loop of LLM calls until a non-command response. | Not implemented. Each tick the LLM makes one decision (act or communicate). No multi-turn within a tick. |

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
- **Shared on request**: Agents can share information via `message` messages when asked by other agents.

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
| Async LLM execution (litellm) | `agents1/async_model_prompting.py` | **Done** — single LLM path via `litellm.completion()` directly (no MARBLE imports) |
| LLM consolidation | `engine/llm_utils.py` (deleted) | **Done** — all callers migrated, file deleted |
| LLM utility relocation | `engine/parsing_utils.py` | **Done** — `parse_json_response()` and `load_few_shot()` relocated |
| Per-agent Ollama routing | `agents1/async_model_prompting.py`, `worlds1/WorldBuilder.py` | **Done** — each agent routes to `localhost:{base_port + agent_nr}` via `api_base` |
| Agent base infrastructure | `agents1/llm_agent_base.py` | Done — needs refactoring for capabilities |
| SearchRescueAgent | `agents1/search_rescue_agent.py` | Done — needs strategy injection |
| EnginePlanner | `engine/engine_planner.py` | Done for Powerful mode. Partial/None modes missing |
| Manual plans | `manual_plans.yaml`, `engine/engine_planner.py` | Done |
| Perception module | `agents1/modules/perception_module.py` | Done — needs strategy pattern |
| Planning module | `agents1/modules/planning_module.py` | Partial (simple + dag modes) — needs strategy formalization |
| Reasoning module | `agents1/modules/reasoning_module.py` | Partial — single implementation |
| ShortTermMemory | `memory/short_term_memory.py` | Implemented but **NOT integrated** |
| LongTermMemory | *(deleted)* | **Deleted** — unused, had MARBLE imports. Re-implement when needed without MARBLE deps. |
| SharedMemory | `memory/shared_memory.py` | Done |
| BaseMemory (simple list) | `memory/base_memory.py` | Done (currently used in production) |
| Tool registry | `agents1/tool_registry.py` | **Done** — capability-aware, SendMessage has message_type |
| Custom actions | `actions1/CustomActions.py` | Done — capability enforcement done, CarryTogether redesign pending |
| Action validators | `agents1/llm_agent_base.py` | Partial — needs to move to environment layer |
| Communication module | `agents1/modules/communication_module.py` | **Done** — environment-sided CommunicationModule with strategies, async summarization, auto-announce |
| Communication strategies | `agents1/modules/communication_module.py` | **Done** — AlwaysRespond + BusyAware, per-agent config |
| SendMessage dispatch | `agents1/llm_agent_base.py` | **Done** — `_apply_communication()` handles outbound, returns Idle |
| Messages in reasoning prompt | `agents1/modules/reasoning_module.py` | **Done** — messages key in prompt info_dict |
| WORLD_STATE_GLOBAL | `agents1/modules/perception_module.py` | Done |
| Agent capability system | `agents1/capabilities.py` | **Done** — presets, resolver, prompt generator, tool filter, game rules |
| Capability presets | `agents1/capabilities.py` | **Done** — scout, medic, heavy_lifter, generalist |
| Engine Partial mode | — | **Missing** |
| Engine None mode | — | **Missing** |
| Human chat panel | — | **Missing** |
| Task override UI | — | **Missing** |
| Configurable logging | — | **Missing** (only ActionLogger CSV exists) |
| Per-agent vision range | `worlds1/WorldBuilder.py` | **Done** — per-agent `SenseCapability` from `caps['vision']` |
| Speed delays | `brains1/ArtificialBrain.py` | **Done** — `decide_on_action()` adds `action_duration=3` for slow agents |
| Strength/Medical enforcement | `actions1/CustomActions.py`, `brains1/ArtificialBrain.py` | **Done** — medical in `CarryObject.is_possible()`, strength in `decide_on_action()` |
| Private message priority reply | — | **NOT_SURE** — not implemented. Messages included in prompt, LLM decides naturally. See Section 5.8. |
| CarryTogether lock + auto-navigate | `actions1/CustomActions.py` | Partial — current impl has retry loop, not lock + auto-navigate |
| TOON at prompt time only | `agents1/modules/utils_prompting.py` | Done (already converts dicts at prompt construction) |
| Headless mode | `main.py` | Not implemented (visualizer always starts) |
