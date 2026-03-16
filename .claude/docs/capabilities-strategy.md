# Agent Capabilities & Strategy Pattern

## Overview

This document describes the planned capability system and strategy pattern architecture for agents. **Both features are NOT yet implemented** — this is the design roadmap from Section 3-4 of `project-specification.md` mapped to codebase entry points.

## Capability System (Section 3)

Each agent will have a configuration of four capability dimensions that affect their action restrictions and performance. The environment enforces these constraints transparently — agents experience limitations through action feedback, not direct access to their capability values.

### Capability Dimensions

**Vision** (1-3): Number of blocks visible around the agent. Currently hardcoded as `object_sense_range=1` in `worlds1/WorldBuilder.py`. Will map to MATRX's `SenseCapability` ranges (`matrx/agents/capabilities/capability.py`).

**Strength** (low/medium/high): Affects carrying and obstacle removal. Low-strength agents cannot carry victims or remove rocks/stones, only trees. High-strength agents can remove rocks solo. Currently not implemented — all agents have identical capabilities.

**Medical** (low/high): High medical allows solo carry of critically injured victims. Low medical requires cooperative `CarryObjectTogether`. Not yet implemented.

**Speed** (slow/normal/fast): Adds delay ticks to movement actions. Currently hardcoded in `brains1/ArtificialBrain.py` (`action_duration` values for different actions).

### Agent Presets

Four predefined capability combinations serve different roles:

- **Scout**: High vision (3), low strength, low medical, fast speed — finds victims, cannot carry
- **Medic**: Low vision (1), medium strength, high medical, slow speed — carries critical victims alone
- **Heavy Lifter**: Low vision, high strength, low medical, normal speed — removes obstacles solo
- **Generalist**: Medium vision (2), medium strength, low medical, normal speed — balanced capabilities

### Knowledge Mode

Runtime parameter controls whether agents know their capabilities:

- **`capability_knowledge='informed'`**: Capabilities described in system prompt
- **`capability_knowledge='discovery'`**: Agents learn limits through action failure feedback

### Implementation Entry Points

- **Agent creation**: `worlds1/WorldBuilder.py` → `add_agents()` — pass capability dict per agent
- **Vision enforcement**: `matrx/agents/capabilities/capability.py` → `SenseCapability` already supports variable ranges per object type
- **Action restrictions**: `brains1/ArtificialBrain.py` → hardcoded `grab_range=1` and action durations need to become capability-dependent
- **Action validation**: `actions1/CustomActions.py` → add capability checks to `is_possible()` methods (e.g., verify strength before `CarryObject`)
- **Tool descriptions**: `agents1/tool_registry.py` → update tool docstrings to reflect capability requirements when `capability_knowledge='informed'`

## Strategy Pattern (Section 4)

Each agent has four swappable module strategies selected at creation time. Currently partial implementations exist but need formalization.

### Module Strategies

**Perception**: Filters raw state into observations and maintains `WORLD_STATE_GLOBAL`. Current: `StandardPerception` (1-block Chebyshev) in `agents1/modules/perception_module.py`. Future: vision-range-aware, priority-based filtering.

**Planning**: Decomposes high-level tasks into sub-tasks. Current: `SimplePlanning` (flat countdown list) and `DAGPlanning` (conditional task graph) in `agents1/modules/planning_module.py`. The `Planning` class already has `mode='simple'|'dag'` parameter but needs strategy base class.

**Reasoning**: Builds prompts and decides actions. Current: `ReasoningIO` in `agents1/modules/reasoning_module.py`. Future: separate strategies for CoT/ReAct/Reflexion. Currently reasoning strategies exist as prompt variants in `agents1/tool_registry.py` → `REASONING_STRATEGIES` dict but not as classes.

**Memory**: Stores past observations and actions. Current: `BaseMemory` (simple list in `memory/base_memory.py`), `ShortTermMemory` (LLM-based compression in `memory/short_term_memory.py`). `ShortTermMemory` is implemented but **not integrated** — agents use `BaseMemory` in production.

### Strategy Selection at Creation

```python
agent = SearchRescueAgent(
    perception_strategy='standard',
    planning_strategy='dag',
    reasoning_strategy='react',
    memory_strategy='short_term',
)
```

Currently only `planning_strategy` parameter exists (`main.py` and `worlds1/WorldBuilder.py` pass `planning_mode='simple'|'dag'`).

### Refactoring Roadmap

1. Extract perception logic from `agents1/modules/perception_module.py` → create `PerceptionBase` abstract class with `StandardPerception` implementation
2. Formalize planning strategies: rename `Planning` → `PlanningBase`, create `SimplePlanning` and `DAGPlanning` subclasses inheriting from it
3. Create `ReasoningBase` abstract class with three implementations: `CoTReasoning`, `ReActReasoning`, `ReflexionReasoning` (reuse prompts from `REASONING_STRATEGIES`)
4. Create `MemoryBase` interface that both `BaseMemory` and `ShortTermMemory` inherit from, wire memory selection into agent initialization
5. Add strategy injection parameters to `SearchRescueAgent.__init__()` and store strategy instances as attributes

## Key Files

**Capability implementation targets**: `worlds1/WorldBuilder.py` (agent creation), `brains1/ArtificialBrain.py` (action durations), `actions1/CustomActions.py` (validation), `agents1/tool_registry.py` (tool descriptions)

**Strategy pattern current code**: `agents1/modules/perception_module.py`, `agents1/modules/planning_module.py` (has simple/dag modes), `agents1/modules/reasoning_module.py` (only `ReasoningIO`), `memory/base_memory.py`, `memory/short_term_memory.py` (not integrated)

**Configuration**: `main.py` (top-level config), `worlds1/WorldBuilder.py` → `add_agents()` (agent instantiation)
