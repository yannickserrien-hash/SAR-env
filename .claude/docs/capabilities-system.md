# Agent Capability System

## Overview

The capability system differentiates agents along four dimensions: vision, strength, medical, and speed. Capabilities are enforced by the environment through action validation and duration penalties. Agents can operate in "informed" mode (know their limits upfront) or "discovery" mode (learn through failure feedback).

## Capability Dimensions

**Vision**: Controls perception range (1/2/3 blocks Chebyshev distance).
- Mapped to MATRX `SenseCapability` in `WorldBuilder.add_agents`
- Affects `CollectableBlock` and `ObstacleObject` sense ranges
- Tracked by `AreaExplorationTracker` for coverage updates

**Strength**: Controls obstacle removal ability (low/medium/high).
- Low: Can only remove trees solo
- Medium: Trees and small stones solo; big rocks require `RemoveObjectTogether`
- High: All obstacles solo
- Enforced in `ArtificialBrain.decide_on_action` and `RemoveObjectTogether.is_possible`

**Medical**: Controls victim carrying ability (low/medium/high).
- Low: All victims require `CarryObjectTogether`
- Medium: Mildly injured solo; critically injured require cooperation
- High: All victims solo
- Enforced in `CarryObject.is_possible`

**Speed**: Controls movement delays (slow/normal/fast).
- Slow: 3 extra ticks per move action
- Normal/Fast: No delay
- Enforced in `ArtificialBrain.decide_on_action` by setting `action_duration`

## Presets

Four presets defined in `agents1/capabilities.py`:
- **Scout**: High vision, low strength/medical, fast (explorer role)
- **Medic**: Low vision, medium strength, high medical, slow (carry critical victims)
- **Heavy Lifter**: Low vision, high strength, low medical, normal (remove rocks)
- **Generalist**: Medium vision/strength, low medical, normal (balanced)

Presets resolve via `resolve_capabilities()` which validates values against `CAPABILITIES_MAP`.

## Knowledge Modes

**Informed** (`capability_knowledge='informed'`):
- `get_capability_prompt()` generates natural-language capability description
- Injected into agent's system prompt in `SearchRescueAgent._build_system_prompt`
- `filter_tools_for_capabilities()` removes tools agent can never use
- Agent plans around known limits

**Discovery** (`capability_knowledge='discovery'`):
- No prompt injection or tool filtering
- Agents learn capabilities through action failure feedback
- Failures stored in `BaseMemory` under `'action_failure'` key
- Encourages trial-and-error exploration

## Enforcement Points

**Environment Layer** (`actions1/CustomActions.py`):
- `CarryObject.is_possible()`: Checks medical capability for critical victims
- `RemoveObjectTogether.is_possible()`: Enforces partner adjacency for rocks/stones (strength validation happens pre-action)
- `_get_agent_capabilities()`: Retrieves capability dict from agent properties

**Agent Layer** (`brains1/ArtificialBrain.py`):
- `decide_on_action()`: Pre-validates solo `RemoveObject` based on strength
- Blocks low-strength agents from removing stones/rocks solo
- Blocks medium-strength agents from removing big rocks solo
- Adds movement delay (`action_duration=3`) for slow agents
- Writes failure feedback to agent memory when blocked

**LLM Layer** (`agents1/llm_agent_base.py`):
- Stores capability dict in `_capabilities` attribute
- Passes capabilities to `ActionValidator` for pre-execution checks
- `SearchRescueAgent`: Filters tool schemas by capabilities (informed mode only)

## Configuration Flow

1. **main.py**: Specifies `agent_presets` list and `capability_knowledge` mode
2. **WorldBuilder.add_agents**: Calls `resolve_capabilities()` per agent, creates `SenseCapability` with vision range, stores capability dict in agent properties
3. **SearchRescueAgent.__init__**: Receives capabilities, passes to `LLMAgentBase`, calls `filter_tools_for_capabilities()` if informed
4. **LLMAgentBase.__init__**: Stores capabilities, passes to `ActionValidator` and `Perception` module

## Code Example

```python
# From main.py configuration
agent_presets = ['scout', 'medic', 'heavy_lifter']
capability_knowledge = 'informed'

# Resolves to capability dict (from capabilities.py)
caps = {
    'vision': 'high',    # Scout preset
    'strength': 'low',
    'medical': 'low',
    'speed': 'fast'
}

# Enforcement at decision point (ArtificialBrain.decide_on_action)
if act == 'RemoveObject' and strength == 'low' and 'rock' in obj_id:
    return 'Idle', {'duration_in_ticks': 1}  # Block action

# Enforcement at environment (CarryObject.is_possible)
if 'critical' in object_id and medical != 'high':
    return GrabObjectResult("You may not have the required ability...", False)
```

## Key Files

- `agents1/capabilities.py`: Presets, resolver, prompt generator, tool filter, game rules
- `actions1/CustomActions.py`: `CarryObject.is_possible`, `RemoveObjectTogether.is_possible`, `_get_agent_capabilities`
- `brains1/ArtificialBrain.py`: Pre-decision validation, strength/speed enforcement, failure feedback
- `worlds1/WorldBuilder.py`: `add_agents()` resolves presets, creates `SenseCapability`, stores in agent properties
- `agents1/llm_agent_base.py`: Stores capabilities, passes to validator and perception
- `agents1/agent_sar.py`: Tool filtering (informed mode), vision-based area tracking
