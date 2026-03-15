# World Environment & Configuration

## Overview
The SAR simulation uses MATRX WorldBuilder to create a 25x24 grid search-and-rescue environment with 7 active rooms (14 total defined), 8 victims, a single drop zone, and configurable agent teams. Configuration is centralized in `worlds1/WorldBuilder.py` with optional manual task overrides via `manual_plans.yaml`.

## Grid Topology

**World dimensions**: 25 wide × 24 high
**Active rooms**: 7 (areas 1-7)
**Commented-out rooms**: Areas 8-14 (can be enabled by uncommenting)

Each room is 5×4 units with a single door. Active room layout:
- Row 1: area_1 [3,4], area_2 [9,4], area_3 [15,4], area_4 [21,4]
- Row 2: area_5 [3,7], area_6 [9,7], area_7 [15,7]

Doors open by default (`doors_open=True`). Each area has a "doormat" property marking the accessible tile outside the door.

## Drop Zone & Agent Spawns

**Drop zone**: Column x=23, rows y=8-15 (8 cells for 8 victims)
**Agent start positions**: `[(22,11), (21,11), (20,11), (22,10), (21,10)]` — AI agents cycle through these locations
**Human spawn**: `(22,12)` — adjacent to drop zone

## Victim Placement

**Total**: 8 victims placed at spawn (GhostBlock objects in drop zone) + 4 active CollectableBlock victims in world
**Active victims**:
- `(2,2)` — mildly injured boy (area_1)
- `(10,3)` — critically injured girl (area_2)
- `(14,8)` — mildly injured woman (area_7)
- `(8,9)` — critically injured dog (area_6)

Many victim locations are commented out. Victims are `is_movable=True`, `is_collectable=True` objects with image names like `/images/critically injured girl.svg`.

## Sensing & Occlusion

Defined in `worlds1/WorldBuilder.py`:
```python
agent_sense_range = 2    # Detect other agents within 2 tiles
object_sense_range = 1   # Detect victims/blocks within 1 tile
other_sense_range = np.inf  # Infinite range for walls/doors/decorations
fov_occlusion = True     # Enabled but not fully implemented (warning in HumanBrain.py)
```

`SenseCapability` applied to both AI and human agents. Obstacles have `sense_range=1`. Comments say "Do not change these values."

## Agent Scaling (1-5 Agents)

Configured via `num_rescue_agents` in `main.py`. Agents spawn cyclically from `_AGENT_START_POSITIONS` using modulo:
```python
loc = _AGENT_START_POSITIONS[agent_nr % len(_AGENT_START_POSITIONS)]
```

Up to 5 agents supported (defined by position list length). Each uses incremental Ollama ports: `ollama_base_port + agent_nr + 1`.

## Condition Variants

Three human strength conditions affect carrying/removal abilities:
- **normal**: Default; can carry mild/critical victims alone
- **weak**: Cannot carry victims alone (`HumanBrain.py` blocks `CarryObject` if `strength=='weak'`)
- **strong**: Full capabilities for all actions

Condition passed to `HumanBrain` via `strength` parameter. AI agents receive `condition` parameter but use it primarily for logging/context.

## Manual Plan Override

`manual_plans.yaml` provides iteration-by-iteration task/plan overrides:
- **iterations**: List of per-iteration task assignments (maps `rescuebot0`, `rescuebot1`, etc. to task+plan strings)
- **agent_plans**: Fallback plans when iterations don't specify one

Activated by setting `manual_plans_file="manual_plans.yaml"` in `main.py`. Set to `None` for full LLM mode.

## Obstacles & Decorations

**Obstacles** (commented out in current config): Rocks, stones, trees at various locations — `ObstacleObject` class, `is_movable=True`, `is_traversable=False`

**Water tiles** (commented out): Slow down agents when traversed — `EnvObject` with `is_traversable=True`

**Decorative objects**: Helicopter, ambulance, plants, street tiles, roof tiles — all non-interactive except streets (traversable)

## Configuration Entry Point

`create_builder()` in `worlds1/WorldBuilder.py` orchestrates world creation:
1. Sets grid size, tick duration (0.1s), random seed
2. Adds rooms via `add_room()` calls
3. Calls `add_agents()`, `add_victims()`, `add_drop_off_zones()`
4. Adds decorative objects and area signs
5. Returns builder + agent list

Called from `main.py` with condition, agent_type, num_rescue_agents, planning_mode parameters.

## Key Files

- `worlds1/WorldBuilder.py` — World topology, victim/agent placement, sense ranges
- `manual_plans.yaml` — Manual task/plan overrides
- `brains1/HumanBrain.py` — Human control mapping (arrow keys for movement, q/w for carry/drop, d/a/s for team actions)
- `main.py` — Configuration entry point (condition, num_rescue_agents, agent_type)
