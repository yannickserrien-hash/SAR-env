# Helper Utilities & Common Functions

## Overview

The `helpers/` directory provides 40+ utility functions across 8 modules that handle perception serialization, navigation setup, action validation, and inter-agent communication. These utilities abstract common patterns (Chebyshev distance checks, TOON encoding, adjacency validation) and are used throughout the agent reasoning pipeline.

## Core Modules

### Perception (helpers/perception_module.py, perception_helpers.py)

`Perception` class converts filtered MATRX state dicts into LLM-friendly structured data. Main entry point is `percept_state(state, agent_id, teammates)` which returns a dict containing `agent` (location, carrying), `victims`, `obstacles`, `walls`, and `teammates` arrays. This dict is passed to `to_toon()` for token-efficient prompt encoding.

Low-level helpers: `_serialize_agent()` extracts self-position and carrying status, `_serialize_nearby()` categorizes visible objects, `_classify_type()` maps MATRX class inheritance to semantic types (`victim`, `rock`, `stone`, `tree`, `wall`, `agent`). The `update_world_belief()` method maintains a persistent global state of all objects ever observed.

### Navigation (helpers/navigation_helpers.py)

`apply_navigation(action_name, kwargs, navigator, state_tracker, env_info, memory)` is the universal entry point for movement actions. Returns `(action, state_updates)` where action is a MATRX move command and state_updates contains any new `nav_target` to store.

Key behaviors: `MoveTo` sets A* waypoint to coordinates, `NavigateToDropZone` routes to drop zone, `MoveToArea` navigates to area door, `SearchArea` generates serpentine (boustrophedon) waypoint pattern through area cells. All navigation uses `Navigator.add_waypoints()` and `Navigator.get_move_action()` from MATRX framework.

### Validation (helpers/logic_module.py, logic_helpers.py)

`ActionValidator` performs pre-dispatch checks before actions reach MATRX. Initialized with capabilities and grid bounds. Main method: `validate(action_name, args, world_state, teammates)` returns `ValidationResult(valid, feedback)`.

Validates: grid boundaries (MoveNorth at edge), coordinate ranges (MoveTo), object adjacency (CarryObject, RemoveObject), capability constraints (medical for critical victims, strength for rocks), cooperative action requirements (teammate must be adjacent). In `capability_knowledge='discovery'` mode, capability checks are skipped to allow learning through environment feedback.

Supporting functions in `logic_helpers.py`: `_chebyshev_distance(a, b)` computes max(|dx|, |dy|) for adjacency, `_get_adjacent(ws)` returns all objects at distance 1, `_is_teammate_adjacent(obj_id, ws, teammates, partner_id)` checks if partner is positioned for cooperative action, `is_object_adjacent(args, ws, allowed_types)` validates object is reachable and correct type.

### Communication (helpers/communication_helpers.py)

`_extract_message(msg, agent_id)` converts MATRX Message objects to structured dicts. Returns `{'from': agent_id, 'to': recipient, 'message_type': type, 'text': content}` or None if message is from self or invalid. Validates message types against `VALID_MESSAGE_TYPES` (`ask_help`, `help`, `message`).

### TOON Encoding (helpers/toon_utils.py)

`to_toon(data)` serializes Python dicts/lists to Token-Oriented Object Notation format. Achieves 30-60% token reduction vs JSON while maintaining identical information. Used by EnginePlanner and all agents for task assignments, observations, and memory serialization in LLM prompts. Falls back to custom encoder if `toon-format` PyPI library unavailable.

### Object Types (helpers/object_types.py)

Constants: `_OBJECT_TYPES` (all actionable objects), `_VICTIM_TYPES` (victims only), `_OBSTACLE_TYPES` (rocks/stones/trees). Used throughout validation and perception layers to filter object categories.

## Integration with Agent Pipeline

LLMAgentBase uses helpers at every stage: `filter_observations()` calls `_chebyshev_distance()` to restrict perception range, `Perception.percept_state()` serializes filtered state, `to_toon()` encodes for prompt, `ActionValidator.validate()` pre-checks LLM action choice, `apply_navigation()` sets up A* waypoints for movement. EnginePlanner uses `to_toon()` for task assignment serialization.

## Key Patterns

**Chebyshev Distance**: Used universally for adjacency (distance 1 = reachable). Formula: `max(abs(x1-x2), abs(y1-y2))`. Efficient for grid-world 8-directional movement.

**Adjacency Checking**: `_get_adjacent()` filters all objects at distance 1, `is_object_adjacent()` validates object is nearby and correct type, `_is_teammate_adjacent()` checks partner positioning for cooperative actions.

**Serialization Flow**: Raw MATRX state -> `filter_observations()` (range limit) -> `percept_state()` (structure) -> `to_toon()` (encode) -> LLM prompt.
