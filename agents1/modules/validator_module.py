"""
Pre-dispatch action validation for LLM-driven rescue agents.

Catches common LLM mistakes (invalid object IDs, out-of-bounds moves,
wrong action for object type, etc.) *before* the action reaches the MATRX
environment, providing faster and clearer feedback to the agent reasoning
loop.

Capability-based checks (medical, strength) are only applied when the
agent is in ``capability_knowledge='informed'`` mode.  In ``'discovery'``
mode, capability enforcement is left to the environment so the agent can
learn through trial and error.

Usage:
    from agents1.modules.validator_module import ActionValidator, ValidationResult

    validator = ActionValidator(capabilities=caps, capability_knowledge='informed')
    result = validator.validate('CarryObject', {'object_id': 'v1'}, world_state, teammates)
    if not result.valid:
        # result.feedback contains an LLM-friendly error message
        ...
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class ValidationResult:
    """Result of an agent-level action validation."""
    valid: bool
    feedback: str = ''


_OK = ValidationResult(valid=True)


class ActionValidator:
    """Pre-dispatch validation for all LLM-chosen actions."""

    _OBJECT_TYPES = frozenset({'victim', 'tree', 'rock', 'stone'})
    _VICTIM_TYPES = frozenset({'victim'})
    _OBSTACLE_TYPES = frozenset({'tree', 'rock', 'stone'})

    def __init__(
        self,
        capabilities: Optional[Dict[str, Any]] = None,
        capability_knowledge: str = 'informed',
        grid_size: Tuple[int, int] = (25, 24),
        valid_areas: Optional[FrozenSet[int]] = None,
    ):
        self._caps = capabilities or {}
        self._informed = capability_knowledge == 'informed'
        self.GRID_WIDTH, self.GRID_HEIGHT = grid_size
        self.VALID_AREAS: FrozenSet[int] = valid_areas or frozenset(range(1, 8))

    # ── Public API ────────────────────────────────────────────────────────

    def validate(
        self,
        action_name: str,
        args: Dict[str, Any],
        world_state: Dict[str, Any],
        teammates: Set[Tuple[str, Tuple[int, int]]],
    ) -> ValidationResult:
        """Validate an action before dispatch.

        Returns ``ValidationResult(valid=True)`` when no issues are found.
        """
        handler = self._DISPATCH.get(action_name)
        if handler is None:
            return _OK
        return handler(self, args, world_state, teammates)

    # ── Movement ──────────────────────────────────────────────────────────

    _MOVE_CHECKS = {
        'MoveNorth': (lambda loc, h, w: loc[1] <= 0, 'north', 'top'),
        'MoveSouth': (lambda loc, h, w: loc[1] >= h - 1, 'south', 'bottom'),
        'MoveEast':  (lambda loc, h, w: loc[0] >= w - 1, 'east', 'right'),
        'MoveWest':  (lambda loc, h, w: loc[0] <= 0, 'west', 'left'),
    }

    def _validate_directional_move(self, args, ws, teammates, *, direction: str):
        check_fn, dir_name, edge_name = self._MOVE_CHECKS[direction]
        loc = self._agent_location(ws)
        if loc and check_fn(loc, self.GRID_HEIGHT, self.GRID_WIDTH):
            return ValidationResult(False,
                f'Cannot move {dir_name} — already at the {edge_name} edge of the grid.')
        return _OK

    def _validate_move_north(self, args, ws, teammates):
        return self._validate_directional_move(args, ws, teammates, direction='MoveNorth')

    def _validate_move_south(self, args, ws, teammates):
        return self._validate_directional_move(args, ws, teammates, direction='MoveSouth')

    def _validate_move_east(self, args, ws, teammates):
        return self._validate_directional_move(args, ws, teammates, direction='MoveEast')

    def _validate_move_west(self, args, ws, teammates):
        return self._validate_directional_move(args, ws, teammates, direction='MoveWest')

    def _validate_move_to(self, args, ws, teammates):
        x = args.get('x')
        y = args.get('y')
        if x is None or y is None:
            return ValidationResult(False,
                'MoveTo requires both x and y coordinates.')
        try:
            x, y = int(x), int(y)
        except (TypeError, ValueError):
            return ValidationResult(False,
                f'MoveTo coordinates must be integers, got x={x}, y={y}.')
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return ValidationResult(False,
                f'MoveTo target ({x}, {y}) is outside the grid '
                f'(valid range: x=0-{self.GRID_WIDTH - 1}, y=0-{self.GRID_HEIGHT - 1}).')
        return _OK

    def _validate_navigate_to_drop_zone(self, args, ws, teammates):
        return _OK

    def _validate_area_action(self, args, ws, teammates):
        area = args.get('area')
        if area is None:
            return ValidationResult(False, 'Area number is required.')
        try:
            area = int(area)
        except (TypeError, ValueError):
            return ValidationResult(False,
                f'Area must be an integer, got {area}.')
        if area not in self.VALID_AREAS:
            return ValidationResult(False,
                f'Area {area} does not exist. Valid areas: {sorted(self.VALID_AREAS)}.')
        return _OK

    # ── Carry ─────────────────────────────────────────────────────────────

    def _validate_carry_object(self, args, ws, teammates):
        # object_id + nearby check (victims only)
        check = self._check_object_nearby(args, ws, self._VICTIM_TYPES)
        if check is not None:
            return check

        # Already carrying?
        carrying = self._get_carrying(ws)
        if carrying:
            return ValidationResult(False,
                f"You are already carrying {carrying[0]}. "
                f"Drop it first before picking up another object.")

        # Capability: medical check (informed only)
        if self._informed and self._caps:
            obj_id = args.get('object_id', '')
            medical = self._caps.get('medical', 'low')
            if 'critical' in obj_id and medical != 'high':
                return ValidationResult(False,
                    "You cannot carry critically injured victims alone — "
                    "your medical skill is too low. "
                    "Use CarryObjectTogether with a partner instead.")

        return _OK

    def _validate_carry_object_together(self, args, ws, teammates):
        # object_id + nearby check (victims only)
        check = self._check_object_nearby(args, ws, self._VICTIM_TYPES)
        if check is not None:
            return check

        # Already carrying?
        carrying = self._get_carrying(ws)
        if carrying:
            return ValidationResult(False,
                f"You are already carrying {carrying[0]}. "
                f"Drop it first before picking up another object.")

        # Teammate adjacent to object?
        obj_id = args.get('object_id', '')
        if not self._is_teammate_adjacent(obj_id, ws, teammates):
            return ValidationResult(False,
                f"CarryObjectTogether failed: no teammate is adjacent to "
                f"victim '{obj_id}'. Ask a teammate for help using "
                f"SendMessage(message_type='ask_help'), then wait for "
                f"them to arrive before retrying.")

        return _OK

    # ── Drop ──────────────────────────────────────────────────────────────

    def _validate_drop(self, args, ws, teammates):
        carrying = self._get_carrying(ws)
        if not carrying:
            return ValidationResult(False,
                'You are not carrying anything. There is nothing to drop.')

        # Capability: critical victim should use DropObjectTogether (informed only)
        if self._informed and self._caps:
            for obj_id in carrying:
                if 'critical' in obj_id:
                    return ValidationResult(False,
                        "You are carrying a critically injured victim in a "
                        "cooperative carry. Use DropObjectTogether instead of Drop.")

        return _OK

    def _validate_drop_together(self, args, ws, teammates):
        carrying = self._get_carrying(ws)
        if not carrying:
            return ValidationResult(False,
                'You are not carrying anything. There is nothing to drop.')
        return _OK

    # ── Remove ────────────────────────────────────────────────────────────

    def _validate_remove_object(self, args, ws, teammates):
        # object_id + nearby check (obstacles only)
        check = self._check_object_nearby(args, ws, self._OBSTACLE_TYPES)
        if check is not None:
            return check

        # Capability: strength check (informed only)
        if self._informed and self._caps:
            obj_id = args.get('object_id', '')
            strength = self._caps.get('strength', 'medium')
            if strength == 'low' and ('stone' in obj_id or 'rock' in obj_id):
                return ValidationResult(False,
                    "Your strength is too low to remove this obstacle solo. "
                    "Ask a teammate to help with RemoveObjectTogether.")
            if strength == 'medium' and 'rock' in obj_id:
                return ValidationResult(False,
                    "Rocks are too heavy for you to remove alone. "
                    "Use RemoveObjectTogether with a partner.")

        return _OK

    def _validate_remove_object_together(self, args, ws, teammates):
        # object_id + nearby check (rock/stone only, not tree)
        check = self._check_object_nearby(
            args, ws, frozenset({'rock', 'stone'}))
        if check is not None:
            return check

        # Check the object is actually a rock or stone (not a tree)
        obj_id = args.get('object_id', '')
        obj = self._find_object(obj_id, ws)
        if obj and obj.get('type') == 'tree':
            return ValidationResult(False,
                f"Trees cannot be removed cooperatively. "
                f"Use RemoveObject to remove '{obj_id}' solo.")

        # Teammate adjacent to object?
        if not self._is_teammate_adjacent(obj_id, ws, teammates):
            return ValidationResult(False,
                f"RemoveObjectTogether failed: no teammate is adjacent to "
                f"obstacle '{obj_id}'. Ask a teammate for help using "
                f"SendMessage(message_type='ask_help'), then wait for "
                f"them to arrive before retrying.")

        return _OK

    # ── Idle ──────────────────────────────────────────────────────────────

    def _validate_idle(self, args, ws, teammates):
        ticks = args.get('duration_in_ticks', 1)
        try:
            ticks = int(ticks)
        except (TypeError, ValueError):
            return ValidationResult(False,
                'Idle duration_in_ticks must be a positive integer.')
        if ticks < 1:
            return ValidationResult(False,
                'Idle duration_in_ticks must be at least 1.')
        return _OK

    # ── SendMessage ───────────────────────────────────────────────────────

    def _validate_send_message(self, args, ws, teammates):
        message = args.get('message', '')
        if not message or not str(message).strip():
            return ValidationResult(False,
                'SendMessage requires a non-empty message.')

        send_to = args.get('send_to', 'all')
        if send_to != 'all':
            teammate_ids = {t[0] for t in teammates}
            if send_to not in teammate_ids:
                return ValidationResult(False,
                    f"Unknown recipient '{send_to}'. "
                    f"Use 'all' for broadcast or one of: "
                    f"{', '.join(sorted(teammate_ids))}.")
        return _OK

    # ── Dispatch table ────────────────────────────────────────────────────

    _DISPATCH = {
        'MoveNorth': _validate_move_north,
        'MoveSouth': _validate_move_south,
        'MoveEast': _validate_move_east,
        'MoveWest': _validate_move_west,
        'MoveTo': _validate_move_to,
        'NavigateToDropZone': _validate_navigate_to_drop_zone,
        'MoveToArea': _validate_area_action,
        'EnterArea': _validate_area_action,
        'CarryObject': _validate_carry_object,
        'CarryObjectTogether': _validate_carry_object_together,
        'Drop': _validate_drop,
        'DropObjectTogether': _validate_drop_together,
        'RemoveObject': _validate_remove_object,
        'RemoveObjectTogether': _validate_remove_object_together,
        'Idle': _validate_idle,
        'SendMessage': _validate_send_message,
    }

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def _agent_location(ws: Dict) -> Optional[Tuple[int, int]]:
        """Extract agent (x, y) from world_state."""
        if not isinstance(ws, dict):
            return None
        loc = ws.get('agent', {}).get('location')
        if loc is None:
            return None
        return (int(loc[0]), int(loc[1]))

    @staticmethod
    def _get_carrying(ws: Dict) -> List[str]:
        """Extract list of carried object IDs from world_state."""
        if not isinstance(ws, dict):
            return []
        return ws.get('agent', {}).get('carrying', [])

    @staticmethod
    def _get_nearby(ws: Dict) -> List[Dict]:
        """Extract all nearby objects (victims + obstacles) from world_state."""
        if not isinstance(ws, dict):
            return []
        return ws.get('victims', []) + ws.get('obstacles', [])

    @staticmethod
    def _find_object(obj_id: str, ws: Dict) -> Optional[Dict]:
        """Find a specific object in current observations by ID."""
        for o in ActionValidator._get_nearby(ws):
            if o.get('id') == obj_id:
                return o
        return None

    def _nearby_summary(self, ws: Dict, type_filter: Optional[set] = None) -> str:
        """Human-readable summary of nearby actionable objects."""
        types = type_filter or self._OBJECT_TYPES
        parts = []
        for o in self._get_nearby(ws):
            if o.get('type') in types:
                desc = f"{o['id']} ({o['type']}"
                if o.get('severity'):
                    desc += f", {o['severity']}"
                desc += f" at {o['location']})"
                parts.append(desc)
        return ', '.join(parts) or 'none'

    def _check_object_nearby(
        self,
        args: Dict,
        ws: Dict,
        allowed_types: set,
    ) -> Optional[ValidationResult]:
        """Validate object_id presence and proximity.

        Returns a failing ``ValidationResult`` if invalid, or ``None`` if
        the object checks pass (caller continues with action-specific checks).
        """
        obj_id = args.get('object_id', '')
        summary = self._nearby_summary(ws, allowed_types)

        if not obj_id:
            return ValidationResult(False,
                f'This action requires an object_id but none was provided. '
                f'Nearby objects: [{summary}].')

        nearby = self._get_nearby(ws)
        nearby_ids = {o['id'] for o in nearby}
        if obj_id not in nearby_ids:
            return ValidationResult(False,
                f"Object '{obj_id}' is not within reach. "
                f'Move closer to the target or choose a different object. '
                f'Nearby objects: [{summary}].')

        # Check type match
        obj = self._find_object(obj_id, ws)
        if obj and obj.get('type') not in allowed_types:
            obj_type = obj.get('type', 'unknown')
            return ValidationResult(False,
                f"Object '{obj_id}' is a {obj_type}, not a valid target for this action. "
                f'Nearby valid objects: [{summary}].')

        return None

    @staticmethod
    def _is_teammate_adjacent(
        obj_id: str,
        ws: Dict,
        teammates: Set[Tuple[str, Tuple[int, int]]],
    ) -> bool:
        """Check if any teammate is within Chebyshev distance 1 of the object."""
        nearby = ActionValidator._get_nearby(ws)
        obj_loc = None
        for o in nearby:
            if o.get('id') == obj_id:
                loc = o.get('location')
                if loc is not None:
                    obj_loc = (int(loc[0]), int(loc[1]))
                break
        if obj_loc is None:
            return False

        agent_loc = ActionValidator._agent_location(ws)
        for tid, t_loc in teammates:
            # Skip self
            if agent_loc and tuple(t_loc) == tuple(agent_loc):
                continue
            dx = abs(int(t_loc[0]) - obj_loc[0])
            dy = abs(int(t_loc[1]) - obj_loc[1])
            if max(dx, dy) <= 1:
                return True
        return False
