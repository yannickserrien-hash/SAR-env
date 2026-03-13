"""
Unified action dispatch: maps a tool/action name + LLM-supplied arguments
to a MATRX-ready (action_class_name, kwargs) pair.

Replaces the near-identical dispatch logic that previously lived in both
SearchRescueAgent._dispatch_tool_call() and ActionMapper._dispatch().

Cooperative actions are enriched with ``partner_name`` — an internal MATRX
parameter that is never exposed to the LLM.  The partner can be a human
agent or another AI agent.

Usage:
    from agents1.action_dispatch import dispatch_action

    action_name, kwargs = dispatch_action('CarryObject', {'object_id': 'v1'}, 'rescuebot0')
"""

import logging
from typing import Dict, Any, Tuple, Optional

from actions1.CustomActions import Idle as _Idle
from actions1.CustomActions import CarryObject as _CarryObject
from actions1.CustomActions import Drop as _Drop
from actions1.CustomActions import CarryObjectTogether as _CarryObjectTogether
from actions1.CustomActions import DropObjectTogether as _DropObjectTogether
from actions1.CustomActions import RemoveObjectTogether as _RemoveObjectTogether
from matrx.actions.object_actions import RemoveObject as _RemoveObject

logger = logging.getLogger('action_dispatch')

# Direction moves require no kwargs.
_MOVE_ACTIONS = frozenset({'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest'})


def execute_action(
    name: str,
    args: Dict[str, Any],
    partner_name: str = '',
    agent_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Map an action *name* + LLM-supplied *args* to a MATRX (action, kwargs) pair.

    Args:
        name:         Action name as returned by the LLM (tool_call or text-JSON).
        args:         Argument dict from the LLM (may be empty).
        partner_name: Identifier of the cooperative partner agent (human or AI) —
                      injected into cooperative actions automatically.
        agent_id:     Optional agent ID for logging context.

    Returns:
        ``(action_class_name, kwargs)`` ready for MATRX ``decide_on_actions``.
    """
    log_prefix = f"[{agent_id}] " if agent_id else ""
    print(f"{log_prefix}Dispatching action '{name}' with args {args} and partner '{partner_name}'")
    
    # ── Movement ──────────────────────────────────────────────────────────
    if name in _MOVE_ACTIONS:
        return name, {}

    if name == 'MoveTo':
        return 'MoveTo', {'x': args.get('x', 0), 'y': args.get('y', 0)}

    if name == 'NavigateToDropZone':
        return 'NavigateToDropZone', {}

    # ── Solo carry / drop ─────────────────────────────────────────────────
    if name == 'CarryObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _CarryObject.__name__, {'object_id': obj_id}
    

    if name == 'Drop':
        return _Drop.__name__, {'partner_name': partner_name}

    # ── Cooperative carry / drop ──────────────────────────────────────────
    if name == 'CarryObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _CarryObjectTogether.__name__, {'object_id': obj_id, 'partner_name': partner_name}

    if name == 'DropObjectTogether':
        return _DropObjectTogether.__name__, {'partner_name': partner_name}

    # ── Remove obstacle (solo) ────────────────────────────────────────────
    if name == 'RemoveObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _RemoveObject.__name__, {'object_id': obj_id, 'remove_range': 1}

    # ── Remove obstacle (cooperative) ─────────────────────────────────────
    if name == 'RemoveObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _RemoveObjectTogether.__name__, {
            'object_id': obj_id,
            'remove_range': 1,
            'partner_name': partner_name,
        }

    # ── Idle ──────────────────────────────────────────────────────────────
    if name == 'Idle':
        ticks = int(args.get('duration_in_ticks', 1))
        return _Idle.__name__, {'duration_in_ticks': ticks}

    logger.warning("%sUnknown action '%s', defaulting to Idle", log_prefix, name)
    return _Idle.__name__, {'duration_in_ticks': 1}
