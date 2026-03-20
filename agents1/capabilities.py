"""
Agent Capabilities System — Presets, resolver, prompt generator, tool filter.

Each agent has 4 capability dimensions:
    - Vision:   1/2/3 blocks visible around agent
    - Strength: low/medium/high — obstacle removal ability
    - Medical:  low/high — victim carrying ability
    - Speed:    slow/normal/fast — movement delay

The environment enforces these via is_possible() checks and action_duration.
"""

from typing import Any, Dict, List, Tuple


# ── Presets ──────────────────────────────────────────────────────────────────

CAPABILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    'scout': {
        'vision': 3,
        'strength': 'low',
        'medical': 'low',
        'speed': 'fast',
    },
    'medic': {
        'vision': 1,
        'strength': 'low',
        'medical': 'high',
        'speed': 'medium',
    },
    'heavy_lifter': {
        'vision': 1,
        'strength': 'high',
        'medical': 'low',
        'speed': 'normal',
    },
    'generalist': {
        'vision': 2,
        'strength': 'medium',
        'medical': 'low',
        'speed': 'normal',
    },
}

DEFAULT_PRESET = 'generalist'

# Extra move ticks for slow agents
SPEED_MOVE_DELAY: Dict[str, int] = {
    'slow': 3,
    'normal': 0,
    'fast': 0,
}

# Valid values per dimension
_VALID = {
    'vision': {1, 2, 3},
    'strength': {'low', 'medium', 'high'},
    'medical': {'low', 'high'},
    'speed': {'slow', 'normal', 'fast'},
}


def resolve_capabilities(preset_or_dict) -> Dict[str, Any]:
    """Resolve a preset name or custom dict into a validated capability dict."""
    if isinstance(preset_or_dict, str):
        if preset_or_dict not in CAPABILITY_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_or_dict}'. "
                f"Available: {list(CAPABILITY_PRESETS.keys())}"
            )
        caps = dict(CAPABILITY_PRESETS[preset_or_dict])
    elif isinstance(preset_or_dict, dict):
        caps = dict(CAPABILITY_PRESETS[DEFAULT_PRESET])
        caps.update(preset_or_dict)
    else:
        caps = dict(CAPABILITY_PRESETS[DEFAULT_PRESET])

    for dim, valid_vals in _VALID.items():
        if caps.get(dim) not in valid_vals:
            raise ValueError(
                f"Invalid capability '{dim}': {caps.get(dim)}. Valid: {valid_vals}"
            )
    return caps


def get_capability_prompt(capabilities: Dict[str, Any]) -> str:
    """Return a human-readable description of agent capabilities for the LLM prompt."""
    lines = ["Your agent capabilities:"]

    # Vision
    v = capabilities.get('vision', 2)
    vis_desc = {1: 'low (1 block)', 2: 'medium (2 blocks)', 3: 'high (3 blocks)'}
    lines.append(f"- Vision: {vis_desc.get(v, str(v))} — you can see objects within {v} block(s).")

    # Strength
    s = capabilities.get('strength', 'medium')
    if s == 'low':
        lines.append(
            "- Strength: low — you can only remove fallen trees solo. "
            "Stones and rocks are too heavy for you alone, but you can always help remove them "
            "cooperatively (RemoveObjectTogether)."
        )
    elif s == 'medium':
        lines.append(
            "- Strength: medium — you can remove trees and small stones solo. "
            "Big rocks require RemoveObjectTogether with a partner."
        )
    else:
        lines.append(
            "- Strength: high — you can remove trees, small stones, and big rocks solo (RemoveObject)."
        )

    # Medical
    m = capabilities.get('medical', 'low')
    if m == 'low':
        lines.append(
            "- Medical: low — you can carry mildly injured victims solo (CarryObject). "
            "Critically injured victims require cooperative carry (CarryObjectTogether) with a partner."
        )
    else:
        lines.append(
            "- Medical: high — you can carry ALL victims solo (CarryObject), "
            "including critically injured ones."
        )

    # Speed
    sp = capabilities.get('speed', 'normal')
    if sp == 'slow':
        lines.append("- Speed: slow — each move costs 3 extra ticks (you move significantly slower than other agents).")
    elif sp == 'fast':
        lines.append("- Speed: fast — you move at full speed with no delays.")
    else:
        lines.append("- Speed: normal — standard movement speed.")

    return '\n'.join(lines)


def filter_tools_for_capabilities(
    tool_schemas: List[Dict],
    tools_by_name: Dict[str, Any],
    capabilities: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Remove tools the agent can NEVER use based on capabilities.

    Returns filtered (tools_by_name, tool_schemas).
    """
    remove_names = set()

    # Joint/cooperative actions (CarryObjectTogether, RemoveObjectTogether) are
    # never filtered — they always succeed regardless of individual capability.

    filtered_tools_by_name = {
        k: v for k, v in tools_by_name.items() if k not in remove_names
    }
    filtered_schemas = [
        s for s in tool_schemas
        if s.get('function', {}).get('name', s.get('name', '')) not in remove_names
    ]

    return filtered_tools_by_name, filtered_schemas


def get_game_rules(capabilities: Dict[str, Any] = None, drop_zone=(23, 8), num_victims: int = None) -> str:
    """Return game rules string, optionally tailored to agent capabilities."""
    dz = drop_zone
    base_rules = [
        "Rules:",
        f"- Deliver rescued victims to the drop zone at {dz}.",
        "- You can only carry one victim at a time.",
    ]
    if num_victims is not None:
        base_rules.append(f"- There are {num_victims} victims total in the world.")

    if capabilities is None:
        # Generic rules (no capability info)
        base_rules.extend([
            "- Critically injured victims require CarryObjectTogether (both agents).",
            "- Big rocks require RemoveObjectTogether (both agents).",
            "- Trees can only be removed by the rescue robot (RemoveObject).",
            "- Small stones can be removed solo (RemoveObject).",
        ])
    else:
        medical = capabilities.get('medical', 'low')
        strength = capabilities.get('strength', 'medium')

        # Medical rules
        if medical == 'high':
            base_rules.append(
                "- You can carry ALL victims solo (CarryObject), including critically injured."
            )
        else:
            base_rules.append(
                "- Mildly injured victims can be carried solo (CarryObject)."
            )
            base_rules.append(
                "- Critically injured victims require CarryObjectTogether (both agents adjacent)."
            )

        # Strength rules
        if strength == 'high':
            base_rules.extend([
                "- You can remove trees, small stones, and big rocks solo (RemoveObject).",
            ])
        elif strength == 'medium':
            base_rules.extend([
                "- Trees and small stones can be removed solo (RemoveObject).",
                "- Big rocks require RemoveObjectTogether (both agents).",
            ])
        else:  # low
            base_rules.extend([
                "- You can only remove fallen trees solo (RemoveObject).",
                "- Small stones and big rocks are too heavy for you solo, "
                "but you can help remove them cooperatively (RemoveObjectTogether).",
            ])

    return '\n'.join(base_rules)