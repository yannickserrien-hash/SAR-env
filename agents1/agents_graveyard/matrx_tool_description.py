"""
OpenAI-style tool / function-calling descriptions for every action
the MATRX rescue-agent LLM can invoke.

Usage:
    from agents1.matrx_tool_description import ALL_TOOLS, REASONING_TOOLS
    # pass ALL_TOOLS (or a subset) as the ``tools`` parameter to the
    # Ollama /api/chat endpoint.
"""

# ── Movement ────────────────────────────────────────────────────────────────

MoveNorth_description = {
    "type": "function",
    "function": {
        "name": "MoveNorth",
        "description": "Move one cell north.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MoveSouth_description = {
    "type": "function",
    "function": {
        "name": "MoveSouth",
        "description": "Move one cell south.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MoveEast_description = {
    "type": "function",
    "function": {
        "name": "MoveEast",
        "description": "Move one cell east.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MoveWest_description = {
    "type": "function",
    "function": {
        "name": "MoveWest",
        "description": "Move one cell west.",
        "parameters": {"type": "object", "properties": {}},
    },
}

MoveTo_description = {
    "type": "function",
    "function": {
        "name": "MoveTo",
        "description": "Navigate to a specific grid position using A* pathfinding.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The x coordinate to move to.",
                },
                "y": {
                    "type": "number",
                    "description": "The y coordinate to move to.",
                },
            },
            "required": ["x", "y"],
        },
    },
}

NavigateToDropZone_description = {
    "type": "function",
    "function": {
        "name": "NavigateToDropZone",
        "description": "Navigate to the drop zone where rescued victims are delivered.",
        "parameters": {"type": "object", "properties": {}},
    },
}

# ── Object manipulation (solo) ──────────────────────────────────────────────

CarryObject_description = {
    "type": "function",
    "function": {
        "name": "CarryObject",
        "description": "Pick up and carry an object. Use for mildly injured victims or movable items. Must be adjacent to the object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the object to carry (from nearby observations).",
                },
            },
            "required": ["object_id"],
        },
    },
}

Drop_description = {
    "type": "function",
    "function": {
        "name": "Drop",
        "description": "Drop a carried object at the current location.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the object to drop. If omitted, drops the last picked-up item.",
                },
            },
        },
    },
}

RemoveObject_description = {
    "type": "function",
    "function": {
        "name": "RemoveObject",
        "description": "Remove an obstacle (small stone or tree) from the grid. Must be adjacent. Trees can only be removed by the rescue agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the obstacle to remove.",
                },
            },
            "required": ["object_id"],
        },
    },
}

# ── Object manipulation (cooperative) ───────────────────────────────────────

CarryObjectTogether_description = {
    "type": "function",
    "function": {
        "name": "CarryObjectTogether",
        "description": "Cooperatively carry an object with the nearest partner agent. Required for critically injured victims and big grey rocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the object to carry together.",
                },
            },
            "required": ["object_id"],
        },
    },
}

DropObjectTogether_description = {
    "type": "function",
    "function": {
        "name": "DropObjectTogether",
        "description": "Drop an object that is being carried cooperatively with a partner agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the object to drop. If omitted, drops the last cooperatively carried item.",
                },
            },
        },
    },
}

RemoveObjectTogether_description = {
    "type": "function",
    "function": {
        "name": "RemoveObjectTogether",
        "description": "Cooperatively remove an obstacle (big grey rock) with a partner agent. Both agents must be adjacent.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The ID of the obstacle to remove together.",
                },
            },
            "required": ["object_id"],
        },
    },
}

# ── Communication ───────────────────────────────────────────────────────────

SendMessage_description = {
    "type": "function",
    "function": {
        "name": "SendMessage",
        "description": "Broadcast a help request to all agents, specifying a location and the action needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_x": {
                    "type": "number",
                    "description": "The x coordinate where help is needed.",
                },
                "target_y": {
                    "type": "number",
                    "description": "The y coordinate where help is needed.",
                },
                "action_needed": {
                    "type": "string",
                    "description": "The cooperative action needed (e.g. 'RemoveObjectTogether', 'CarryObjectTogether').",
                },
                "object_id": {
                    "type": "string",
                    "description": "The ID of the object that requires cooperative action.",
                },
            },
            "required": ["target_x", "target_y", "action_needed", "object_id"],
        },
    },
}

SendDirectMessage_description = {
    "type": "function",
    "function": {
        "name": "SendDirectMessage",
        "description": "Send a direct message to a specific agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "description": "The ID of the agent to message.",
                },
                "message": {
                    "type": "string",
                    "description": "The message content to send.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about why the message is being sent.",
                },
            },
            "required": ["target_agent", "message"],
        },
    },
}

# ── Meta / control ──────────────────────────────────────────────────────────

Replan_description = {
    "type": "function",
    "function": {
        "name": "Replan",
        "description": "Request a new plan because the current plan cannot make progress. Use when stuck, blocked, or the situation has changed.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Explanation of why replanning is needed.",
                },
            },
            "required": ["reason"],
        },
    },
}

TaskComplete_description = {
    "type": "function",
    "function": {
        "name": "TaskComplete",
        "description": "Signal that the current task is fully completed and request a new task.",
        "parameters": {"type": "object", "properties": {}},
    },
}

Idle_description = {
    "type": "function",
    "function": {
        "name": "Idle",
        "description": "Do nothing for a specified number of ticks. Use when waiting for another agent or event.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration_in_ticks": {
                    "type": "number",
                    "description": "Number of ticks to idle (default 1).",
                },
            },
        },
    },
}

# ── Convenience lists ───────────────────────────────────────────────────────

ALL_TOOLS = [
    MoveNorth_description,
    MoveSouth_description,
    MoveEast_description,
    MoveWest_description,
    MoveTo_description,
    NavigateToDropZone_description,
    CarryObject_description,
    CarryObjectTogether_description,
    Drop_description,
    DropObjectTogether_description,
    RemoveObject_description,
    RemoveObjectTogether_description,
    SendMessage_description,
    SendDirectMessage_description,
    Replan_description,
    TaskComplete_description,
    Idle_description,
]

# Tools used during per-tick reasoning (action selection).
# Excludes Idle — the framework returns Idle automatically when needed.
REASONING_TOOLS = [t for t in ALL_TOOLS if t["function"]["name"] != "Idle"]
