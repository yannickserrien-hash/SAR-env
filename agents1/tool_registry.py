"""
Tool registry: @tool-decorated MATRX action functions, reasoning strategy
prompts, game rules, and a schema-builder helper.

All 13 LangChain @tool functions live here so search_rescue_agent.py stays
focused on tick-loop orchestration.  Each tool returns a plain
(action_name, args_dict) tuple — the actual MATRX dispatch and partner_name
enrichment happen in action_dispatch.py at runtime.

Usage:
    from agents1.tool_registry import (
        ALL_ACTION_TOOLS, REASONING_STRATEGIES, GAME_RULES,
        build_tool_schemas,
    )
"""

from typing import Dict, Any, List, Tuple

from langchain.tools import tool


# ── Reasoning strategy system prompts ─────────────────────────────────────────

REASONING_STRATEGIES: Dict[str, str] = {
    'cot': (
        "You are a search and rescue robot. Use Chain-of-Thought reasoning: "
        "think step-by-step about your goal and current situation, "
        "then call the single best action tool."
    ),
    'react': (
        "You are a search and rescue robot. Use ReAct reasoning:\n"
        "Thought: <reason about goal, observations, and constraints>\n"
        "Then call the single best action tool."
    ),
    'reflexion': (
        "You are a search and rescue robot. Use Reflexion reasoning: "
        "before acting, reflect on what you have done and what failed.\n"
        "If a previous action failed, try a completely different approach, "
        "then call the single best action tool."
    ),
}

# Static fallback game rules (used when capabilities module is not available).
# For capability-aware rules, use agents1.capabilities.get_game_rules(caps).
GAME_RULES = (
    "Rules:\n"
    "- Critically injured victims require CarryObjectTogether (both agents).\n"
    "- Big grey rocks require RemoveObjectTogether (both agents).\n"
    "- Trees can only be removed by the rescue robot (RemoveObject).\n"
    "- Small stones can be removed solo (RemoveObject).\n"
    "- Deliver rescued victims to the drop zone at (23, 8).\n"
    "- You can only carry one victim at a time."
)

# ── Action tools ──────────────────────────────────────────────────────────────
# Defined at module level so they don't shadow the aliased MATRX action-class
# imports used elsewhere.  Each tool returns the MATRX action name and the
# LLM-supplied arguments; partner_name enrichment happens in action_dispatch.py.


@tool
def MoveNorth():
    """Move one cell north (decreases y by 1)."""
    return 'MoveNorth', {}, {'task_completing': "moving north"}


@tool
def MoveSouth():
    """Move one cell south (increases y by 1)."""
    return 'MoveSouth', {}, {'task_completing': "moving south"}


@tool
def MoveEast():
    """Move one cell east (increases x by 1)."""
    return 'MoveEast', {}, {'task_completing': "moving east"}


@tool
def MoveWest():
    """Move one cell west (decreases x by 1)."""
    return 'MoveWest', {}, {'task_completing': "moving west"}


@tool
def MoveTo(x: int, y: int, task_completing: str):
    """Navigate to a specific grid coordinate using A* pathfinding.

    Args:
        x: Target column (east-west axis).
        y: Target row (north-south axis).
        task_completing: A brief description of the subtask this move will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'MoveTo', {'x': x, 'y': y, 'task_completing': task_completing}


@tool
def NavigateToDropZone(task_completing: str = "navigating to drop zone"):
    """Navigate to the rescue drop zone at grid position (23, 8) to deliver a carried victim."""
    return 'NavigateToDropZone', {'task_completing': task_completing}


@tool
def CarryObject(object_id: str, task_completing: str = "carrying victim"):
    """Pick up and carry a mildly injured victim or small item. Must be adjacent (distance ≤ 1).

    Args:
        object_id: The ID of the object to carry (from nearby observations).
        task_completing: A brief description of the subtask this action will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'CarryObject', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def CarryObjectTogether(object_id: str, task_completing: str = "carrying victim cooperatively"):
    """Cooperatively carry a critically injured victim or big grey rock with a partner agent.
    Both agents must be adjacent to the object.

    Args:
        object_id: The ID of the object to carry cooperatively.
        task_completing: A brief description of the subtask this action will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'CarryObjectTogether', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def Drop():
    """Drop the currently carried object at the current grid position."""
    return 'Drop', {}, {'task_completing': "dropping carried victim"}


@tool
def DropObjectTogether():
    """Drop an object that is being cooperatively carried with a partner agent."""
    return 'DropObjectTogether', {}, {'task_completing': "dropping carried victim cooperatively"}


@tool
def RemoveObject(object_id: str, task_completing: str = "removing obstacle"):
    """Remove a small stone or tree obstacle from the grid. Must be adjacent (distance ≤ 1).
    Note: only the rescue robot can remove trees; either agent can remove small stones.

    Args:
        object_id: The ID of the obstacle to remove.
        task_completing: A brief description of the subtask this action will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'RemoveObject', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def RemoveObjectTogether(object_id: str, task_completing: str = "removing obstacle cooperatively"):
    """Cooperatively remove a big grey rock obstacle with a partner agent.
    Both agents must be adjacent to the rock.

    Args:
        object_id: The ID of the rock obstacle to remove cooperatively.
        task_completing: A brief description of the subtask this action will complete
                        (e.g. "approaching victim to carry" or "navigating to drop zone").
    """
    return 'RemoveObjectTogether', {'object_id': object_id}, {'task_completing': task_completing}


@tool
def Idle(duration_in_ticks: int = 1):
    """Do nothing for the specified number of ticks.
    Use when waiting for a partner agent or an expected event.

    Args:
        duration_in_ticks: Number of ticks to wait (default 1).
    """
    return 'Idle', {'duration_in_ticks': duration_in_ticks}, {'task_completing': f"idling for {duration_in_ticks} ticks"}

@tool
def SendInfoFromMemory(information: str, memory:Dict[str, Any]):
    """Retrieve relevant information from memory

    Args:
        information: A description of the current situation or what information is needed.
        memory: The agent's memory storage to search through.
    """
    # send_message(Message(content=f"Retrieving information from memory: {information}", from_id=self.agent_id))
    
    return 'Idle', {'duration_in_ticks': 1} , {'task_completing': f"retrieving information from memory about {information}"}

@tool
def SendMessage(message: str, send_to: str, message_type: str = "message"):
    """Send a message to one or all teammates. This uses your action for this tick.

    Args:
        message: The message content to send.
        send_to: Agent name for directed message, or "all" for broadcast.
        message_type: One of: ask_help (expects reply), help (response to ask_help), message (general).
    """
    return 'SendMessage', {'message': message, 'send_to': send_to,
        'message_type': message_type}, {'task_completing': f"sending {message_type} message"}


# Ordered list of every action tool — used to build the registry + LLM schemas.
ALL_ACTION_TOOLS = [
    MoveNorth, MoveSouth, MoveEast, MoveWest,
    MoveTo, NavigateToDropZone,
    CarryObject, CarryObjectTogether,
    Drop, DropObjectTogether,
    RemoveObject, RemoveObjectTogether,
    Idle, SendMessage
]


# ── Schema builder ────────────────────────────────────────────────────────────

def build_tool_schemas() -> Tuple[Dict[str, Any], List[Dict]]:
    """Build (tools_by_name, tool_schemas) from ALL_ACTION_TOOLS.

    Returns:
        tools_by_name:  ``{name: StructuredTool}`` lookup dict.
        tool_schemas:   OpenAI-compatible tool schema list for LiteLLM.
    """
    import logging
    from langchain_core.utils.function_calling import convert_to_openai_tool

    logger = logging.getLogger('tool_registry')

    tools_by_name: Dict[str, Any] = {t.name: t for t in ALL_ACTION_TOOLS}

    try:
        tool_schemas: List[Dict] = [
            convert_to_openai_tool(t) for t in ALL_ACTION_TOOLS
        ]
    except Exception as exc:
        logger.warning(
            "convert_to_openai_tool failed (%s); using matrx_tool_description fallback", exc
        )
        from agents1.agents_graveyard.matrx_tool_description import ALL_TOOLS
        tool_schemas = ALL_TOOLS

    return tools_by_name, tool_schemas
