# Tool Registry & Game Rules

## Overview

The tool registry defines the 16 LangChain `@tool`-decorated functions that LLM agents can invoke during search-and-rescue missions. Each tool corresponds to a MATRX action class and includes parameter schemas, docstrings, and validation logic. The game rules are embedded in tool docstrings, capability checks in `CustomActions.py`, and scoring logic in `WorldBuilder.py`.

## Tool Definition (`agents1/tool_registry.py`)

All 16 tools live in a single module to separate action definitions from agent orchestration logic. Each tool returns `(action_name, args_dict, metadata_dict)` — the actual MATRX dispatch happens later in `agents1/action_dispatch.py`.

**Tool categories:**
- **Movement** (8): MoveNorth/South/East/West, MoveTo, MoveToArea, EnterArea, NavigateToDropZone
- **Object handling** (5): CarryObject, CarryObjectTogether, Drop, RemoveObject, RemoveObjectTogether
- **Support** (3): SearchArea, Idle, SendMessage

**Cooperative actions** (CarryObjectTogether, RemoveObjectTogether) require two agents adjacent to the target object. The `partner_id` parameter specifies which teammate to cooperate with. The LLM supplies `partner_id`; dispatch logic enriches it to `partner_name` (internal MATRX parameter).

**Schema generation:** `build_tool_schemas()` converts LangChain tools to OpenAI-compatible JSON schemas for Ollama using `convert_to_openai_tool()`.

## Game Rules Enforcement

Rules are defined in three layers:

### 1. Tool Docstrings (LLM-visible)

`get_game_rules_str()` returns static fallback rules. Capability-aware rules come from `agents1/capabilities.get_game_rules()`. Example rules:
- Critically injured victims require CarryObjectTogether
- Big rocks require RemoveObjectTogether  
- Trees can only be removed by rescue robots
- Only one victim carried at a time

### 2. Environment Enforcement (`actions1/CustomActions.py`)

Custom action classes extend MATRX base actions with capability checks in `is_possible()` methods:

**CarryObject.is_possible()** (lines 331-384):
- Blocks obstacles (stones, rocks, trees) — line 369
- Medical capability check: critically injured victims need `medical='high'` (lines 373-381)
- Falls back to MATRX `_is_possible_grab()` for standard validation

**RemoveObjectTogether.is_possible()** (lines 202-258):
- Requires both agents within range of the rock (lines 183-195)
- Strength capability check: big rocks need `strength >= medium` (enforced by tool availability filtering, not in this method)

**CarryObjectTogether.is_possible()** (lines 779-835):
- Verifies calling agent is adjacent to object (line 819)
- Finds nearest partner using `_find_partner_agent()` (line 824)
- Checks partner is also adjacent (line 829)

**Solo RemoveObject** strength enforcement happens in `brains1/ArtificialBrain.decide_on_action()` via capability-aware delays.

### 3. Capability-based Tool Filtering (`agents1/capabilities.py`)

`filter_tools_for_capabilities()` (lines 144+) removes tools agents can NEVER use based on their preset. For example, a scout with `strength='low'` cannot see RemoveObject for big rocks.

## Action Dispatch (`agents1/action_dispatch.py`)

`execute_action()` maps LLM tool calls to MATRX-ready (action_class_name, kwargs) tuples. Cooperative actions extract `partner_id` from LLM args and inject `partner_name` for MATRX (lines 99, 107).

## Scoring System (`worlds1/WorldBuilder.py`)

**CollectionGoal.__check_completion()** (lines 478-556):
- **Critical victims**: +6 points (line 517)
- **Mild victims**: +3 points (line 520)
- **Healthy victims**: 0 points (excluded, line 513)

Scoring only triggers when victims are delivered to the drop zone (`drop_zone_x`, `drop_zone_y_min` to `drop_zone_y_max`). Each victim is scored once via `_scored_victims` set (line 524). Score updates are written to `logs/score.json` for EnginePlanner consumption.

## Validation Pipeline

**Order of checks** (per tick):
1. **Agent chooses tool** → SearchRescueAgent submits LLM call
2. **Tool execution** → `action_dispatch.execute_action()` maps to MATRX action
3. **Pre-validation** → `agents1/llm_agent_base.py` checks action feasibility via `CustomActions.is_possible()` (includes capability checks)
4. **MATRX execution** → If valid, action mutates grid world
5. **Post-action** → Action result returned to agent for next tick's reasoning

Validation failures (e.g., "You may not have the required ability to carry this victim alone") are fed back into the LLM prompt as `last_action_result`.

## Key Files

- **agents1/tool_registry.py** — 16 tool definitions, schemas, reasoning strategies
- **actions1/CustomActions.py** — Capability enforcement, cooperative action logic
- **agents1/action_dispatch.py** — Tool-to-action mapping, partner_name enrichment
- **agents1/capabilities.py** — Presets, capability-aware game rules, tool filtering
- **worlds1/WorldBuilder.py** — CollectionGoal scoring logic (lines 373-556)
- **agents1/llm_agent_base.py** — Action validation before MATRX dispatch
