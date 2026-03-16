# Agent Decision Cycle & Tool Calling

## Overview
Agent reasoning follows a structured tick-by-tick cycle: perceive filtered state, build TOON-compressed prompt with task/observations/memory, submit async LLM call via LiteLLM with OpenAI-compatible tool schemas, poll for result (structured tool_call or plain-text JSON), validate against WORLD_STATE, dispatch to MATRX action. Fallback parsing handles LLMs that don't support tool calling.

## Decision Loop Architecture

**SearchRescueAgent.decide_on_actions()** (agents1/search_rescue_agent.py) orchestrates the tick-by-tick cycle:

1. **Setup**: `_tick_setup()` updates state tracker and builds `WORLD_STATE` dict (agent location, nearby objects within 1-block Chebyshev radius, carried items)
2. **Infrastructure preamble**: `_run_preamble()` handles carry retry loops, ongoing A* navigation, rendezvous, and LLM future polling before agent reasoning
3. **Reasoning step**: If `_reasoning_step` flag is true and tasks remain, build prompt and submit async LLM call
4. **Idle**: Return Idle action if infrastructure is active or no reasoning needed

The system maintains persistent state across ticks: `_pending_future` (in-flight LLM call), `_nav_target` (A* destination), `_pending_carry_kwargs` (cooperative carry retry state), `_action_feedback` (validation errors for next prompt).

## Tool Registry & Schema Generation

**agents1/tool_registry.py** defines 14 LangChain `@tool`-decorated functions representing the agent's action space:

- Movement: `MoveNorth`, `MoveSouth`, `MoveEast`, `MoveWest`, `MoveTo`, `NavigateToDropZone`
- Object interaction: `CarryObject`, `CarryObjectTogether`, `Drop`, `DropObjectTogether`
- Obstacle removal: `RemoveObject`, `RemoveObjectTogether`
- Utility: `Idle`, `SendMessage`

Each tool returns a `(action_name, args_dict)` tuple. Game rules are embedded in docstrings (critically injured victims require CarryObjectTogether, big rocks require RemoveObjectTogether, only rescue robot can remove trees).

**build_tool_schemas()** converts tools to OpenAI-compatible schemas via LangChain's `convert_to_openai_tool()`. These schemas are passed to LiteLLM's `tools` parameter during LLM calls. Reasoning strategies (cot/react/reflexion) are stored in `REASONING_STRATEGIES` dict and injected as system prompts.

## Prompt Construction

**ReasoningIO.get_reasoning_prompt()** (agents1/modules/reasoning_module.py) builds a 2-message list:
- System message: Reasoning strategy prompt + game rules
- User message: TOON-encoded dict with keys `observation`, `tasks`, `feedback`, `memory`

**TOON compression** (agents1/modules/utils_prompting.py): `to_toon()` achieves 30-60% token reduction vs JSON by encoding dicts as indented `key: value` pairs and arrays as `key[N]: v1,v2,v3`. Uniform object arrays use tabular format with field headers.

Observations merge local filtered state with globally known objects from `WORLD_STATE_GLOBAL` (SharedMemory). Memory retrieves last 15 entries. Action feedback from previous tick's validation failures is included.

## LLM Submission & Polling

**Async submission** (agents1/async_model_prompting.py): `submit_llm_call()` wraps `litellm.completion()` in a ThreadPoolExecutor (8 workers by default, resizable via `init_marble_pool()`). Each agent has a dedicated Ollama instance on port 11434+N specified via `api_base` parameter.

**Non-blocking poll**: `get_llm_result()` checks `Future.done()` without blocking. If still running, agent returns Idle and polls again next tick. On completion, returns `List[Message]` with tool_call or text content.

Retry logic: `_retry_with_backoff()` decorator retries failed LLM calls 5 times with exponential backoff (1s, 2s, 4s, 8s, 16s).

## Tool Call vs Fallback Parsing

**LLMAgentBase._handle_llm_result()** (agents1/llm_agent_base.py) handles two response paths:

**Path A: Structured tool_call** (preferred)
- Extract `message.tool_calls[0].function.name` and `arguments` (JSON string or dict)
- Validate object_id against nearby objects in WORLD_STATE via `_validate_action()`
- Check MATRX feasibility via `is_action_possible()` in `_check_matrx_action()`
- Dispatch to MATRX action via `execute_action()` (agents1/modules/execution_module.py), which enriches cooperative actions with `partner_name`
- Update planner task state and memory
- Apply A* navigation for MoveTo/NavigateToDropZone

**Path B: Plain-text JSON fallback**
- Extract JSON from text via `ActionMapper.parse_raw()` (agents1/action_mapper.py)
- Tries ````json ... ```` fenced blocks, then first `{ ... }` span, then Python `ast.literal_eval()` for single-quoted dicts
- Same validation and dispatch pipeline as Path A

**Validation edge cases**: `_validate_action()` populates `_action_feedback` string when object_id is not in nearby objects list. Includes nearby actionable objects summary and agent location for next LLM prompt. `_check_matrx_action()` verifies MATRX physics (adjacency, carry capacity). Both validations return Idle action on failure and set `_reasoning_step = True` to retry next tick.

## Stale Observation Handling

State freshness is maintained per-tick:
- `filter_observations()` rebuilds `WORLD_STATE` every tick before reasoning
- A* navigation uses unfiltered `state_for_navigation` (full grid visibility)
- SharedMemory publishes carry/obstacle events immediately after action execution via `_maybe_share_observation()`
- Cooperative carry retry loop refreshes partner position from SharedMemory each tick

No explicit staleness detection—agents always reason on current tick's filtered state.

## Configuration

- `MAX_NR_TOKENS = 8192` (agents1/llm_agent_base.py): LLM completion limit
- `TEMPERATURE = 0.0`: Deterministic tool calls
- `CARRY_WAIT_TIMEOUT_TICKS = 5`: Cooperative carry retry limit before abandoning
- Tool choice: `'auto'` when tools provided, `'none'` otherwise
