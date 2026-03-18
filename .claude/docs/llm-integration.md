# LLM Integration & Async Prompting

## Overview

This codebase uses Ollama for local LLM inference with a fully asynchronous architecture. All LLM calls run in background threads via `concurrent.futures.ThreadPoolExecutor`, allowing agents to submit prompts and continue operating while responses generate. The system supports both text generation and structured tool-calling, with fallback parsing for non-compliant responses.

## Architecture Pattern

**Non-blocking submission + polling**: Agents submit LLM calls and receive a `Future` immediately. Each simulation tick, they poll `future.done()` and idle until the result arrives. This prevents blocking the game loop while multiple agents reason in parallel.

**Thread pool sizing**: Pools auto-scale based on agent count. Formula: `max(4, num_agents * 3)` workers, allowing ~3 concurrent LLM calls per agent (reasoning + memory extraction + communication).

## Core Components

### Client Pooling (engine/llm_utils.py)

Thread-safe Ollama client cache (`_client_cache`) ensures one `Client` instance per host URL. The `_get_client()` function uses a lock to prevent race conditions during initialization. Clients persist across all calls to the same Ollama endpoint.

### Query Functions

- **`query_llm()`**: Synchronous text generation via `/api/generate` endpoint. Constructs prompts by concatenating system prompt, few-shot examples, and user prompt with custom delimiters.
- **`query_llm_async()`**: Wraps `query_llm()` in `executor.submit()`, returns `Future` immediately.
- **`query_llm_with_tools()`**: Structured tool-calling via Ollama SDK's `client.chat()`. Returns `{"name": str, "arguments": dict}` or falls back to plain text.
- **`query_llm_with_tools_async()`**: Non-blocking version for tool calls.

### Thread Pool Management

**Engine LLM pool** (`_llm_executor` in llm_utils.py): Call `init_llm_pool(num_agents)` at startup to size the pool. Workers named `llm_worker-N`. Used by engine planner and rescue agents via the `llm_utils` module.

**MARBLE pool** (agents1/async_model_prompting.py): Separate executor for MARBLE-based agents. Call `init_marble_pool(num_agents)` to resize. Workers named `marble_llm-N`. Wraps MARBLE's `model_prompting()` with retry/backoff.

### Request/Response Lifecycle

1. **Agent tick**: `_submit_llm(messages, tools)` calls `submit_llm_call()` → `Future` stored in `self._pending_future`
2. **Agent returns Idle**: Agent yields control to MATRX with `_idle()` action
3. **Next tick**: `check_if_llm_response_ready()` checks `future.done()`
4. **Result ready**: `get_llm_result(future)` retrieves `List[Message]` or `None`
5. **Parse response**:
   - Path A: Extract `tool_calls[0].function.name` and `.arguments`
   - Path B: Fallback to `ActionMapper.parse_raw(text)` for JSON extraction
6. **Validation**: `_validate_action()` checks object_id exists in WORLD_STATE, `_check_matrx_action()` verifies feasibility
7. **Execution**: `execute_action()` maps to MATRX action class, agent returns `(action_name, kwargs)`

## Response Parsing

**Structured tool calls**: LLMs trained with tool support return `tool_calls` array. Codebase extracts first call and JSON-parses `.arguments` if string.

**Text fallback** (`parse_json_response()` in llm_utils.py): Three-stage parser handles non-compliant responses:
1. Extract ` ```json ... ``` ` fenced block via regex
2. Strict `json.loads()` for valid JSON
3. `ast.literal_eval()` for Python dict literals (handles single-quoted strings from LLMs)

If parsing fails, agent re-submits on next tick with error feedback.

## Token Budget & Configuration

**Default limits**: `MAX_NR_TOKENS = 3000` per agent call (agents1/llm_agent_base.py), `temperature = 0.3` for reasoning, `0.1` for tool calls.

**Per-call overrides**: All query functions accept `max_tokens` and `temperature` parameters. Engine planner uses lower budgets (512 tokens) for task assignment, higher (5000) for summarization.

**TOON compression** (engine/toon_utils.py): World state encoded in Token-Oriented Object Notation before prompting. Achieves 30-60% token reduction vs JSON through indented key:value syntax and inline arrays. LLMs show higher accuracy reading TOON vs JSON in benchmarks.

## Few-Shot Loading

**few_shot_examples.yaml**: Centralized prompt examples for reasoning, planning, memory extraction. Loaded once at import time into `_few_shot_cache`.

**Usage**: `load_few_shot('reasoning')` returns `List[Dict[str, str]]` ready for injection between system and user prompts. Returns `[]` if key missing.

**Format**: Each example is `{"user": "...", "assistant": "..."}` pairs converted to OpenAI message format.

## Prefetch Pattern (Engine Planner)

Engine planner uses background prefetching for task generation. At end of iteration N, it submits LLM call for iteration N+1 tasks while agents execute current ticks. When `generate_tasks()` is called, the prefetched `Future` is already done, eliminating wait time. This overlaps LLM latency with simulation execution.

Implementation: `_prefetch_future` stores next iteration's task generation. `submit_summarize()` kicks off prefetch after summary completes.

## Error Handling

**Connection failures**: `requests.ConnectionError` caught and logged. Returns `None` to caller, agent re-submits next tick.

**Timeouts**: 60s timeout on HTTP requests. Returns `None`, no retry at LLM utils level (agents handle retry).

**Parsing failures**: Invalid JSON logs warning with first 200 chars of response. Agent receives feedback string in next prompt.

**Future exceptions**: `check_if_llm_response_ready()` wraps `future.result()` in try/except, logs exception, resets agent to reasoning state.

## Key Files

- **engine/llm_utils.py**: Client cache, sync/async query functions, JSON parsing, few-shot loader
- **agents1/async_model_prompting.py**: MARBLE executor wrapper, submit/poll API
- **agents1/llm_agent_base.py**: Agent-side LLM submission, future polling, response handling, action validation
- **engine/toon_utils.py**: Token-efficient encoding (30-60% reduction vs JSON)
- **few_shot_examples.yaml**: Centralized prompt examples (reasoning, planning, Q&A)
- **engine/engine_planner.py**: Prefetch pattern for overlapping LLM calls with simulation ticks

## Configuration

Set `OLLAMA_BASE_URL` in llm_utils.py or pass `api_url` parameter to query functions. Default: `http://localhost:11434`.

Agent LLM model specified via `llm_model` constructor parameter (e.g., `'ollama/llama3'` for MARBLE agents, `'qwen3:8b'` for engine planner).
