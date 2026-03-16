# LLM Integration & Async Execution

## Overview

All LLM calls route through a single unified path: `litellm.completion()` in `agents1/async_model_prompting.py`. This module replaced the previous MARBLE-dependent infrastructure with a zero-dependency async execution layer using ThreadPoolExecutor and per-agent Ollama routing.

## Unified Completion Path

**Single entry point**: `agents1/async_model_prompting.py` provides both async (agents) and sync (planner/memory) APIs. Every LLM call uses `_llm_completion()`, which wraps `litellm.completion()` with exponential backoff retry (5 attempts, base 1s wait). LiteLLM handles Ollama/OpenAI format translation.

**No MARBLE imports**: The consolidation removed all MARBLE framework dependencies. `_retry_with_backoff` decorator replaced MARBLE's `error_handler`, implementing the same retry logic without external deps.

## Per-Agent Ollama Routing

**api_base parameter**: Routes each agent to its own Ollama instance. `worlds1/WorldBuilder.py` assigns `api_base=f"http://localhost:{ollama_base_port + agent_nr}"` when creating agents. EnginePlanner uses port 11434 (base). Agent 0 uses 11434, Agent 1 uses 11435, etc. This prevents request queueing and enables true parallel inference.

**Parameter flow**: `WorldBuilder` → `SearchRescueAgent.__init__(api_base)` → `LLMAgentBase.__init__()` (stores as `self._api_base`) → `_submit_llm()` → `submit_llm_call(api_base=self._api_base)` → `litellm.completion(base_url=api_base)`.

## Thread Pool Architecture

**Single shared pool**: `_get_executor()` lazily initializes a `ThreadPoolExecutor` with 8 workers (default) or `max(8, num_agents * 3)` if `init_marble_pool(num_agents)` is called at startup (`main.py`). All agent LLM calls share this pool. Non-blocking: agents submit calls via `submit_llm_call()` and poll via `get_llm_result(future)` each tick.

**EnginePlanner pool**: Separate 4-worker pool (`engine/engine_planner.py`) for task generation, summarization, prefetching, and Q&A. Prefetch strategy: while iteration N runs, tasks for iteration N+1 are generated in background, so `submit_generate_tasks()` returns instantly after the first iteration.

## API Surface

**Async API** (agents): `submit_llm_call()` returns `Future` immediately. `get_llm_result(future)` returns `List[Message]` if done, `None` if in-flight. Used by `LLMAgentBase._submit_llm()` / `_poll_llm_future()`.

**Sync API** (planner/memory): `call_llm_sync()` builds message list (system/few-shot/user), calls `_llm_completion()` blocking, returns text content. Used by `EnginePlanner` (task gen, summarization, Q&A) and `ShortTermMemory` (summarization).

## Retry & Error Handling

**Exponential backoff**: Decorator retries failed calls 5 times with `2^attempt * base_wait_time` delays (1s, 2s, 4s, 8s, 16s). Logs warnings for transient failures, raises on final exhaustion. Handles network errors, model timeouts, malformed responses.

## Response Parsing

**Structured tool calls**: LLMs with tool support return `tool_calls` array. Code extracts first call and JSON-parses `.arguments` if string.

**Text fallback** (`parse_json_response()` in `engine/parsing_utils.py`): Three-stage parser handles non-compliant responses:
1. Extract ` ```json ... ``` ` fenced block via regex
2. Strict `json.loads()` for valid JSON
3. `ast.literal_eval()` for Python dict literals (single-quoted strings)

## Configuration

**Token limits**: `MAX_NR_TOKENS = 3000` per agent call (`agents1/llm_agent_base.py`), `temperature = 0.0` default. EnginePlanner uses 512 tokens for task generation, 5000 for summarization.

**TOON compression** (`engine/toon_utils.py`): World state encoded in Token-Oriented Object Notation (30-60% reduction vs JSON) before prompting.

**Few-shot loading** (`engine/parsing_utils.py`): `load_few_shot(key)` reads `few_shot_examples.yaml` (cached at import). Returns list of message dicts ready for injection between system and user prompts.

## Key Files

- `agents1/async_model_prompting.py` — Core module, all LLM logic
- `engine/parsing_utils.py` — JSON extraction utilities (relocated from old `llm_utils.py`)
- `engine/engine_planner.py` — Uses `call_llm_sync()` for planner LLM calls
- `agents1/llm_agent_base.py` — Stores `_api_base`, calls `submit_llm_call()`
- `worlds1/WorldBuilder.py` — Assigns per-agent `api_base` URLs
- `memory/short_term_memory.py` — Uses `call_llm_sync()` for memory summarization
- `main.py` — Calls `init_marble_pool(num_rescue_agents)` at startup
