# Error Handling & Edge Cases

## Overview

The SAR-env system handles errors at four layers: LLM call failures (retry with exponential backoff), action feasibility mismatches (validation + MATRX checks), cooperative action deadlocks (timeout-based recovery), and parsing failures (JSON fallback + Idle default). All error feedback flows back into the LLM prompt via `_action_feedback` to enable self-correction.

## LLM Call Failures

**Retry with exponential backoff**: `agents1/async_model_prompting.py` wraps `litellm.completion()` in `_retry_with_backoff()` decorator (5 retries, base wait 1s, exponential backoff 2^attempt). Retries all transient errors (network timeouts, Ollama server hiccups). Final failure after 5 attempts raises exception caught by `_poll_llm_future()`.

**Async future polling**: `LLMAgentBase._poll_llm_future()` polls the ThreadPoolExecutor future non-blocking. On exception, logs warning, resets `_pending_future`, sets `_reasoning_step = True`, returns Idle. Agent will re-submit LLM call next tick with fresh prompt.

## Action Validation & Feasibility

**Two-stage validation**: Every object-based action (`CarryObject`, `RemoveObject`, `CarryObjectTogether`, `RemoveObjectTogether`) passes through `_validate_action()` then `_check_matrx_action()` before dispatch.

**Stage 1 - Object existence check** (`agents1/llm_agent_base.py:_validate_action()`): Verifies `object_id` exists in `WORLD_STATE['nearby']` (1-block Chebyshev radius). If missing or empty, populates `_action_feedback` with detailed error message listing all nearby actionable objects + agent location, logs warning, returns Idle. Feedback injected into next LLM prompt at `search_rescue_agent.py`.

**Stage 2 - MATRX feasibility check** (`agents1/llm_agent_base.py:_check_matrx_action()`): Calls MATRX's `is_action_possible()` (inherited from `ArtificialBrain`) which queries simulation state via `callback_is_action_possible`. Skipped for navigation (`MoveTo`, `MoveNorth`, etc.), messaging, and `CarryObjectTogether` (handled by carry retry loop). Non-victim objects rejected preemptively. Failures populate `_action_feedback` with MATRX rejection reason + nearby objects list.

**Feedback loop**: `_action_feedback` cleared after each LLM submission (`search_rescue_agent.py`). Empty string when action succeeds. Agent learns from failures via prompt context.

## Parsing Failures

**Tool call path** (`agents1/llm_agent_base.py:_handle_llm_result()`): If LLM returns `tool_calls`, extracts first tool call, parses `function.arguments` (JSON string → dict). If parsing fails, falls through to text fallback.

**Text fallback** (`agents1/action_mapper.py:_extract_json()`): Three-stage JSON extraction:
1. Fenced code block: ````json {...} ```
2. Outermost `{...}` span with strict `json.loads()`
3. Python dict literal via `ast.literal_eval()` (handles single-quoted LLM output)

Returns `None` on all failures. `ActionMapper.parse_raw()` returns `(None, {})`, caller checks and returns Idle. Agent re-reasons next tick.

## Cooperative Action Deadlocks

**Carry retry loop** (`agents1/llm_agent_base.py:_handle_carry_retry()`): After initiating `CarryObjectTogether`, agent enters wait state (`_pending_carry_kwargs` set, `_reasoning_step = False`). Each tick:
- Check if victim still in range (delivered → exit loop)
- Increment `_carry_wait_ticks` counter
- Timeout after 100 ticks: populate `_action_feedback` with partner_timeout reason, log to `memory.update('carry_failure')`, clear rendezvous, return Idle

**Rendezvous mechanism** (`agents1/llm_agent_base.py:_handle_rendezvous()`): Waiting agent publishes location to `SharedMemory['carry_rendezvous']`. Partner navigates if `status == 'waiting_for_partner'` and not already adjacent. Cleared on success or timeout. Thread-safe via `threading.Lock` in `memory/shared_memory.py`.

## Stale Observations

**Perception refresh**: `LLMAgentBase._tick_setup()` updates `StateTracker` and rebuilds `WORLD_STATE` every tick from fresh filtered state. `Perception.process_observations()` merges new observations into `WORLD_STATE_GLOBAL` (persistent memory of all seen objects). Object locations updated on each sighting.

**Missing objects**: If agent acts on an object no longer in range, `_validate_action()` catches it before MATRX sees the action. Feedback lists current nearby objects so LLM can choose a valid alternative.

## Thread Race Conditions

**SharedMemory thread-safety**: `memory/shared_memory.py` uses `threading.Lock()` for all `update()` and `retrieve()` operations. Lock acquired before dict read/write, released immediately after. Prevents concurrent writes from multiple agent threads.

**Async LLM isolation**: Each agent has independent `_pending_future`. ThreadPoolExecutor isolates LLM calls. No shared mutable state between agent reasoning threads. World state updates serialized by MATRX tick loop (single-threaded grid world updates).

## Edge Cases

**Empty task**: `_current_task` is `None` → agent returns Idle until EnginePlanner injects task via `set_current_task()`.

**Navigation target unreachable**: A* pathfinding (`Navigator.get_move_action()`) returns `None` if no path exists → `_handle_navigation_tick()` clears `_nav_target`, sets `_reasoning_step = True`, agent re-reasons. No explicit feedback; LLM infers from lack of progress.

**Duplicate carry attempts**: `SharedMemory` publish prevents multiple agents waiting on same victim (first to publish wins). Second agent sees `carry_rendezvous['agent'] != self.agent_id` and navigates to assist.

**LLM returns no action**: Text parser returns `(None, {})` → caller returns Idle, agent re-submits prompt next tick with existing task.

## Debugging

**Logging**: All validation failures log to `LLMAgentBase` logger at WARNING level with agent_id prefix. LLM retry attempts log to `async_model_prompting` logger.

**Action feedback inspection**: Print `_action_feedback` in agent's reasoning prompt (already done in `search_rescue_agent.py`) to see what errors the agent receives.

**Carry timeout tuning**: `CARRY_WAIT_TIMEOUT_TICKS = 100` constant in `agents1/llm_agent_base.py`. Increase if agents move slowly; decrease to fail fast.
