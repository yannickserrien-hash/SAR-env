# Error Handling & Recovery

## Overview

SAR-env handles failures at three layers: LLM execution (timeouts, malformed responses), action validation (pre-dispatch checks), and cooperative action coordination (partner availability). Recovery is automatic where possible, with fallback behaviors preventing simulation deadlock.

## LLM Failure Modes

### Retry with Exponential Backoff

**Location**: `agents1/async_model_prompting.py`

All LLM calls (both agent reasoning and EnginePlanner coordination) use a unified retry decorator with exponential backoff: 5 attempts, starting at 1s delay, doubling each retry. Handles connection errors, timeouts, and Ollama HTTP failures.

```python
@_retry_with_backoff(retries=5, base_wait_time=1.0)
def _llm_completion(llm_model, messages, ...):
    # Retries on any exception
```

HTTP backend sets 120s timeout on requests (`agents1/async_model_prompting.py:224`). After all retries exhausted, `call_llm_sync()` returns `None` and logs error; async paths propagate exception to caller.

### Malformed Response Parsing

**Location**: `engine/parsing_utils.py`

`parse_json_response()` uses 3-stage fallback for EnginePlanner task generation:

1. Extract fenced ```json block
2. Parse raw JSON (strict double quotes)
3. Use `ast.literal_eval` for single-quoted Python dict literals

Returns `None` on failure. EnginePlanner falls back to hardcoded default tasks: "explore area 1 and find victims" for all agents (`engine/engine_planner.py:331-337`).

Agent tool-call parsing: Ollama SDK responses are normalized to `_Message` dataclass with `tool_calls` list. If LLM returns plain text instead of tool call, agent logs warning and idles (`agents1/agent_sar.py:143-148`).

### Future Polling Exceptions

**Location**: `agents1/agent_sar.py`, `agents1/modules/communication_module.py`

Agents poll LLM futures via `get_llm_result()` wrapped in try-except. On exception, future is cleared and agent idles that tick:

```python
try:
    result = get_llm_result(self._pending_future)
except Exception as exc:
    logger.warning('[%s] LLM future raised: %s', self.agent_id, exc)
    self._pending_future = None
    return self._idle()  # safe fallback
```

No infinite loops — agent retries reasoning on next tick with updated observations.

## Action Validation Pipeline

**Location**: `helpers/logic_module.py` (`ActionValidator`)

Pre-dispatch validation catches LLM mistakes before actions reach MATRX engine. Returns `ValidationResult(valid=False, feedback="...")` with LLM-friendly error message. Agent writes feedback to memory and idles.

**Common validation failures**:
- Out-of-bounds movement (grid edges)
- Invalid object ID or object not adjacent
- Already carrying another object
- Capability violations (informed mode only): low-medical agent trying to solo-carry critical victim, low-strength agent trying to remove rock
- Cooperative action without adjacent partner
- Empty SendMessage content

Validation runs in `LLMAgentBase._validate_action()` before `execute_action()`. Invalid actions return `Idle` instead of propagating to MATRX, preventing engine errors.

### Capability Enforcement

Two-layer enforcement (Section 3 of project-specification):

1. **Validator (informed mode)**: Pre-checks capabilities, blocks invalid actions at LLM layer
2. **Environment (all modes)**: `CustomActions.py` `is_possible()` methods enforce strength/medical rules. Used in discovery mode for trial-and-error learning.

Failed `is_possible()` checks return MATRX `ActionResult(succeeded=False)` which agent detects via `previous_action_result`.

## Cooperative Action Coordination

**Location**: `agents1/llm_agent_base.py` (carry autopilot system)

### Successful Carry Completion

When `CarryObjectTogether` succeeds (detected via `previous_action_result.succeeded`), both agents enter autopilot mode:

- **Carrier**: Victim in inventory, navigates to drop zone, executes `DropObject` on arrival
- **Partner**: SharedMemory signals autopilot start, partner joins navigation to drop zone, idles on arrival while carrier drops

Autopilot overrides normal reasoning pipeline until victim delivered. No timeout — agents navigate indefinitely until destination reached.

### Partner Unavailability

`ActionValidator` checks partner adjacency before submitting `CarryObjectTogether` or `RemoveObjectTogether`. If partner not found or not adjacent, validation returns feedback: "Ask a teammate for help using SendMessage(message_type='ask_help'), then wait for them to arrive before retrying."

Agent must:
1. Send help request via `SendMessage`
2. Wait (Idle) for partner to navigate
3. Retry cooperative action when observations show partner nearby

No automatic timeout — agent reasoning decides when to give up or try different approach.

### Edge Case: Partner Disappears Mid-Autopilot

If SharedMemory `carry_autopilot` signal clears while partner is still navigating, partner immediately exits autopilot and returns to normal reasoning (`llm_agent_base.py:312-317`). Prevents deadlock if carrier completes drop before partner arrives.

## Developer Debugging Checklist

**When LLM calls fail**:
1. Check Ollama process running on expected port (11434 + agent index)
2. Verify model loaded: `ollama list`
3. Inspect retry logs: search for "LLM call failed (attempt X/5)"
4. Check `call_llm_sync()` return value in EnginePlanner (None = failure)

**When actions consistently fail**:
1. Check validation feedback in agent memory
2. Verify object IDs match world state (no typos, object not already carried/removed)
3. For cooperative actions: confirm both agents see each other in observations (within perception range)
4. In informed mode: check agent capabilities match task requirements

**When agents idle indefinitely**:
1. Check if LLM future exception caused fallback to Idle
2. Verify task assignment not None (EnginePlanner fallback may assign default task)
3. Check SharedMemory `carry_autopilot` for stuck cooperative state

No formal recovery timeout for stuck agents — simulation continues until termination conditions (all victims rescued or max iterations). Use manual plans (`manual_plans_file`) to reproduce deterministic failure scenarios.
