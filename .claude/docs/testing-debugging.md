# Testing & Debugging Strategy

## Overview

This document covers testing infrastructure, debugging techniques, common failure modes, and troubleshooting strategies for the SAR-env multi-agent simulation. Focus areas: thread safety verification, LLM integration debugging, cooperative action validation, and Ollama multi-port configuration issues.

## Testing Infrastructure

**No formal unit tests exist** in the codebase. The project uses runtime logging, score validation, and manual observation via the Flask visualizer at http://localhost:3000 for quality assurance. One legacy test file exists at `matrx-master/test_api.py` (MATRX API smoke test, not used in production).

### Validation Mechanisms

**Action Validation** (`agents1/llm_agent_base.py`): Two-layer validation before MATRX dispatch. `_validate_action()` checks if `object_id` exists in 1-block Chebyshev radius using current `WORLD_STATE.nearby[]`. `_check_matrx_action()` queries MATRX's `is_action_possible()` for feasibility. On failure, agents idle and populate `_action_feedback` for next LLM prompt.

**Score Tracking** (`logs/score.json`): Initialized in `main.py` with `{score: 0, victims_rescued: 0, total_victims: 8, block_hit_rate: 0.0}`. Updated by WorldBuilder's CollectionGoal when victims enter drop zone. EnginePlanner reads `block_hit_rate` each iteration to check termination (1.0 = all rescued). Post-run analysis via `loggers/OutputLogger.py` parses action logs to calculate unique actions, final score, and tick count.

**LLM Retry Logic** (`agents1/async_model_prompting.py`): `_retry_with_backoff()` decorator wraps `_llm_completion()` with 5 retries and exponential backoff (1s base). On final failure, logs error and returns None. Propagates exceptions up to agent decision loop.

## Debugging Techniques

### Multi-Port Ollama Troubleshooting

**Port Configuration**: Each agent uses `ollama_base_port + agent_nr` (default: 11434 for EnginePlanner + Agent 0, 11435 for Agent 1, etc.). Set in `main.py` and passed to `worlds1/WorldBuilder.py`. Agents receive `api_base=f"http://localhost:{port}"` in constructor.

**Connection Verification**: Run `curl http://localhost:11434/api/tags` per port to verify Ollama instances. If connection fails, check `OLLAMA_HOST` environment variable in terminal (e.g., `OLLAMA_HOST=0.0.0.0:11435 ollama serve`). Mismatch between `main.py` config and terminal port causes "Cannot connect to Ollama" errors.

**Timeout Issues**: Default 60s timeout in LiteLLM. Large models (27B+) may exceed on slow hardware. Reduce model size (`qwen2.5:3b` instead of `gemma3:27b`) or increase `max_token_num` in `async_model_prompting.py`.

### Thread Safety & Race Conditions

**SharedMemory Lock** (`memory/shared_memory.py`): Thread-safe via `threading.Lock`. All reads/writes (`update()`, `retrieve()`, `retrieve_all()`) use context manager `with self.lock:`. Used for cooperative carry rendezvous: agent writes `carry_rendezvous` dict with `{agent, victim_id, location, status}`, partner reads and navigates.

**LLM ThreadPoolExecutor** (`agents1/async_model_prompting.py`): Global `_executor` with `max_workers = max(8, num_agents * 3)`. Initialized via `init_marble_pool(num_rescue_agents)` in `main.py`. Each agent submits async LLM calls via `submit_llm_call()`, polls via `get_llm_result(future)`. No shared state between threads — each Future is agent-owned.

**Stale Observation Issue**: Agents cache `WORLD_STATE` at tick start via `_tick_setup()` → `percept_state()`. If object moves/disappears mid-tick, `_validate_action()` catches ID mismatch and returns Idle + feedback. Not a race condition — just eventual consistency by design.

### Cooperative Action Debugging

**Carry Retry Loop** (`agents1/llm_agent_base.py`): When `CarryObjectTogether` dispatched, agent enters `_handle_carry_retry()` loop. Writes rendezvous to SharedMemory, increments `_carry_wait_ticks` each tick (max 100). Timeout → Idle + `carry_failure` in memory. Success → resets counter, clears rendezvous. Partner navigates via `_handle_rendezvous()` using A* to published location.

**Rendezvous Validation**: Check SharedMemory via `self.shared_memory.retrieve('carry_rendezvous')`. Should contain `{agent, victim_id, location, status: 'waiting_for_partner'}`. If None or wrong agent, rendezvous inactive. Debug prints: `"[agent_id] Carry retry N/100"`, `"[agent_id] Navigating to carry rendezvous at [x,y]"`.

### LLM Integration Debugging

**Verbose Logging**: Enable Python logging before `main.py` execution:
```python
import logging
logging.basicConfig(level=logging.INFO)
```
Key loggers: `'async_model_prompting'` (retry warnings), `'LLMAgentBase'` (action validation), `'EnginePlanner'` (task generation). SearchRescueAgent prints `"[agent_id] Submitting LLM call"` on each reasoning step.

**Tool Call Validation**: LLM returns `tool_calls` list via `litellm.completion()`. Extracted in `agents1/search_rescue_agent.py` → `_process_llm_response()`. If missing, falls back to `ActionMapper` (regex parser for plain-text responses). Invalid tool names → Idle + feedback `"Unknown action: {name}"`.

**Prompt Inspection**: Reasoning prompts built via `agents1/modules/reasoning_module.py` → `ReasoningIO.get_reasoning_prompt()`. Includes task decomposition, observation (TOON-compressed nearby objects), feedback from previous action, and last 15 memory entries. Print `prompt` before `_submit_llm()` to debug context issues.

## Common Failure Modes

**Agent Idle Loops**: Symptom: Agent repeatedly idles. Causes: (1) No task assigned (check PlannerChannel), (2) Navigation stuck (check A* path exists), (3) Object validation fails (object out of range or removed). Solution: Enable validation logging, check `_action_feedback` in next prompt.

**Carry Timeout**: Symptom: "Carry timeout after 100 ticks". Cause: Partner never navigates to rendezvous (wrong task, navigation blocked, or already carrying). Solution: Check both agents' tasks, verify SharedMemory rendezvous entry, add debug prints in `_handle_rendezvous()`.

**LLM Call Hangs**: Symptom: Simulation freezes. Cause: Sync LLM call in main thread (EnginePlanner uses separate executor). Solution: Never use `call_llm_sync()` in agent decision loop — only in EnginePlanner/memory modules. Verify `submit_llm_call()` returns Future immediately.

**Score Not Updating**: Symptom: `block_hit_rate` stays 0. Cause: Victims not entering drop zone (wrong location, never carried, or dropped outside). Solution: Check drop zone coordinates (23, 8-15 in `worlds1/WorldBuilder.py`), verify `CarryObject` → `NavigateToDropZone` → `DropObject` sequence.

## Integration Testing Strategies

**Manual Plans Override** (`main.py`): Set `manual_plans_file = "manual_plans.yaml"` to bypass LLM task generation. Define per-iteration tasks to isolate action execution bugs from LLM reasoning failures. See `manual_plans.yaml` for YAML structure.

**Single-Agent Mode**: Set `num_rescue_agents = 1` and `include_human = False` to test individual agent logic without cooperative action complexity. Reduces LLM port management to single Ollama instance.

**Score Validation**: After simulation, check `logs/iteration_history.json` for per-iteration `{score, block_hit_rate, task_assignments, summary}`. Cross-reference with `logs/world_1/output.csv` (unique actions, completeness, ticks). Score should increase monotonically.

**Visualizer Observation**: Use "God" view (http://localhost:3000) to manually verify: (1) Agents move toward victims, (2) Carry animations trigger, (3) Victims appear in drop zone, (4) No infinite navigation loops.

## Key Files Reference

- **agents1/async_model_prompting.py**: LLM retry, timeout, thread pool
- **agents1/llm_agent_base.py**: Action validation, carry retry, rendezvous
- **memory/shared_memory.py**: Thread-safe cross-agent state
- **loggers/ActionLogger.py**: Tick-by-tick action/score CSV
- **loggers/OutputLogger.py**: Post-run summary generation
- **main.py**: Score.json initialization, iteration history export
- **.claude/docs/local-development.md**: Ollama port setup guide
