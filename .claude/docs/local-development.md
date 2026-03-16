# Local Development Setup

## Overview

This document covers environment setup, Ollama multi-port configuration, execution modes, manual plan overrides, score tracking, and debugging for the SAR-env simulation.

## Dependencies Installation

Install Python packages via `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies: MATRX 2.2.0 (simulation framework), ollama >= 0.4.0 (SDK), litellm >= 0.1.0 (unified LLM interface), toon-format (token compression), langgraph >= 0.2.0 (graph workflows).

## Ollama Multi-Port Setup

Each agent requires a dedicated Ollama instance. **Agent N uses port `ollama_base_port + N`**. The EnginePlanner uses `ollama_base_port` (same as Agent 0).

```bash
# Terminal 1 (port 11434 — EnginePlanner + Agent 0)
ollama serve

# Terminal 2 (port 11435 — Agent 1)
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3 (port 11436 — Agent 2, only if num_rescue_agents >= 3)
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

**Port routing**: Configured via `api_base` parameter in `worlds1/WorldBuilder.py`. Agent instantiation passes `api_base=f"http://localhost:{ollama_base_port + agent_nr}"` to `SearchRescueAgent`. EnginePlanner receives `api_url=f"http://localhost:{ollama_base_port}"` in `main.py`.

Default models: `qwen3:8b` (EnginePlanner), `qwen3:8b` (SearchRescueAgents). Pull models via `ollama pull qwen3:8b`.

## Running the Simulation

```bash
python main.py
```

Opens visualization at http://localhost:3000. Use "God" view or individual agent perspectives. Press Ctrl+F5 to clear browser cache if UI doesn't update.

### Configuration Parameters (main.py)

All configuration lives at the top of `main.py`:

- **num_rescue_agents** (1-5): Number of AI agents. Each requires an Ollama instance.
- **agent_type**: `'marble'` (SearchRescueAgent, current), `'llm'` (deprecated RescueAgent), `'baseline'` (OfficialAgent)
- **ticks_per_iteration** (default: 1200): MATRX ticks per planning cycle. 1200 ticks × 0.1s/tick = 2 minutes.
- **include_human** (bool): Add keyboard-controlled agent
- **planning_mode**: `'simple'` (flat task list) or `'dag'` (TaskGraph with conditionals)
- **planner_model**: LLM model for EnginePlanner (e.g., `'qwen3:8b'`)
- **ollama_base_port** (default: 11434): Base port for Ollama instances
- **manual_plans_file**: `"manual_plans.yaml"` to override LLM task/plan generation, `None` for LLM mode

## Manual Plans Override

Set `manual_plans_file = "manual_plans.yaml"` in `main.py` to bypass LLM-generated tasks and plans. The file supports two independent sections:

**iterations**: Replaces EnginePlanner's LLM task generation. Each list entry defines per-agent tasks and optional plans for one iteration. If more iterations run than entries exist, the last entry repeats.

**agent_plans**: Fallback plan decomposition (bypasses PlanningModule LLM). EnginePlanner still uses LLM for task generation.

Structure:
```yaml
iterations:
  - rescuebot0:
      task: "Explore area 1"
      plan: |
        1. Move to area_1 door at [3, 4]
        2. Carry victim if nearby
        3. Navigate to Drop Zone [23, 8]
        4. Drop the victim
    rescuebot1:
      task: "Explore area 2"
      # plan omitted → PlanningModule uses LLM
```

Omit `plan` key to let the agent's PlanningModule use LLM for decomposition. Full syntax and door locations documented in `manual_plans.yaml`.

## LLM Integration

**Single execution path**: All LLM calls use `litellm.completion()` via `agents1/async_model_prompting.py`. No external framework dependencies (MARBLE, Ollama SDK).

**Per-agent routing**: Each agent's `api_base` parameter routes to its dedicated Ollama port. EnginePlanner uses `self._api_base`, SearchRescueAgent stores `self.api_base`.

**Two APIs**:
- **Async** (agents): `submit_llm_call()` returns `Future`, poll with `get_llm_result(future)`
- **Sync** (EnginePlanner): `call_llm_sync(llm_model, system_prompt, user_prompt, api_base=...)` returns text

Thread pool sizing: `init_marble_pool(num_rescue_agents)` in `main.py` scales workers to `max(8, num_agents * 3)`.

## Score Tracking & Output

**logs/score.json**: Initialized in `main.py` with defaults: `{"score": 0, "victims_rescued": 0, "total_victims": 8, "block_hit_rate": 0.0}`. Updated live by simulation. EnginePlanner reads `block_hit_rate` for termination (1.0 = all victims rescued).

**logs/iteration_history.json**: Saved on completion. Contains per-iteration: task_assignments, summary, score, block_hit_rate. Generated in `main.py` from `world.run_with_planner()` return value.

**OutputLogger**: Runs post-simulation via `output_logger(fld)`. Parses action logs, calculates stats, writes `output.csv`. See `loggers/OutputLogger.py`.

## Debugging Common Issues

**Connection errors**: "Cannot connect to Ollama" → Verify `ollama serve` running on expected ports. Test with `curl http://localhost:11434/api/tags`. Check `ollama_base_port` in `main.py` matches terminal configuration.

**Timeout errors**: Default 60s in `_llm_completion()`. Large models may exceed this. Reduce model size or increase `retries`/`base_wait_time` in `async_model_prompting.py`.

**LLM call failures**: Check `agents1/async_model_prompting.py` logs for retry attempts. Verify model exists via `ollama list`.

**Verbose logging**: Add before `main.py` execution:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key loggers: `'EnginePlanner'`, `'async_model_prompting'`, `'SearchRescueAgent'`.

## Key Files Reference

- **main.py**: Entry point, configuration, score/history initialization
- **requirements.txt**: Python dependencies
- **manual_plans.yaml**: Optional LLM override for tasks/plans
- **agents1/async_model_prompting.py**: Unified LLM interface (`call_llm_sync`, `submit_llm_call`)
- **engine/engine_planner.py**: Top-level coordinator, uses `call_llm_sync` with `self._api_base`
- **worlds1/WorldBuilder.py**: Agent instantiation with per-agent `api_base` URLs
- **logs/**: score.json, iteration_history.json, output.csv
