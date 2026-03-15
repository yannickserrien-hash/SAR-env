# Testing Strategy & Local Development

## Overview
This document covers the multi-agent rescue simulation setup including Ollama multi-port configuration, dependency installation, execution modes, debugging output, and score tracking.

## Dependencies Installation

Install Python packages via `requirements.txt`:
```bash
pip install -r requirements.txt
pip install requests pyyaml
```

Key dependencies: MATRX 2.2.0 (simulation core), ollama >= 0.4.0 (LLM SDK), toon-format (token reduction), langgraph >= 0.2.0 (graph-based workflows).

## Ollama Multi-Port Setup

Each agent requires a dedicated Ollama instance. The EnginePlanner shares port 11434 with Agent 0. Launch one terminal per agent using `OLLAMA_HOST`:

```bash
# Terminal 1 (default port 11434 — EnginePlanner + Agent 0)
ollama serve

# Terminal 2 (port 11435 — Agent 1)
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3 (port 11436 — Agent 2, only if num_rescue_agents=3)
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

**Port formula**: Agent N uses `ollama_base_port + N + 1`. Configure `ollama_base_port` in `main.py`. Default models: `qwen3:8b` (EnginePlanner), `qwen2.5:3b` (RescueAgents). Pull models via `ollama pull <model>`.

## Running the Simulation

### Single Configuration
```bash
python main.py
```

Opens visualization at http://localhost:3000 (use "God" view or human agent keyboard control). Press Ctrl+F5 to clear browser cache.

### Configuration Parameters (main.py)

- **num_rescue_agents** (1–5): Number of AI agents
- **agent_type**: `'marble'` (SearchRescueAgent), `'llm'` (RescueAgent), `'baseline'` (OfficialAgent)
- **ticks_per_iteration** (default: 1200): MATRX ticks per planning cycle (1200 ticks = 2 minutes)
- **include_human** (bool): Add keyboard-controlled agent
- **planning_mode**: `'simple'` (flat task list) or `'dag'` (graph with conditionals)
- **manual_plans_file**: Set to `"manual_plans.yaml"` to override LLM task generation, `None` for LLM mode

### Manual Plans Override

Set `manual_plans_file = "manual_plans.yaml"` in `main.py` to bypass LLM task generation. The YAML file defines per-iteration task/plan pairs. If more iterations run than entries exist, the last entry repeats. Structure:

```yaml
iterations:
  - rescuebot0:
      task: "Explore area 1"
      plan: |
        1. Move to area_1 door at [3, 4]
        2. Carry victim if nearby
        3. Navigate to Drop Zone [23, 8]
```

Plans are optional per agent — omit `plan` to let the PlanningModule use LLM. Fallback `agent_plans` section applies when iteration entries lack plans. See `manual_plans.yaml` for full syntax.

## Score Tracking & Output

### score.json
Initialized in `logs/score.json` with defaults (0 score, 0 victims_rescued, 8 total_victims). Updated by simulation. EnginePlanner reads `block_hit_rate` to check termination (1.0 = all victims rescued). See `main.py`.

### iteration_history.json
Saved to `logs/iteration_history.json` on completion. Each entry contains: iteration number, task_assignments (agent_id → task), summary (LLM-generated), score, block_hit_rate. Dataclass defined in `engine/iteration_data.py`. Generated in `main.py`.

### OutputLogger
Runs post-simulation via `output_logger(fld)` in `main.py`. Parses action logs from most recent world run, calculates unique agent/human actions, extracts final score/completeness/tick count, writes `output.csv` to world logs. See `loggers/OutputLogger.py`.

## Debugging Common Issues

### Connection Errors
**Symptom**: "Cannot connect to Ollama" in logs
**Fix**: Ensure `ollama serve` is running on the correct ports. Verify with `curl http://localhost:11434/api/tags`. Check `ollama_base_port` in `main.py` matches terminal configuration.

### Timeout Errors
**Symptom**: "Ollama request timed out" (60s default in `engine/llm_utils.py`)
**Fix**: Large models (27B+) may exceed timeout on slow hardware. Reduce model size (e.g., `qwen3:8b` instead of `gemma3:27b`) or increase `timeout=60` in `query_llm()`.

### LLM Thread Pool
Initialized via `init_llm_pool(num_rescue_agents)` in `main.py`. Workers = max(4, num_agents * 3) to handle concurrent reasoning/memory/communication calls per agent. Adjust in `engine/llm_utils.py` if agents block on LLM responses.

### Verbose Debugging
Enable Python logging before main execution:
```python
import logging
logging.basicConfig(level=logging.INFO)
```
Key loggers: `'EnginePlanner'`, `'llm_utils'`, `'RescueAgent'`. LLM query entry points print "LLM queried" by default (see `query_llm_async` in `engine/llm_utils.py`).

## Key Files Reference

- **main.py**: Entry point, configuration hub, score/history initialization
- **HOW_TO_RUN.md**: Setup instructions, dependency list, config table
- **requirements.txt**: Python dependencies
- **manual_plans.yaml**: Optional LLM override for task/plan generation
- **engine/engine_planner.py**: EnginePlanner (task coordination, termination logic)
- **engine/llm_utils.py**: Ollama query functions, timeout/connection handling, thread pool
- **engine/iteration_data.py**: IterationData dataclass (task_assignments, summary, score)
- **loggers/OutputLogger.py**: Post-run CSV output generation
- **worlds1/WorldBuilder.py**: Agent instantiation with per-agent Ollama ports
