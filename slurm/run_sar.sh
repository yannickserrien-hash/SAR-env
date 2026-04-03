#!/bin/bash
#SBATCH --job-name=sar-sim
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ============================================================================
# SAR Simulation on Delft Blue
#
# Prerequisites (run ONCE on login node, not in this script):
#   module load python/3.10
#   python -m venv ~/sar-venv
#   source ~/sar-venv/bin/activate
#   pip install vllm
#   pip install -r requirements.txt
#
# Model weights should be at: /scratch/$USER/MAS/models/<model-name>
#   e.g. /scratch/$USER/MAS/models/Qwen--Qwen2.5-7B-Instruct
#
# Download models on login node (has internet):
#   huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
#       --local-dir /scratch/$USER/MAS/models/Qwen--Qwen2.5-7B-Instruct
#
# Submit:  sbatch slurm/run_sar.sh
# Monitor: tail -f slurm-<jobid>.out
# ============================================================================

set -euo pipefail

# --- Configuration (edit these) ---------------------------------------------
MODEL_PATH="/scratch/$USER/MAS/models/Qwen--Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"    # name vLLM will serve the model under
VLLM_PORT=8000                       # port for the vLLM OpenAI-compatible API
PROJECT_DIR="/scratch/$USER/MAS/SAR-env"  # path to the SAR project on scratch
# ----------------------------------------------------------------------------

# Load modules (adjust to what's available on Delft Blue)
module load 2024r1
module load python/3.10
module load cuda/12.1

# Activate virtualenv
source ~/sar-venv/bin/activate

cd "$PROJECT_DIR"

# Create a job-specific log directory on scratch
export SAR_LOG_DIR="/scratch/$USER/MAS/logs/sar-${SLURM_JOB_ID}"
mkdir -p "$SAR_LOG_DIR"

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Model:     $MODEL_PATH"
echo "vLLM port: $VLLM_PORT"
echo "Log dir:   $SAR_LOG_DIR"
echo "=========================================="

# --- Start vLLM server in the background ------------------------------------
echo "[setup] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    &> "$SAR_LOG_DIR/vllm.log" &

VLLM_PID=$!
echo "[setup] vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready (polls the health endpoint)
echo "[setup] Waiting for vLLM to load model..."
MAX_WAIT=300  # 5 minutes max
WAITED=0
until curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[ERROR] vLLM process died. Check $SAR_LOG_DIR/vllm.log"
        exit 1
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[ERROR] vLLM did not start within ${MAX_WAIT}s. Check $SAR_LOG_DIR/vllm.log"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "[setup] ... waiting ($WAITED/${MAX_WAIT}s)"
done
echo "[setup] vLLM ready after ${WAITED}s"

# --- Run SAR simulation -----------------------------------------------------
# Overrides for HPC (these match what you set in main.py):
#   enable_gui = False
#   llm_backend = 'requests'
#   api_base = 'http://localhost:8000'
#   planner_model = 'qwen2.5-7b-instruct'
#   agent_model   = 'qwen2.5-7b-instruct'

echo "[run] Starting SAR simulation..."
python main.py 2>&1 | tee "$SAR_LOG_DIR/simulation.log"
SIM_EXIT=$?

# --- Cleanup -----------------------------------------------------------------
echo "[cleanup] Stopping vLLM server..."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

echo "[done] Simulation exited with code $SIM_EXIT"
echo "[done] Logs at: $SAR_LOG_DIR"
