#!/bin/bash
# shellcheck disable=SC2016
#==================================================================================
# SLURM Debate Evaluation Launcher — BIG interrogator models (TP=7)
#==================================================================================
#
# Variant of launch_debate_eval.sh for large interrogator models that need
# multiple GPUs. Layout:
#   - GPU 0:   responder vLLM server (TP=1, single GPU)
#   - GPU 1-7: interrogator vLLM server (TP=7, tensor-parallel across 7 GPUs)
#
# Interrogator models are processed sequentially (one at a time) since each
# requires 7 GPUs.
#
# Submit with:
#   sbatch scripts/launch_debate_eval_big.sh
#
# Or override:
#   RESPONDER_MODEL="./full_checkpoints/RL_HT_49-600/" \
#   INTERROGATOR_MODELS="facebook/cwm,Qwen/Qwen3-Coder-Next" \
#   LMDB_PATHS="./RL-HT-49-600-eval/benchmark_export/" \
#       sbatch scripts/launch_debate_eval_big.sh
#
# Pin interrogator vLLM version (uses isolated venv, separate from repo):
#   INTERROGATOR_VLLM_VERSION="vllm==0.8.5" sbatch scripts/launch_debate_eval_big.sh
#
# Reuse an existing interrogator venv (skips creation if already set up):
#   INTERROGATOR_VENV_DIR="/raid/tmp/my_interr_venv" sbatch scripts/launch_debate_eval_big.sh
#
#==================================================================================
# SLURM DIRECTIVES
#==================================================================================

#SBATCH --job-name=debate-eval-big
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=/lambdafs/users/a.palmas/logs/slurm/debate_eval_big_%j.out
#SBATCH --error=/lambdafs/users/a.palmas/logs/slurm/debate_eval_big_%j.err
# #SBATCH --time=48:00:00
# #SBATCH --partition=gpu

set -euo pipefail

#==================================================================================
# CONFIGURATION
#==================================================================================

# Workspace (shared filesystem)
WORKSPACE="${WORKSPACE:-/lambdafs/users/a.palmas/new_tests/code-interp-benchmark}"

# Responder model (fixed across all evaluations)
RESPONDER_MODEL="${RESPONDER_MODEL:-${PYINE_RL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}}"

# Interrogator models (comma-separated list — processed one at a time)
INTERROGATOR_MODELS="${INTERROGATOR_MODELS:-facebook/cwm,Qwen/Qwen3-Coder-Next}"

# vLLM server settings
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-900}"  # big models need more time
HEALTH_CHECK_INTERVAL=5                               # seconds between checks

# Interrogator venv settings — isolated from repo to allow newer vLLM versions
INTERROGATOR_VLLM_VERSION="${INTERROGATOR_VLLM_VERSION:-vllm}"  # e.g. "vllm==0.8.5" or "vllm"
INTERROGATOR_VENV_DIR="${INTERROGATOR_VENV_DIR:-}"              # auto-created in tmp if empty

# Debate eval settings
MAX_WORKERS="${MAX_WORKERS:-128}"
MAX_DEBATE_TURNS="${MAX_DEBATE_TURNS:-5}"
DEBATE_TIMEOUT="${DEBATE_TIMEOUT:-180}"

# LMDB data path(s) — comma-separated, will be converted to Hydra list
LMDB_PATHS="${LMDB_PATHS:-}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-pyine-guardrails}"
USE_WANDB="${USE_WANDB:-true}"
LOGLEVEL="${LOGLEVEL:-INFO}"

# Logs on shared filesystem
LOG_DIR="${LOG_DIR:-/lambdafs/users/a.palmas/logs/debate_eval/job_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"

# Fast node-local storage for caches
RAID_BASE="${RAID_BASE:-/raid}"
CACHE_BASE="${CACHE_BASE:-${RAID_BASE}/tmp/cache}"

# Module loads (space-separated). Set to empty string to skip.
MODULE_LOADS="${MODULE_LOADS:-cuda12.8/toolkit/12.8.1}"

#==================================================================================
# GPU LAYOUT — Big Model Mode
#==================================================================================
# GPU 0:   responder    (TP=1, port 9100)
# GPU 1-4: interrogator (TP=4, port 9000)
# GPU 5-7: unused (TP must evenly divide attention heads)

RESPONDER_GPU=0
RESPONDER_PORT=9100
INTERROGATOR_GPUS="1,2,3,4"
INTERROGATOR_TP=4
INTERROGATOR_PORT=9000

#==================================================================================
# HELPERS
#==================================================================================

load_modules() {
    if [ -n "$MODULE_LOADS" ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh 2>/dev/null || true
        for mod in $MODULE_LOADS; do
            module load "$mod" 2>/dev/null || true
        done
    fi
}

setup_cache_env() {
    local cb="$1"
    export HF_HOME="${cb}/huggingface"
    export HF_HUB_CACHE="${cb}/huggingface/hub"
    export HF_DATASETS_CACHE="${cb}/huggingface/datasets"
    export TRANSFORMERS_CACHE="${cb}/huggingface/transformers"
    export TORCH_HOME="${cb}/torch"
    export TORCH_EXTENSIONS_DIR="${cb}/torch_extensions"
    export WANDB_DIR="${cb}/wandb"
    export WANDB_CACHE_DIR="${cb}/wandb/cache"
    export TIKTOKEN_CACHE_DIR="${cb}/tiktoken"
    export TRITON_CACHE_DIR="${cb}/triton"
    export XDG_CACHE_HOME="${cb}/xdg_cache"
    export TMPDIR="${cb}/tmp"
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local timeout="$3"
    local elapsed=0

    echo "  Waiting for ${name} at ${url} ..."
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            echo "  ${name} is ready (${elapsed}s)"
            return 0
        fi
        sleep "$HEALTH_CHECK_INTERVAL"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    echo "  ERROR: ${name} did not become healthy after ${timeout}s" >&2
    return 1
}

kill_vllm_servers() {
    echo "Stopping vLLM servers..."
    for pid_file in "${LOG_DIR}"/vllm_*.pid; do
        [ -f "$pid_file" ] || continue
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Wait briefly, then force-kill if still alive
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    done
    # Also kill any orphaned vllm processes we started
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    echo "All vLLM servers stopped."
}

kill_interrogator_server() {
    local pid_file="${LOG_DIR}/vllm_interrogator.pid"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

setup_interrogator_venv() {
    if [ -n "$INTERROGATOR_VENV_DIR" ]; then
        INTERROGATOR_VENV="$INTERROGATOR_VENV_DIR"
    else
        INTERROGATOR_VENV="$(mktemp -d "${TMPDIR:-/tmp}/interrogator_venv_XXXXXX")"
    fi
    echo "Creating isolated interrogator venv at: $INTERROGATOR_VENV"
    uv venv "$INTERROGATOR_VENV" --python 3.12 --quiet
    echo "Installing ${INTERROGATOR_VLLM_VERSION} in interrogator venv..."
    uv pip install "$INTERROGATOR_VLLM_VERSION" ninja --python "$INTERROGATOR_VENV/bin/python" --quiet
    echo "Installing latest transformers..."
    uv pip install -U transformers --python "$INTERROGATOR_VENV/bin/python" --quiet
    echo "Interrogator venv ready."
}

start_responder_server() {
    local model="$1"
    local log_file="${LOG_DIR}/vllm_responder.log"
    local pid_file="${LOG_DIR}/vllm_responder.pid"

    echo "  Starting responder vLLM: model=${model} gpu=${RESPONDER_GPU} port=${RESPONDER_PORT} tp=1"

    CUDA_VISIBLE_DEVICES="$RESPONDER_GPU" \
        uv run vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$RESPONDER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        --enable-prefix-caching \
        --disable-log-requests \
        ${VLLM_MAX_MODEL_LEN:+--max-model-len "$VLLM_MAX_MODEL_LEN"} \
        > "$log_file" 2>&1 &

    echo $! > "$pid_file"
}

start_interrogator_server() {
    local model="$1"
    local log_file="${LOG_DIR}/vllm_interrogator.log"
    local pid_file="${LOG_DIR}/vllm_interrogator.pid"

    local extra_args=()
    extra_args+=(--host 0.0.0.0)
    extra_args+=(--port "$INTERROGATOR_PORT")
    extra_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    extra_args+=(--tensor-parallel-size "$INTERROGATOR_TP")
    extra_args+=(--trust-remote-code)
    extra_args+=(--enable-prefix-caching)
    if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
        extra_args+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
    fi

    echo "  Starting interrogator vLLM: model=${model} gpus=${INTERROGATOR_GPUS} port=${INTERROGATOR_PORT} tp=${INTERROGATOR_TP}"

    CUDA_VISIBLE_DEVICES="$INTERROGATOR_GPUS" \
        "$INTERROGATOR_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1 &

    echo $! > "$pid_file"
}

#==================================================================================
# MAIN
#==================================================================================

echo "=============================================="
echo "LLM Debate Evaluation Launcher (BIG MODELS)"
echo "=============================================="
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Node:                $(hostname)"
echo "Workspace:           $WORKSPACE"
echo "Responder model:     $RESPONDER_MODEL"
echo "Interrogator models: $INTERROGATOR_MODELS"
echo "Layout:              GPU ${RESPONDER_GPU} = responder (TP=1)"
echo "                     GPU ${INTERROGATOR_GPUS} = interrogator (TP=${INTERROGATOR_TP})"
echo "Max workers/eval:    $MAX_WORKERS"
echo "Max debate turns:    $MAX_DEBATE_TURNS"
echo "Log dir:             $LOG_DIR"
echo "=============================================="
echo ""

# Pre-flight
load_modules
mkdir -p "$LOG_DIR" "$CACHE_BASE/tmp"
setup_cache_env "$CACHE_BASE"
cd "$WORKSPACE"

# Install dependencies for responder (repo venv)
echo "Syncing repo environment (responder)..."
uv sync --extra vllm --quiet
echo "Repo environment ready."
echo ""

# Create isolated venv for interrogator servers
setup_interrogator_venv
echo ""

# Parse interrogator models into array
IFS=',' read -ra ALL_INTERROGATOR_MODELS <<< "$INTERROGATOR_MODELS"
TOTAL_MODELS=${#ALL_INTERROGATOR_MODELS[@]}
echo "Total interrogator models to evaluate: $TOTAL_MODELS (sequential)"
echo ""

# Ensure cleanup on exit
cleanup() {
    kill_vllm_servers
    # Remove temporary interrogator venv if we created it
    if [ -z "$INTERROGATOR_VENV_DIR" ] && [ -n "${INTERROGATOR_VENV:-}" ] && [ -d "$INTERROGATOR_VENV" ]; then
        echo "Cleaning up temporary interrogator venv: $INTERROGATOR_VENV"
        rm -rf "$INTERROGATOR_VENV"
    fi
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# Start responder server (GPU 0, stays up for all models)
# --------------------------------------------------------------------------
echo "=== Starting responder vLLM server (model: $RESPONDER_MODEL) ==="
start_responder_server "$RESPONDER_MODEL"

echo "Waiting for responder server to become healthy..."
wait_for_server "http://localhost:${RESPONDER_PORT}" \
    "responder (GPU ${RESPONDER_GPU})" "$HEALTH_CHECK_TIMEOUT"
echo "Responder server ready."
echo ""

# --------------------------------------------------------------------------
# Process interrogator models one at a time (each uses 7 GPUs)
# --------------------------------------------------------------------------
TOTAL_FAILED=0

for model_idx in $(seq 0 $((TOTAL_MODELS - 1))); do
    interr_model="${ALL_INTERROGATOR_MODELS[$model_idx]}"
    safe_model_name=$(echo "$interr_model" | tr '/' '_')

    echo "=============================================="
    echo "MODEL $((model_idx + 1))/${TOTAL_MODELS}: ${interr_model}"
    echo "=============================================="

    # ------------------------------------------------------------------
    # Start interrogator server (GPUs 1-7, TP=7)
    # ------------------------------------------------------------------
    start_interrogator_server "$interr_model"

    echo "Waiting for interrogator server to become healthy..."
    wait_for_server "http://localhost:${INTERROGATOR_PORT}" \
        "interrogator (${interr_model}, GPUs ${INTERROGATOR_GPUS}, TP=${INTERROGATOR_TP})" "$HEALTH_CHECK_TIMEOUT"
    echo "Interrogator server ready."
    echo ""

    # ------------------------------------------------------------------
    # Launch debate eval
    # ------------------------------------------------------------------
    eval_output_dir="${LOG_DIR}/eval_${safe_model_name}"
    debate_output_dir="${eval_output_dir}/debate_transcripts"
    mkdir -p "$debate_output_dir"

    eval_log="${LOG_DIR}/debate_eval_${safe_model_name}.log"

    echo "  Launching debate eval: interrogator=${interr_model}"

    # Build LMDB paths override if provided
    lmdb_override=()
    if [ -n "$LMDB_PATHS" ]; then
        lmdb_override=("config.evals_config.datamodule_config.lmdb_paths=[${LMDB_PATHS}]")
    fi

    PYINE_INTERROGATOR_MODEL_NAME="$interr_model" \
    PYINE_RL_MODEL_NAME="$RESPONDER_MODEL" \
    VLLM_INTERROGATOR_BASE_URL="http://localhost:${INTERROGATOR_PORT}/v1" \
    VLLM_BASE_URL="http://localhost:${RESPONDER_PORT}/v1" \
        uv run python -m pyine.apps.guardrail_eval.debate_eval \
            +experiment=guardrail/debate_eval_vllm \
            runtime.exp_name="debate_eval_${safe_model_name}" \
            runtime.run_name="debate_eval_${safe_model_name}" \
            config.guardrail_config.max_workers="$MAX_WORKERS" \
            config.guardrail_config.max_debate_turns="$MAX_DEBATE_TURNS" \
            config.guardrail_config.debate_timeout_seconds="$DEBATE_TIMEOUT" \
            config.guardrail_config.debate_output_dir="$debate_output_dir" \
            config.guardrail_config.interrogator_provider.model_kwargs.model="$interr_model" \
            config.guardrail_config.interrogator_provider.model_kwargs.base_url="http://localhost:${INTERROGATOR_PORT}/v1" \
            config.guardrail_config.responder_provider.model_kwargs.model="$RESPONDER_MODEL" \
            config.guardrail_config.responder_provider.model_kwargs.base_url="http://localhost:${RESPONDER_PORT}/v1" \
            config.use_wandb_logging="$USE_WANDB" \
            config.wandb_project="$WANDB_PROJECT" \
            config.evals_config.result_dump_dir="$eval_output_dir" \
            hydra.run.dir="$eval_output_dir" \
            "${lmdb_override[@]}" \
            > "$eval_log" 2>&1

    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "  DONE: ${interr_model} — success"
    else
        echo "  FAIL: ${interr_model} — exit code ${exit_code}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    echo ""

    # ------------------------------------------------------------------
    # Stop interrogator server before next model
    # ------------------------------------------------------------------
    echo "Stopping interrogator server..."
    kill_interrogator_server
    # Give GPU memory time to free up before next model
    sleep 5
    echo "Interrogator server stopped."
    echo ""
done

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "=============================================="
echo "All debate evaluations complete."
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Results:   $LOG_DIR"
if [ "$TOTAL_FAILED" -gt 0 ]; then
    echo "FAILURES:  ${TOTAL_FAILED}/${TOTAL_MODELS}"
fi
echo "=============================================="
echo ""
echo "Per-model outputs:"
for interr_model in "${ALL_INTERROGATOR_MODELS[@]}"; do
    safe_name=$(echo "$interr_model" | tr '/' '_')
    echo "  ${interr_model}:"
    echo "    eval log:     ${LOG_DIR}/debate_eval_${safe_name}.log"
    echo "    results:      ${LOG_DIR}/eval_${safe_name}/"
    echo "    transcripts:  ${LOG_DIR}/eval_${safe_name}/debate_transcripts/"
done
