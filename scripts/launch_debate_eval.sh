#!/bin/bash
# shellcheck disable=SC2016
#==================================================================================
# SLURM Debate Evaluation Launcher
#==================================================================================
#
# Runs guardrail debate evaluations on a single 8-GPU node, varying the
# interrogator model against a fixed responder. One script, three modes
# (selected with MODE=...):
#
#   MODE=vllm       (default) Small/medium vLLM interrogators.
#                   Layout: 4 (interrogator TP=1, responder TP=1) pairs on 8 GPUs.
#                   Models processed in batches of 4 in parallel.
#
#   MODE=vllm-big   Large vLLM interrogators that need tensor parallelism.
#                   Layout: GPU 0 = responder (TP=1), GPUs 1..N = interrogator
#                   (TP=N, default N=4). Models processed sequentially.
#
#   MODE=openai     OpenAI API interrogator (no interrogator GPUs).
#                   Layout: all 8 GPUs run responder vLLM servers in parallel,
#                   one per (model, reasoning_effort) config. Requires
#                   OPENAI_API_KEY in the environment.
#
# ---------------------------------------------------------------------------------
# QUICK START
# ---------------------------------------------------------------------------------
#
#   # Default (MODE=vllm), default models:
#   sbatch scripts/launch_debate_eval.sh
#
#   # Custom responder + interrogator list + dataset:
#   RESPONDER_MODEL="./full_checkpoints/RL_HT_49-600/" \
#   INTERROGATOR_MODELS="Qwen/Qwen3-4B,Qwen/Qwen3-8B" \
#   LMDB_PATHS="./RL-HT-49-600-eval/benchmark_export/" \
#       sbatch scripts/launch_debate_eval.sh
#
#   # Large interrogator with TP=4:
#   MODE=vllm-big \
#   INTERROGATOR_MODELS="facebook/cwm,Qwen/Qwen3-Coder-Next" \
#   INTERROGATOR_TP=4 \
#       sbatch scripts/launch_debate_eval.sh
#
#   # OpenAI interrogator, varying reasoning_effort:
#   MODE=openai \
#   OPENAI_API_KEY=sk-... \
#   INTERROGATOR_CONFIGS="gpt-5-mini:minimal,gpt-5-mini:low,gpt-5-mini:default" \
#   RATE_LIMIT_RPS=30 \
#       sbatch scripts/launch_debate_eval.sh
#
# ---------------------------------------------------------------------------------
# COMMON ENV VARS (apply to all modes)
# ---------------------------------------------------------------------------------
#
#   WORKSPACE                  Repo root on shared FS
#   RESPONDER_MODEL            HF id or local path of the responder model
#   LMDB_PATHS                 Comma-separated; Hydra override for
#                              datamodule_config.lmdb_paths. Empty = config default.
#   GPU_MEMORY_UTILIZATION     vLLM gpu memory util (default 0.9)
#   VLLM_MAX_MODEL_LEN         vLLM max-model-len (default 20000)
#   HEALTH_CHECK_TIMEOUT       Seconds to wait for vLLM servers
#                              (default 600 / 900 / 1800 by mode)
#   MAX_WORKERS                Debate eval max_workers (default 128)
#   MAX_DEBATE_TURNS           Debate max turns (default 5)
#   DEBATE_TIMEOUT             Per-debate timeout in seconds (default 180)
#   WANDB_PROJECT, USE_WANDB   wandb logging
#   LOG_DIR                    Override log directory
#   CACHE_BASE                 Node-local cache base (default /raid/tmp/cache)
#   MODULE_LOADS               Space-separated module list (default cuda12.8)
#
# ---------------------------------------------------------------------------------
# MODE=vllm / MODE=vllm-big specific
# ---------------------------------------------------------------------------------
#
#   INTERROGATOR_MODELS        Comma-separated HF ids / local paths
#   INTERROGATOR_TP            (vllm-big only) tensor-parallel size, default 4.
#                              Interrogator runs on GPUs 1..INTERROGATOR_TP.
#   INTERROGATOR_VLLM_VERSION  Pin a vLLM version for the isolated interr venv,
#                              e.g. "vllm==0.8.5"
#   INTERROGATOR_VENV_DIR      Reuse an existing interr venv at this path
#                              (skip creation). Empty = create temp venv.
#
# Interrogator vLLM runs in a venv separate from the repo so it can use a
# different vLLM version than the responder/repo.
#
# ---------------------------------------------------------------------------------
# MODE=openai specific
# ---------------------------------------------------------------------------------
#
#   OPENAI_API_KEY             Required.
#   INTERROGATOR_CONFIGS       Comma-separated "model:reasoning_effort" list.
#                              reasoning_effort="default" omits the parameter
#                              (uses OpenAI's default). Default config covers
#                              gpt-5.4-{mini,nano} x {minimal,low,default}.
#   MAX_PARALLEL               Max parallel evals = max responder GPUs (1..8,
#                              default 8)
#   RATE_LIMIT_RPS             Per-eval OpenAI rate limit (default 30)
#   RATE_LIMIT_BUCKET          Token bucket size (default 30)
#
#==================================================================================
# SLURM DIRECTIVES
#==================================================================================

#SBATCH --job-name=debate-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=/lambdafs/users/user/logs/slurm/debate_eval_%j.out
#SBATCH --error=/lambdafs/users/user/logs/slurm/debate_eval_%j.err
# #SBATCH --time=48:00:00
# #SBATCH --partition=gpu

set -euo pipefail

#==================================================================================
# MODE DISPATCH
#==================================================================================

MODE="${MODE:-vllm}"
case "$MODE" in
    vllm|vllm-big|openai) ;;
    *) echo "ERROR: invalid MODE='$MODE' (expected vllm, vllm-big, openai)" >&2; exit 1 ;;
esac

#==================================================================================
# COMMON CONFIGURATION
#==================================================================================

WORKSPACE="${WORKSPACE:-/lambdafs/users/user/new_tests/code-interp-benchmark}"
RESPONDER_MODEL="${RESPONDER_MODEL:-${PYINE_RL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
HEALTH_CHECK_INTERVAL=5
case "$MODE" in
    vllm)     _DEFAULT_HEALTH_TIMEOUT=600 ;;
    vllm-big) _DEFAULT_HEALTH_TIMEOUT=900 ;;
    openai)   _DEFAULT_HEALTH_TIMEOUT=1800 ;;
esac
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-$_DEFAULT_HEALTH_TIMEOUT}"

MAX_WORKERS="${MAX_WORKERS:-128}"
MAX_DEBATE_TURNS="${MAX_DEBATE_TURNS:-5}"
DEBATE_TIMEOUT="${DEBATE_TIMEOUT:-180}"

LMDB_PATHS="${LMDB_PATHS:-}"

WANDB_PROJECT="${WANDB_PROJECT:-pyine-guardrails}"
USE_WANDB="${USE_WANDB:-true}"
LOGLEVEL="${LOGLEVEL:-INFO}"

if [ "$MODE" = "openai" ]; then
    _LOG_DIR_BASE="/lambdafs/users/user/logs/debate_eval_openai"
else
    _LOG_DIR_BASE="/lambdafs/users/user/logs/debate_eval"
fi
LOG_DIR="${LOG_DIR:-${_LOG_DIR_BASE}/job_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"

RAID_BASE="${RAID_BASE:-/raid}"
CACHE_BASE="${CACHE_BASE:-${RAID_BASE}/tmp/cache}"
MODULE_LOADS="${MODULE_LOADS:-cuda12.8/toolkit/12.8.1}"

# Interrogator venv (vllm modes only)
INTERROGATOR_VLLM_VERSION="${INTERROGATOR_VLLM_VERSION:-vllm}"
INTERROGATOR_VENV_DIR="${INTERROGATOR_VENV_DIR:-}"
INTERROGATOR_VENV=""

#==================================================================================
# MODE-SPECIFIC CONFIGURATION
#==================================================================================

case "$MODE" in
    vllm)
        INTERROGATOR_MODELS="${INTERROGATOR_MODELS:-Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-4B-Instruct,Qwen/Qwen3-8B-Instruct}"
        NUM_PAIRS=4
        INTERROGATOR_GPUS_LIST=("0" "2" "4" "6")
        RESPONDER_GPUS_LIST=("1" "3" "5" "7")
        INTERROGATOR_PORTS=(9000 9001 9002 9003)
        RESPONDER_PORTS=(9100 9101 9102 9103)
        INTERROGATOR_TP=1
        RESPONDER_TP=1
        HYDRA_EXPERIMENT="guardrail/debate_eval_vllm"
        ;;

    vllm-big)
        INTERROGATOR_MODELS="${INTERROGATOR_MODELS:-facebook/cwm,Qwen/Qwen3-Coder-Next}"
        INTERROGATOR_TP="${INTERROGATOR_TP:-4}"
        RESPONDER_TP=1
        NUM_PAIRS=1
        # Responder on GPU 0; interrogator spans GPUs 1..INTERROGATOR_TP
        RESPONDER_GPUS_LIST=("0")
        _interr_gpu_csv=""
        for ((i=1; i<=INTERROGATOR_TP; i++)); do
            [ -n "$_interr_gpu_csv" ] && _interr_gpu_csv+=","
            _interr_gpu_csv+="$i"
        done
        INTERROGATOR_GPUS_LIST=("$_interr_gpu_csv")
        INTERROGATOR_PORTS=(9000)
        RESPONDER_PORTS=(9100)
        HYDRA_EXPERIMENT="guardrail/debate_eval_vllm"
        ;;

    openai)
        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo "ERROR: MODE=openai requires OPENAI_API_KEY in the environment." >&2
            exit 1
        fi
        export OPENAI_API_KEY
        INTERROGATOR_CONFIGS="${INTERROGATOR_CONFIGS:-gpt-5.4-mini:minimal,gpt-5.4-mini:low,gpt-5.4-mini:default,gpt-5.4-nano:minimal,gpt-5.4-nano:low,gpt-5.4-nano:default}"
        MAX_PARALLEL="${MAX_PARALLEL:-8}"
        RESPONDER_BASE_PORT=9100
        RATE_LIMIT_RPS="${RATE_LIMIT_RPS:-30}"
        RATE_LIMIT_BUCKET="${RATE_LIMIT_BUCKET:-30}"
        HYDRA_EXPERIMENT="guardrail/debate_eval_openai"
        ;;
esac

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

kill_pidfile() {
    local pid_file="$1"
    [ -f "$pid_file" ] || return 0
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

kill_all_vllm_servers() {
    echo "Stopping vLLM servers..."
    for pid_file in "${LOG_DIR}"/vllm_*.pid; do
        [ -f "$pid_file" ] || continue
        kill_pidfile "$pid_file"
    done
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    echo "All vLLM servers stopped."
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
    uv pip install "$INTERROGATOR_VLLM_VERSION" --python "$INTERROGATOR_VENV/bin/python" --quiet
    echo "Installing latest transformers..."
    uv pip install -U transformers --python "$INTERROGATOR_VENV/bin/python" --quiet
    echo "Interrogator venv ready."
}

# Start a vLLM server in the background.
# Args: kind ("interr"|"resp"), model, gpu_csv, tp, port, role.
#   kind=interr -> isolated interrogator venv + gdn_prefill_backend triton
#   kind=resp   -> repo venv (uv run) + --disable-log-requests
start_vllm_server() {
    local kind="$1"
    local model="$2"
    local gpu_csv="$3"
    local tp="$4"
    local port="$5"
    local role="$6"
    local log_file="${LOG_DIR}/vllm_${role}.log"
    local pid_file="${LOG_DIR}/vllm_${role}.pid"

    local extra_args=()
    extra_args+=(--host 0.0.0.0)
    extra_args+=(--port "$port")
    extra_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    extra_args+=(--tensor-parallel-size "$tp")
    extra_args+=(--trust-remote-code)
    extra_args+=(--enable-prefix-caching)
    if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
        extra_args+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
    fi

    echo "  Starting ${role} vLLM: model=${model} gpus=${gpu_csv} tp=${tp} port=${port}"

    if [ "$kind" = "interr" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_csv" \
            "$INTERROGATOR_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --additional-config '{"gdn_prefill_backend": "triton"}' \
            "${extra_args[@]}" \
            > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu_csv" \
            uv run vllm serve "$model" \
            --disable-log-requests \
            "${extra_args[@]}" \
            > "$log_file" 2>&1 &
    fi

    echo $! > "$pid_file"
}

sanitize_name() {
    echo "$1" | tr '/' '_' | tr ' ' '_'
}

# Parse "model:reasoning" string, sets globals PARSED_MODEL and PARSED_REASONING.
parse_openai_config() {
    local config_str="$1"
    PARSED_MODEL="${config_str%%:*}"
    PARSED_REASONING="${config_str#*:}"
}

#==================================================================================
# BANNER + PRE-FLIGHT
#==================================================================================

echo "=============================================="
echo "LLM Debate Evaluation Launcher (mode: $MODE)"
echo "=============================================="
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Node:                $(hostname)"
echo "Workspace:           $WORKSPACE"
echo "Responder model:     $RESPONDER_MODEL"
case "$MODE" in
    vllm|vllm-big)
        echo "Interrogator models: $INTERROGATOR_MODELS"
        echo "Pairs:               $NUM_PAIRS (interr TP=${INTERROGATOR_TP}, resp TP=${RESPONDER_TP})"
        ;;
    openai)
        echo "Interrogator configs:"
        IFS=',' read -ra _CFG_PRINT <<< "$INTERROGATOR_CONFIGS"
        for cfg in "${_CFG_PRINT[@]}"; do
            parse_openai_config "$cfg"
            echo "  - model=${PARSED_MODEL}  reasoning_effort=${PARSED_REASONING}"
        done
        echo "Max parallel:        $MAX_PARALLEL"
        echo "Rate limit:          ${RATE_LIMIT_RPS} req/s per eval"
        ;;
esac
echo "Max workers/eval:    $MAX_WORKERS"
echo "Max debate turns:    $MAX_DEBATE_TURNS"
echo "Log dir:             $LOG_DIR"
echo "=============================================="
echo ""

load_modules
mkdir -p "$LOG_DIR" "$CACHE_BASE/tmp"
setup_cache_env "$CACHE_BASE"
cd "$WORKSPACE"

echo "Syncing repo environment..."
uv sync --extra vllm --quiet
echo "Repo environment ready."
echo ""

# Interrogator venv only needed for vllm modes
if [ "$MODE" != "openai" ]; then
    setup_interrogator_venv
    echo ""
fi

cleanup() {
    kill_all_vllm_servers
    if [ -z "$INTERROGATOR_VENV_DIR" ] && [ -n "$INTERROGATOR_VENV" ] && [ -d "$INTERROGATOR_VENV" ]; then
        echo "Cleaning up temporary interrogator venv: $INTERROGATOR_VENV"
        rm -rf "$INTERROGATOR_VENV"
    fi
}
trap cleanup EXIT

# Common Hydra override for LMDB paths
LMDB_OVERRIDE=()
if [ -n "$LMDB_PATHS" ]; then
    LMDB_OVERRIDE=("config.evals_config.datamodule_config.lmdb_paths=[${LMDB_PATHS}]")
fi

#==================================================================================
# MAIN LOOP — vllm / vllm-big: persistent responders, batched interrogators
#==================================================================================

if [ "$MODE" = "vllm" ] || [ "$MODE" = "vllm-big" ]; then
    IFS=',' read -ra ALL_INTERROGATOR_MODELS <<< "$INTERROGATOR_MODELS"
    TOTAL_MODELS=${#ALL_INTERROGATOR_MODELS[@]}
    echo "Total interrogator models: $TOTAL_MODELS"
    echo ""

    # Start NUM_PAIRS responder servers (persistent across batches)
    echo "=== Starting responder vLLM servers (model: $RESPONDER_MODEL) ==="
    for i in $(seq 0 $((NUM_PAIRS - 1))); do
        start_vllm_server "resp" "$RESPONDER_MODEL" \
            "${RESPONDER_GPUS_LIST[$i]}" "$RESPONDER_TP" \
            "${RESPONDER_PORTS[$i]}" "responder_${i}"
    done

    echo "Waiting for responder servers..."
    for i in $(seq 0 $((NUM_PAIRS - 1))); do
        wait_for_server "http://localhost:${RESPONDER_PORTS[$i]}" \
            "responder_${i} (GPUs ${RESPONDER_GPUS_LIST[$i]})" "$HEALTH_CHECK_TIMEOUT"
    done
    echo "All responder servers ready."
    echo ""

    batch_idx=0
    model_idx=0
    while [ "$model_idx" -lt "$TOTAL_MODELS" ]; do
        batch_size=$NUM_PAIRS
        remaining=$((TOTAL_MODELS - model_idx))
        [ "$remaining" -lt "$batch_size" ] && batch_size=$remaining

        batch_idx=$((batch_idx + 1))
        echo "=============================================="
        echo "BATCH ${batch_idx}: models $((model_idx + 1))-$((model_idx + batch_size)) of $TOTAL_MODELS"
        echo "=============================================="

        # Start interrogator servers for this batch
        echo "Starting interrogator vLLM servers..."
        for i in $(seq 0 $((batch_size - 1))); do
            interr_model="${ALL_INTERROGATOR_MODELS[$((model_idx + i))]}"
            start_vllm_server "interr" "$interr_model" \
                "${INTERROGATOR_GPUS_LIST[$i]}" "$INTERROGATOR_TP" \
                "${INTERROGATOR_PORTS[$i]}" "interrogator_${i}"
        done

        echo "Waiting for interrogator servers..."
        for i in $(seq 0 $((batch_size - 1))); do
            interr_model="${ALL_INTERROGATOR_MODELS[$((model_idx + i))]}"
            wait_for_server "http://localhost:${INTERROGATOR_PORTS[$i]}" \
                "interrogator_${i} (${interr_model})" "$HEALTH_CHECK_TIMEOUT"
        done
        echo "Batch ${batch_idx} interrogators ready."
        echo ""

        # Launch debate eval processes (one per pair, in parallel)
        EVAL_PIDS=()
        for i in $(seq 0 $((batch_size - 1))); do
            interr_model="${ALL_INTERROGATOR_MODELS[$((model_idx + i))]}"
            interr_port="${INTERROGATOR_PORTS[$i]}"
            resp_port="${RESPONDER_PORTS[$i]}"

            safe_model_name=$(sanitize_name "$interr_model")
            eval_output_dir="${LOG_DIR}/eval_${safe_model_name}"
            debate_output_dir="${eval_output_dir}/debate_transcripts"
            mkdir -p "$debate_output_dir"
            eval_log="${LOG_DIR}/debate_eval_${safe_model_name}.log"

            echo "  Launching: ${interr_model} (pair ${i})"

            PYINE_INTERROGATOR_MODEL_NAME="$interr_model" \
            PYINE_RL_MODEL_NAME="$RESPONDER_MODEL" \
            VLLM_INTERROGATOR_BASE_URL="http://localhost:${interr_port}/v1" \
            VLLM_BASE_URL="http://localhost:${resp_port}/v1" \
                uv run python -m pyine.apps.guardrail_eval.debate_eval \
                    +experiment="$HYDRA_EXPERIMENT" \
                    runtime.exp_name="debate_eval_${safe_model_name}" \
                    runtime.run_name="debate_eval_${safe_model_name}" \
                    config.guardrail_config.max_workers="$MAX_WORKERS" \
                    config.guardrail_config.max_debate_turns="$MAX_DEBATE_TURNS" \
                    config.guardrail_config.debate_timeout_seconds="$DEBATE_TIMEOUT" \
                    config.guardrail_config.debate_output_dir="$debate_output_dir" \
                    config.guardrail_config.interrogator_provider.model_kwargs.model="$interr_model" \
                    config.guardrail_config.interrogator_provider.model_kwargs.base_url="http://localhost:${interr_port}/v1" \
                    config.guardrail_config.responder_provider.model_kwargs.model="$RESPONDER_MODEL" \
                    config.guardrail_config.responder_provider.model_kwargs.base_url="http://localhost:${resp_port}/v1" \
                    config.use_wandb_logging="$USE_WANDB" \
                    config.wandb_project="$WANDB_PROJECT" \
                    config.evals_config.result_dump_dir="$eval_output_dir" \
                    hydra.run.dir="$eval_output_dir" \
                    "${LMDB_OVERRIDE[@]}" \
                    > "$eval_log" 2>&1 &

            EVAL_PIDS+=($!)
            echo "    PID=$! log=$eval_log"
        done

        echo ""
        echo "Waiting for ${batch_size} debate evals..."
        BATCH_FAILED=0
        for j in $(seq 0 $((batch_size - 1))); do
            pid="${EVAL_PIDS[$j]}"
            interr_model="${ALL_INTERROGATOR_MODELS[$((model_idx + j))]}"
            if wait "$pid"; then
                echo "  DONE: ${interr_model} (PID ${pid})"
            else
                exit_code=$?
                echo "  FAIL: ${interr_model} (PID ${pid}) — exit ${exit_code}"
                BATCH_FAILED=$((BATCH_FAILED + 1))
            fi
        done
        if [ "$BATCH_FAILED" -gt 0 ]; then
            echo "WARNING: ${BATCH_FAILED}/${batch_size} evaluations failed in batch ${batch_idx}"
        fi
        echo ""

        # Stop interrogator servers (responders stay up)
        echo "Stopping interrogator servers for batch ${batch_idx}..."
        for i in $(seq 0 $((batch_size - 1))); do
            kill_pidfile "${LOG_DIR}/vllm_interrogator_${i}.pid"
        done
        sleep 5
        echo ""

        model_idx=$((model_idx + batch_size))
    done

#==================================================================================
# MAIN LOOP — openai: per-batch responders, OpenAI API interrogators
#==================================================================================

elif [ "$MODE" = "openai" ]; then
    IFS=',' read -ra ALL_CONFIGS <<< "$INTERROGATOR_CONFIGS"
    TOTAL_CONFIGS=${#ALL_CONFIGS[@]}
    echo "Total interrogator configs: $TOTAL_CONFIGS"
    echo ""

    config_idx=0
    batch_idx=0
    while [ "$config_idx" -lt "$TOTAL_CONFIGS" ]; do
        batch_size=$MAX_PARALLEL
        remaining=$((TOTAL_CONFIGS - config_idx))
        [ "$remaining" -lt "$batch_size" ] && batch_size=$remaining

        batch_idx=$((batch_idx + 1))
        echo "=============================================="
        echo "BATCH ${batch_idx}: configs $((config_idx + 1))-$((config_idx + batch_size)) of $TOTAL_CONFIGS"
        echo "=============================================="

        echo "Starting ${batch_size} responder vLLM servers..."
        for i in $(seq 0 $((batch_size - 1))); do
            resp_port=$((RESPONDER_BASE_PORT + i))
            start_vllm_server "resp" "$RESPONDER_MODEL" "$i" 1 "$resp_port" "responder_${i}"
        done

        echo "Waiting for responder servers..."
        for i in $(seq 0 $((batch_size - 1))); do
            resp_port=$((RESPONDER_BASE_PORT + i))
            wait_for_server "http://localhost:${resp_port}" \
                "responder_${i} (GPU ${i})" "$HEALTH_CHECK_TIMEOUT"
        done
        echo "All responder servers ready."
        echo ""

        EVAL_PIDS=()
        for i in $(seq 0 $((batch_size - 1))); do
            cfg="${ALL_CONFIGS[$((config_idx + i))]}"
            parse_openai_config "$cfg"
            interr_model="$PARSED_MODEL"
            reasoning_effort="$PARSED_REASONING"
            resp_port=$((RESPONDER_BASE_PORT + i))

            safe_name=$(sanitize_name "${interr_model}_reasoning_${reasoning_effort}")
            eval_output_dir="${LOG_DIR}/eval_${safe_name}"
            debate_output_dir="${eval_output_dir}/debate_transcripts"
            mkdir -p "$debate_output_dir"
            eval_log="${LOG_DIR}/debate_eval_${safe_name}.log"

            echo "  Launching: model=${interr_model} reasoning=${reasoning_effort} -> GPU ${i}"

            reasoning_override=()
            if [ "$reasoning_effort" != "default" ]; then
                reasoning_override=("+config.guardrail_config.interrogator_provider.model_kwargs.reasoning_effort=${reasoning_effort}")
            fi

            PYINE_RL_MODEL_NAME="$RESPONDER_MODEL" \
            VLLM_BASE_URL="http://localhost:${resp_port}/v1" \
                uv run python -m pyine.apps.guardrail_eval.debate_eval \
                    +experiment="$HYDRA_EXPERIMENT" \
                    runtime.exp_name="debate_eval_${safe_name}" \
                    runtime.run_name="debate_eval_${safe_name}" \
                    config.guardrail_config.interrogator_provider.model_kwargs.model="$interr_model" \
                    config.guardrail_config.interrogator_provider.rate_limiter_config.requests_per_second="$RATE_LIMIT_RPS" \
                    config.guardrail_config.interrogator_provider.rate_limiter_config.max_bucket_size="$RATE_LIMIT_BUCKET" \
                    config.guardrail_config.max_workers="$MAX_WORKERS" \
                    config.guardrail_config.max_debate_turns="$MAX_DEBATE_TURNS" \
                    config.guardrail_config.debate_timeout_seconds="$DEBATE_TIMEOUT" \
                    config.guardrail_config.debate_output_dir="$debate_output_dir" \
                    config.guardrail_config.responder_provider.model_kwargs.model="$RESPONDER_MODEL" \
                    config.guardrail_config.responder_provider.model_kwargs.base_url="http://localhost:${resp_port}/v1" \
                    config.use_wandb_logging="$USE_WANDB" \
                    config.wandb_project="$WANDB_PROJECT" \
                    config.evals_config.result_dump_dir="$eval_output_dir" \
                    hydra.run.dir="$eval_output_dir" \
                    "${reasoning_override[@]}" \
                    "${LMDB_OVERRIDE[@]}" \
                    > "$eval_log" 2>&1 &

            EVAL_PIDS+=($!)
            echo "    PID=$! log=$eval_log"
        done

        echo ""
        echo "Waiting for ${batch_size} debate evals..."
        BATCH_FAILED=0
        for j in $(seq 0 $((batch_size - 1))); do
            pid="${EVAL_PIDS[$j]}"
            cfg="${ALL_CONFIGS[$((config_idx + j))]}"
            if wait "$pid"; then
                echo "  DONE: ${cfg} (PID ${pid})"
            else
                exit_code=$?
                echo "  FAIL: ${cfg} (PID ${pid}) — exit ${exit_code}"
                BATCH_FAILED=$((BATCH_FAILED + 1))
            fi
        done
        if [ "$BATCH_FAILED" -gt 0 ]; then
            echo "WARNING: ${BATCH_FAILED}/${batch_size} evaluations failed in batch ${batch_idx}"
        fi
        echo ""

        echo "Stopping responder servers for batch ${batch_idx}..."
        for i in $(seq 0 $((batch_size - 1))); do
            kill_pidfile "${LOG_DIR}/vllm_responder_${i}.pid"
        done
        sleep 5
        echo ""

        config_idx=$((config_idx + batch_size))
    done
fi

#==================================================================================
# SUMMARY
#==================================================================================

echo "=============================================="
echo "All debate evaluations complete."
echo "Mode:      $MODE"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Results:   $LOG_DIR"
echo "=============================================="
echo ""
echo "Per-eval outputs:"
case "$MODE" in
    vllm|vllm-big)
        for interr_model in "${ALL_INTERROGATOR_MODELS[@]}"; do
            safe_name=$(sanitize_name "$interr_model")
            echo "  ${interr_model}:"
            echo "    eval log:     ${LOG_DIR}/debate_eval_${safe_name}.log"
            echo "    results:      ${LOG_DIR}/eval_${safe_name}/"
            echo "    transcripts:  ${LOG_DIR}/eval_${safe_name}/debate_transcripts/"
        done
        ;;
    openai)
        for cfg in "${ALL_CONFIGS[@]}"; do
            parse_openai_config "$cfg"
            safe_name=$(sanitize_name "${PARSED_MODEL}_reasoning_${PARSED_REASONING}")
            echo "  ${PARSED_MODEL} (reasoning=${PARSED_REASONING}):"
            echo "    eval log:     ${LOG_DIR}/debate_eval_${safe_name}.log"
            echo "    results:      ${LOG_DIR}/eval_${safe_name}/"
            echo "    transcripts:  ${LOG_DIR}/eval_${safe_name}/debate_transcripts/"
        done
        ;;
esac
