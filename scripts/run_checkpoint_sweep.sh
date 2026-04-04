#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------
# run_checkpoint_sweep.sh --- Evaluates multiple HF checkpoints via vLLM-served inference.
#
# For each checkpoint in the provided list, this script:
#   1. Starts a vLLM server pointing at the checkpoint
#   2. Waits for the server to become healthy
#   3. Runs the hf_trainer predict pipeline (code exec eval) with a unique experiment name
#   4. Shuts down the vLLM server
#   5. Moves on to the next checkpoint
#
# Results are logged to separate Hydra output directories per checkpoint.
#
# Usage:
#   bash scripts/run_checkpoint_sweep.sh \
#     --experiment original/v0_50perc_dataset_qwen3_vllm_eval \
#     --checkpoints /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3
#
#   # With explicit GPU and port control:
#   bash scripts/run_checkpoint_sweep.sh \
#     --experiment original/v0_50perc_dataset_qwen3_vllm_eval \
#     --cuda-devices 0,1,2,3 \
#     --port 8000 \
#     --checkpoints /path/to/ckpt1 /path/to/ckpt2
#
#   # Pass extra Hydra overrides after --:
#   bash scripts/run_checkpoint_sweep.sh \
#     --experiment original/v0_50perc_dataset_qwen3_vllm_eval \
#     --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
#     -- config.evals_config.eval_batch_size=48
#
# ------------------------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ---- defaults ----
EXPERIMENT=""
PORT=8000
CUDA_DEVICES=""
TENSOR_PARALLEL_SIZE=""
GPU_MEM_UTIL="0.9"
MAX_MODEL_LEN=""
HEALTH_TIMEOUT=600  # seconds to wait for vLLM to become healthy
HEALTH_INTERVAL=5   # seconds between health checks
VLLM_SERVER_EXTRA_ARGS=()
CHECKPOINTS=()
HYDRA_OVERRIDES=()
VLLM_PID=""
VLLM_PGID=""
CLEANUP_DONE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ensure uv and hydra resolve the correct project regardless of the caller's cwd
cd "${REPO_ROOT}"

# load .env so child processes (vLLM, eval) inherit variables like HF_TOKEN
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ---- helpers ----

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

server_serves_expected_model() {
    local expected_model="$1"
    local url="http://localhost:${PORT}/v1/models"
    local models_payload
    if ! models_payload="$(curl -sf "${url}")"; then
        return 1
    fi
    EXPECTED_MODEL="${expected_model}" python3 -c '
import json
import os
import sys

payload = json.loads(sys.stdin.read())
expected_model = os.environ["EXPECTED_MODEL"]
models = payload.get("data", [])
sys.exit(0 if any(model.get("id") == expected_model for model in models) else 1)
' <<< "${models_payload}"
}

port_is_open() {
    python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

vllm_resources_released() {
    # a tracked process group is still alive
    if [[ -n "${VLLM_PGID:-}" ]] && kill -0 -- "-${VLLM_PGID}" 2>/dev/null; then
        return 1
    fi
    # a tracked process is still alive
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        return 1
    fi
    # the port is still in use
    if port_is_open "${PORT}"; then
        return 1
    fi
    return 0
}

wait_for_vllm_stop() {
    local timeout_secs="$1"
    local waited=0
    while (( waited < timeout_secs )); do
        if vllm_resources_released; then
            return 0
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    vllm_resources_released
}

usage() {
    local exit_code="${1:-0}"
    cat <<'USAGE'
Usage: bash scripts/run_checkpoint_sweep.sh [OPTIONS] --checkpoints CKPT1 [CKPT2 ...] [-- HYDRA_OVERRIDES...]

Required:
  --experiment EXP        Hydra experiment config name (e.g., original/v0_50perc_dataset_qwen3_vllm_eval)
  --checkpoints PATH ...  One or more checkpoint directories (consumed until next flag or --)

Optional:
  --port PORT             vLLM server port (default: 8000)
  --cuda-devices IDS      Comma-separated GPU IDs for vLLM (e.g., 0,1,2,3)
  --tensor-parallel-size N  Override tensor parallel size (default: auto from cuda-devices)
  --gpu-mem-util FRAC     GPU memory utilization fraction (default: 0.9)
  --max-model-len LEN     Max sequence length for vLLM
  --health-timeout SECS   Max seconds to wait for vLLM readiness (default: 600)
  --vllm-extra-arg ARG    Repeatable single extra argument to pass to vllm_server.py
  -h, --help              Show this help

Everything after -- is forwarded as Hydra overrides to the eval run.
USAGE
    exit "${exit_code}"
}

# ---- parse arguments ----

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --experiment)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --experiment requires a value"
                usage 1
            fi
            EXPERIMENT="$2"
            shift 2
            ;;
        --port)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --port requires a value"
                usage 1
            fi
            PORT="$2"
            shift 2
            ;;
        --cuda-devices)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --cuda-devices requires a value"
                usage 1
            fi
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --tensor-parallel-size requires a value"
                usage 1
            fi
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-mem-util)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --gpu-mem-util requires a value"
                usage 1
            fi
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --max-model-len)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --max-model-len requires a value"
                usage 1
            fi
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --health-timeout)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --health-timeout requires a value"
                usage 1
            fi
            HEALTH_TIMEOUT="$2"
            shift 2
            ;;
        --vllm-extra-arg)
            if [[ $# -lt 2 ]]; then
                echo "Error: --vllm-extra-arg requires a value"
                usage 1
            fi
            VLLM_SERVER_EXTRA_ARGS+=("$2")
            shift 2
            ;;
        --checkpoints)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CHECKPOINTS+=("$1")
                shift
            done
            ;;
        --)
            shift
            HYDRA_OVERRIDES=("$@")
            break
            ;;
        *)
            echo "Error: unknown argument: $1"
            usage 1
            ;;
    esac
done

# ---- validate ----

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Error: --experiment is required"
    usage 1
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Error: --checkpoints requires at least one checkpoint path"
    usage 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    if [[ ! -d "${ckpt}" ]]; then
        echo "Error: checkpoint directory does not exist: ${ckpt}"
        exit 1
    fi
done

# ---- functions ----

derive_model_name() {
    # mirrors the auto-derive logic in vllm_server.py: last 3 path components joined by /
    local ckpt_path="$1"
    local parts
    IFS='/' read -ra parts <<< "${ckpt_path}"
    local num_parts=${#parts[@]}
    local num_components=$(( num_parts < 3 ? num_parts : 3 ))
    local start_idx=$(( num_parts - num_components ))
    local name=""
    for (( idx=start_idx; idx < num_parts; idx++ )); do
        if [[ -n "${name}" ]]; then
            name="${name}/"
        fi
        name="${name}${parts[idx]}"
    done
    echo "${name}"
}

derive_experiment_suffix() {
    # create a short, filesystem-safe suffix from the checkpoint path
    local ckpt_path="$1"
    local base
    base="$(basename "${ckpt_path}")"
    local parent
    parent="$(basename "$(dirname "${ckpt_path}")")"
    echo "${parent}_${base}"
}

start_vllm() {
    local ckpt_path="$1"
    local server_cmd=(
        uv run python "${SCRIPT_DIR}/vllm_eval/vllm_server.py"
        --checkpoint_path "${ckpt_path}"
        --port "${PORT}"
        --gpu_memory_utilization "${GPU_MEM_UTIL}"
    )
    if [[ -n "${CUDA_DEVICES}" ]]; then
        server_cmd+=(--cuda_devices "${CUDA_DEVICES}")
    fi
    if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
        server_cmd+=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
    fi
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        server_cmd+=(--max_model_len "${MAX_MODEL_LEN}")
    fi
    if [[ ${#VLLM_SERVER_EXTRA_ARGS[@]} -gt 0 ]]; then
        server_cmd+=("${VLLM_SERVER_EXTRA_ARGS[@]}")
    fi

    log "Starting vLLM server: ${server_cmd[*]}"
    if command -v setsid > /dev/null 2>&1; then
        setsid "${server_cmd[@]}" &
        VLLM_PGID=$!
    else
        "${server_cmd[@]}" &
        VLLM_PGID=""
    fi
    VLLM_PID=$!
    log "vLLM server started (PID: ${VLLM_PID})"
}

wait_for_health() {
    local expected_model="$1"
    local url="http://localhost:${PORT}/health"
    local elapsed=0
    local warned_wrong_model=0
    log "Waiting for vLLM server to become healthy at ${url} (timeout: ${HEALTH_TIMEOUT}s, expected model: ${expected_model})..."
    while (( elapsed < HEALTH_TIMEOUT )); do
        if curl -sf "${url}" > /dev/null 2>&1; then
            if server_serves_expected_model "${expected_model}"; then
                log "vLLM server is healthy and serving ${expected_model} (took ${elapsed}s)"
                return 0
            fi
            if (( warned_wrong_model == 0 )); then
                log "vLLM health endpoint is up, but a different model is still being served on port ${PORT}; waiting for the new server..."
                warned_wrong_model=1
            fi
        fi
        # also check that the server process is still alive
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "Error: vLLM server process (PID ${VLLM_PID}) died before becoming healthy"
            return 1
        fi
        local remaining=$(( HEALTH_TIMEOUT - elapsed ))
        local sleep_secs="${HEALTH_INTERVAL}"
        if (( sleep_secs > remaining )); then
            sleep_secs="${remaining}"
        fi
        if (( sleep_secs > 0 )); then
            sleep "${sleep_secs}"
            elapsed=$(( elapsed + sleep_secs ))
        fi
    done
    log "Error: vLLM server did not become healthy within ${HEALTH_TIMEOUT}s"
    return 1
}

stop_vllm() {
    if [[ -z "${VLLM_PID:-}" && -z "${VLLM_PGID:-}" ]]; then
        return
    fi
    local stopped_cleanly=0

    if [[ -n "${VLLM_PGID:-}" ]] && kill -0 -- "-${VLLM_PGID}" 2>/dev/null; then
        log "Stopping vLLM server process group (PGID: ${VLLM_PGID})..."
        kill -TERM -- "-${VLLM_PGID}" 2>/dev/null || true
    elif [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping vLLM server process (PID: ${VLLM_PID})..."
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
    fi

    if wait_for_vllm_stop 45; then
        stopped_cleanly=1
    fi

    if (( stopped_cleanly == 0 )); then
        if [[ -n "${VLLM_PGID:-}" ]] && kill -0 -- "-${VLLM_PGID}" 2>/dev/null; then
            log "Force-killing vLLM server process group (PGID: ${VLLM_PGID})..."
            kill -KILL -- "-${VLLM_PGID}" 2>/dev/null || true
        elif [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "Force-killing vLLM server process (PID: ${VLLM_PID})..."
            kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi

        if wait_for_vllm_stop 15; then
            stopped_cleanly=1
        fi
    fi

    if [[ -n "${VLLM_PID:-}" ]]; then
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    if (( stopped_cleanly == 1 )); then
        log "vLLM server stopped"
    else
        log "Warning: timed out waiting for vLLM server resources to fully release"
    fi

    VLLM_PID=""
    VLLM_PGID=""
}

# ensure vLLM is stopped on exit/interrupt
cleanup() {
    if (( CLEANUP_DONE == 1 )); then
        return
    fi
    CLEANUP_DONE=1
    log "Cleaning up..."
    stop_vllm
}

handle_signal() {
    local signal_name="$1"
    log "Received ${signal_name}; stopping checkpoint sweep..."
    cleanup
    trap - EXIT
    if [[ "${signal_name}" == "INT" ]]; then
        exit 130
    fi
    exit 143
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

# ---- main loop ----

TOTAL=${#CHECKPOINTS[@]}
PASSED=0
FAILED=0

log "Starting checkpoint sweep: ${TOTAL} checkpoint(s), experiment=${EXPERIMENT}"
log "Checkpoints:"
for ckpt in "${CHECKPOINTS[@]}"; do
    log "  - ${ckpt}"
done
echo ""

for ckpt_idx in $(seq 0 $(( TOTAL - 1 ))); do
    ckpt="${CHECKPOINTS[${ckpt_idx}]}"
    ckpt_num=$(( ckpt_idx + 1 ))
    model_name="$(derive_model_name "${ckpt}")"
    exp_suffix="$(derive_experiment_suffix "${ckpt}")"

    log "=========================================="
    log "Checkpoint ${ckpt_num}/${TOTAL}: ${ckpt}"
    log "  Model name: ${model_name}"
    log "  Experiment suffix: ${exp_suffix}"
    log "=========================================="

    # 1. start vLLM server for this checkpoint
    start_vllm "${ckpt}"

    # 2. wait for it to be ready
    if ! wait_for_health "${model_name}"; then
        log "FAILED: vLLM server did not start for checkpoint: ${ckpt}"
        stop_vllm
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # 3. run the eval pipeline
    run_name="${EXPERIMENT}_${exp_suffix}"
    eval_cmd=(
        uv run python -m pyine.apps.trainers.hf_trainer
        "+experiment=${EXPERIMENT}"
        "runtime.exp_name=${run_name}"
        "runtime.run_name=${run_name}"
        "config.evals_config.vllm_provider_config.model_kwargs.base_url=http://localhost:${PORT}/v1"
        "config.evals_config.vllm_provider_config.model_kwargs.model=${model_name}"
    )
    # append any extra Hydra overrides from the user
    if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
        eval_cmd+=("${HYDRA_OVERRIDES[@]}")
    fi

    log "Running eval: ${eval_cmd[*]}"
    if "${eval_cmd[@]}"; then
        log "PASSED: checkpoint ${ckpt_num}/${TOTAL}"
        PASSED=$(( PASSED + 1 ))
    else
        log "FAILED: eval pipeline returned non-zero for checkpoint: ${ckpt}"
        FAILED=$(( FAILED + 1 ))
    fi

    # 4. shut down vLLM before moving to next checkpoint
    stop_vllm
    echo ""
done

# ---- summary ----

log "=========================================="
log "Sweep complete: ${PASSED} passed, ${FAILED} failed out of ${TOTAL}"
log "=========================================="

if (( FAILED > 0 )); then
    exit 1
fi
