#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------
# run_prompted_llm_eval_vllm_sweep.sh --- Prompted LLM guardrail evaluation via vLLM-served models.
#
# For each model in the provided list, this script:
#   1. Starts a vLLM server for the model
#   2. Waits for the server to become healthy
#   3. Runs the prompted_llm_eval pipeline with the vLLM experiment config
#   4. Shuts down the vLLM server
#   5. Moves on to the next model
#
# The run name (runtime.run_name) is a per-model timestamp combined with the model name
# (e.g., "20260408_120000_Qwen_Qwen2.5-7B-Instruct").
#
# Usage:
#   bash scripts/run_prompted_llm_eval_vllm_sweep.sh \
#     --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct
#
#   # With a custom vLLM virtual environment:
#   bash scripts/run_prompted_llm_eval_vllm_sweep.sh \
#     --vllm-venv /path/to/vllm-venv \
#     --models meta-llama/Llama-3.1-8B-Instruct
#
#   # With explicit GPU and port control:
#   bash scripts/run_prompted_llm_eval_vllm_sweep.sh \
#     --cuda-devices 0,1,2,3 \
#     --port 8000 \
#     --models meta-llama/Llama-3.1-8B-Instruct
#
#   # With extra Hydra overrides:
#   bash scripts/run_prompted_llm_eval_vllm_sweep.sh \
#     --models meta-llama/Llama-3.1-8B-Instruct \
#     -- ++config.guardrail_config.max_workers=16
#
# ------------------------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ---- defaults ----
EXPERIMENT="guardrail/prompted_llm_eval_vllm"
PORT=8666
CUDA_DEVICES=""
TENSOR_PARALLEL_SIZE=""
GPU_MEM_UTIL="0.9"
MAX_MODEL_LEN="32768"
HEALTH_TIMEOUT=600  # seconds to wait for vLLM to become healthy
HEALTH_INTERVAL=5   # seconds between health checks
VLLM_VENV=""        # path to a venv with vLLM installed (default: use project's uv env)
VLLM_SERVER_EXTRA_ARGS=()
MODELS=()
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
    if [[ -n "${VLLM_PGID:-}" ]] && kill -0 -- "-${VLLM_PGID}" 2>/dev/null; then
        return 1
    fi
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        return 1
    fi
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

derive_run_name() {
    # filesystem-safe run name from model id + timestamp
    local model_name="$1"
    local safe_name="${model_name//\//_}"
    echo "$(date '+%Y%m%d_%H%M%S')_${safe_name}"
}

infer_tensor_parallel_size() {
    if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
        echo "${TENSOR_PARALLEL_SIZE}"
        return
    fi
    if [[ -n "${CUDA_DEVICES}" ]]; then
        local devices
        IFS=',' read -r -a devices <<< "${CUDA_DEVICES}"
        echo "${#devices[@]}"
        return
    fi
    if command -v nvidia-smi &>/dev/null; then
        local gpu_count
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
        if (( gpu_count > 0 )); then
            echo "${gpu_count}"
            return
        fi
    fi
    echo "1"
}

usage() {
    local exit_code="${1:-0}"
    cat <<'USAGE'
Usage: bash scripts/run_prompted_llm_eval_vllm_sweep.sh [OPTIONS] --models MODEL1 [MODEL2 ...] [-- HYDRA_OVERRIDES...]

Required:
  --models NAME ...       One or more HuggingFace model names (consumed until next flag or --)

Experiment config:
  --experiment EXP        Hydra experiment config name (default: guardrail/prompted_llm_eval_vllm)

Optional:
  --vllm-venv PATH        Path to a virtual environment with vLLM installed (default: use project's uv env)
  --port PORT             vLLM server port (default: 8000)
  --cuda-devices IDS      Comma-separated GPU IDs for vLLM (e.g., 0,1,2,3)
  --tensor-parallel-size N  Override tensor parallel size (default: auto from cuda-devices or visible GPUs)
  --gpu-mem-util FRAC     GPU memory utilization fraction (default: 0.9)
  --max-model-len LEN     Max sequence length for vLLM
  --health-timeout SECS   Max seconds to wait for vLLM readiness (default: 600)
  --vllm-extra-arg ARG    Repeatable single extra argument to pass to vllm serve
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
        --vllm-venv)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --vllm-venv requires a value"
                usage 1
            fi
            VLLM_VENV="$2"
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
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS+=("$1")
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

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: --models requires at least one model name"
    usage 1
fi

if [[ -n "${VLLM_VENV}" ]]; then
    if [[ ! -d "${VLLM_VENV}" ]]; then
        echo "Error: vLLM venv directory does not exist: ${VLLM_VENV}"
        exit 1
    fi
    if [[ ! -x "${VLLM_VENV}/bin/vllm" ]]; then
        echo "Error: vllm binary not found in venv: ${VLLM_VENV}/bin/vllm"
        exit 1
    fi
fi

# ---- vLLM lifecycle functions ----

start_vllm() {
    local model_name="$1"
    local tp_size
    tp_size="$(infer_tensor_parallel_size)"

    local server_cmd=()
    if [[ -n "${VLLM_VENV}" ]]; then
        server_cmd=("${VLLM_VENV}/bin/vllm" serve "${model_name}")
    else
        server_cmd=(uv run vllm serve "${model_name}")
    fi

    server_cmd+=(
        --host 0.0.0.0
        --port "${PORT}"
        --tensor-parallel-size "${tp_size}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --served-model-name "${model_name}"
        --trust-remote-code
        --disable-log-stats
        --enable-prefix-caching
    )

    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        server_cmd+=(--max-model-len "${MAX_MODEL_LEN}")
    fi
    if [[ ${#VLLM_SERVER_EXTRA_ARGS[@]} -gt 0 ]]; then
        server_cmd+=("${VLLM_SERVER_EXTRA_ARGS[@]}")
    fi

    log "Starting vLLM server: ${server_cmd[*]}"

    local launch_cmd=()
    if [[ -n "${CUDA_DEVICES}" ]]; then
        launch_cmd+=(env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}")
    fi
    launch_cmd+=("${server_cmd[@]}")

    if command -v setsid > /dev/null 2>&1; then
        setsid "${launch_cmd[@]}" &
        VLLM_PGID=$!
    else
        "${launch_cmd[@]}" &
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
    log "Received ${signal_name}; stopping sweep..."
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

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

log "Starting prompted LLM eval vLLM sweep: ${TOTAL} model(s), experiment=${EXPERIMENT}"
if [[ -n "${VLLM_VENV}" ]]; then
    log "Using custom vLLM venv: ${VLLM_VENV}"
fi
log "Models:"
for model in "${MODELS[@]}"; do
    log "  - ${model}"
done
echo ""

for model_idx in $(seq 0 $(( TOTAL - 1 ))); do
    model="${MODELS[${model_idx}]}"
    model_num=$(( model_idx + 1 ))
    run_name="$(derive_run_name "${model}")"

    log "=========================================="
    log "Model ${model_num}/${TOTAL}: ${model}"
    log "  Run name: ${run_name}"
    log "=========================================="

    # 1. start vLLM server for this model
    start_vllm "${model}"

    # 2. wait for it to be ready
    if ! wait_for_health "${model}"; then
        log "FAILED: vLLM server did not start for model: ${model}"
        stop_vllm
        FAILED=$(( FAILED + 1 ))
        continue
    fi

    # 3. run the prompted LLM eval pipeline
    eval_cmd=(
        uv run python -m pyine.apps.guardrail_eval.prompted_llm_eval
        "+experiment=${EXPERIMENT}"
        "runtime.run_name='${run_name}'"
        "config.guardrail_config.llm_provider.model_kwargs.model=${model}"
        "config.guardrail_config.llm_provider.model_kwargs.base_url=http://localhost:${PORT}/v1"
    )
    if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
        eval_cmd+=("${HYDRA_OVERRIDES[@]}")
    fi

    log "Running eval: ${eval_cmd[*]}"
    if "${eval_cmd[@]}"; then
        log "PASSED: model ${model_num}/${TOTAL}"
        PASSED=$(( PASSED + 1 ))
    else
        log "FAILED: eval pipeline returned non-zero for model: ${model}"
        FAILED=$(( FAILED + 1 ))
    fi

    # 4. shut down vLLM before moving to next model
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
