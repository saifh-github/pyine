#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------
# run_classifier_bias_sweep.sh --- Trains LLM classifiers across resampling bias presets.
#
# Runs the classifier trainer sequentially for each skewed_*_bias resampling preset
# (weak, moderate, strong), overriding the datamodule resampling, calibration resampling,
# and run name for each preset.
#
# Usage:
#   # single GPU (default)
#   bash scripts/run_classifier_bias_sweep.sh
#   bash scripts/run_classifier_bias_sweep.sh --experiment llm_classifier/v0_qwen2
#   bash scripts/run_classifier_bias_sweep.sh --presets weak strong
#
#   # multi-GPU via torchrun (uses run_ddp.sh)
#   bash scripts/run_classifier_bias_sweep.sh --nproc_per_node 4
#
#   # extra Hydra overrides
#   bash scripts/run_classifier_bias_sweep.sh -- config.num_replicas=5
#
# ------------------------------------------------------------------------------------------

set -euo pipefail

# ---- defaults ----
EXPERIMENT="llm_classifier/v0_modernbert"
PRESETS=(weak moderate strong)
NPROC_PER_NODE=""
HYDRA_OVERRIDES=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DDP="${SCRIPT_DIR}/run_ddp.sh"
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ---- helpers ----

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_classifier_bias_sweep.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

Options:
  --experiment EXP        Hydra experiment config (default: llm_classifier/v0_modernbert)
  --presets P1 P2 ...     Bias presets to sweep (default: weak moderate strong)
  --nproc_per_node N      Number of GPUs per run; launches via run_ddp.sh when set
  -h, --help              Show this help

Everything after -- is forwarded as extra Hydra overrides to each run.
USAGE
    exit "${1:-0}"
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
        --nproc_per_node)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --nproc_per_node requires a value"
                usage 1
            fi
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --presets)
            PRESETS=()
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                PRESETS+=("$1")
                shift
            done
            if [[ ${#PRESETS[@]} -eq 0 ]]; then
                echo "Error: --presets requires at least one preset name"
                usage 1
            fi
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

# ---- main loop ----

TOTAL=${#PRESETS[@]}
PASSED=0
FAILED=0

log "Starting classifier bias sweep: ${TOTAL} preset(s), experiment=${EXPERIMENT}"
log "Presets: ${PRESETS[*]}"
if [[ -n "${NPROC_PER_NODE}" ]]; then
    log "Multi-GPU: ${NPROC_PER_NODE} processes per node (via run_ddp.sh)"
fi
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
    log "Extra overrides: ${HYDRA_OVERRIDES[*]}"
fi
echo ""

for preset_idx in $(seq 0 $(( TOTAL - 1 ))); do
    preset="${PRESETS[${preset_idx}]}"
    preset_num=$(( preset_idx + 1 ))
    bias_name="skewed_${preset}_bias"
    run_name="\${now:%Y%m%d_%H%M%S}_${bias_name}"

    log "=========================================="
    log "  Preset ${preset_num}/${TOTAL}: ${bias_name}"
    log "=========================================="

    hydra_args=(
        "+experiment=${EXPERIMENT}"
        "config/datamodule_config/resampling=${bias_name}"
        "config/evals_config/calibration_resampling=${bias_name}"
        "runtime.run_name=${run_name}"
    )
    if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
        hydra_args+=("${HYDRA_OVERRIDES[@]}")
    fi

    if [[ -n "${NPROC_PER_NODE}" ]]; then
        # run_ddp.sh needs torchrun and python on PATH; use uv run so the
        # project venv's bin dir is prepended automatically
        run_cmd=(
            uv run bash "${RUN_DDP}"
            --app pyine.apps.trainers.llm_classifier_trainer
            --nproc_per_node "${NPROC_PER_NODE}"
            -- "${hydra_args[@]}"
        )
    else
        run_cmd=(
            uv run python -m pyine.apps.trainers.llm_classifier_trainer
            "${hydra_args[@]}"
        )
    fi

    log "Running: ${run_cmd[*]}"
    if "${run_cmd[@]}"; then
        log "PASSED: preset ${preset_num}/${TOTAL} (${bias_name})"
        PASSED=$(( PASSED + 1 ))
    else
        log "FAILED: preset ${preset_num}/${TOTAL} (${bias_name})"
        FAILED=$(( FAILED + 1 ))
    fi
    echo ""
done

# ---- summary ----

log "=========================================="
log "Sweep complete: ${PASSED} passed, ${FAILED} failed out of ${TOTAL}"
log "=========================================="

if (( FAILED > 0 )); then
    exit 1
fi
