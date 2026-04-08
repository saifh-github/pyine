#!/bin/bash
# shellcheck disable=SC2016
#==================================================================================
# SLURM Multi-Node DeepSpeed Zero3 Training Launcher
#==================================================================================
#
# Submit with:  sbatch scripts/launch_slurm.sh
# Or override:  TRAIN_ARGS="+experiment=keywords/v0_rl.yaml" sbatch scripts/launch_slurm.sh
#
# This script replaces the manual SSH-based launch_multinode.sh for SLURM-managed
# clusters. SLURM handles node allocation, process placement, and cleanup.
#
#==================================================================================
# SLURM DIRECTIVES — edit these or override via sbatch flags
#==================================================================================

#SBATCH --job-name=rl-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=/lambdafs/users/a.palmas/logs/slurm/job_%j.out
#SBATCH --error=/lambdafs/users/a.palmas/logs/slurm/job_%j.err
# Uncomment and set if your cluster requires it:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -euo pipefail

#==================================================================================
# CONFIGURATION — edit this section for your run
#==================================================================================

# Workspace: where the repo lives. Use shared filesystem for simplicity with SLURM.
# Code reads from /lambdafs are fine; heavy I/O (caches, checkpoints) goes to /raid.
WORKSPACE="${WORKSPACE:-/lambdafs/users/a.palmas/new_tests/code-interp-benchmark}"

# Accelerate config (relative to WORKSPACE)
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml}"

# Training script (relative to WORKSPACE)
TRAIN_SCRIPT="${TRAIN_SCRIPT:-pyine/apps/trainers/hf_trainer.py}"

# Training arguments (Hydra overrides)
TRAIN_ARGS="${TRAIN_ARGS:-+experiment=keywords/v0_rl.yaml}"

# GPUs per node (8 for standard multi-GPU nodes)
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Fast node-local storage for caches
RAID_BASE="${RAID_BASE:-/raid}"
CACHE_BASE="${CACHE_BASE:-${RAID_BASE}/tmp/cache}"

# Application paths — these control where the training app reads data and writes outputs.
# PYINE_DATA_ROOT: where datasets (LMDB traces) live. Defaults to {WORKSPACE}/data.
#   Since WORKSPACE is on /lambdafs (shared), datasets are accessible from all nodes.
# PYINE_LOGS_ROOT: where Hydra writes run outputs (configs, checkpoints, trainer state).
#   On shared FS so all nodes can write and outputs survive job completion.
PYINE_DATA_ROOT="${PYINE_DATA_ROOT:-${WORKSPACE}/data}"
PYINE_LOGS_ROOT="${PYINE_LOGS_ROOT:-${WORKSPACE}/logs}"

WANDB_PROJECT="${WANDB_PROJECT:-pyine}"
LOGLEVEL="${LOGLEVEL:-INFO}"

# SLURM log directory for launcher-level logs (stdout/stderr tee, env dump)
LOG_DIR="${LOG_DIR:-/lambdafs/users/a.palmas/logs/slurm/run_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)}"

# Network interface for NCCL communication (check with: ip link show)
# Common values: eth0, bond0, ens5, ibp*s0 (InfiniBand)
NETWORK_INTERFACE="${NETWORK_INTERFACE:-bond0}"

# Module loads (space-separated). Set to empty string to skip.
MODULE_LOADS="${MODULE_LOADS:-cuda12.8/toolkit/12.8.1 nccl2-cuda12.8-gcc/2.25.1}"

# Option to force recreate cache
RECREATE_CACHE="${RECREATE_CACHE:-false}"

# Resume configuration (optional)
RESUME_FROM_RUN_DIR="${RESUME_FROM_RUN_DIR:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Checkpoint output directory override (optional; defaults to Hydra output_dir)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# Debug mode for NCCL/PyTorch distributed (10-30%+ overhead)
ENABLE_DEBUG="${ENABLE_DEBUG:-false}"

#==================================================================================
# DERIVED CONFIGURATION — do not edit
#==================================================================================

TOTAL_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

# Get the list of allocated nodes
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MAIN_NODE="${NODELIST[0]}"

# Resolve main node IP from network interface
MAIN_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$MAIN_NODE" \
    bash -c "ip -4 addr show dev ${NETWORK_INTERFACE} 2>/dev/null | grep -oP 'inet \K[0-9.]+'" 2>/dev/null)

if [ -z "$MAIN_NODE_IP" ]; then
    # Fallback: resolve hostname
    MAIN_NODE_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$MAIN_NODE" hostname -I 2>/dev/null | awk '{print $1}')
fi

if [ -z "$MAIN_NODE_IP" ]; then
    echo "ERROR: Cannot determine IP for main node $MAIN_NODE" >&2
    echo "Set NETWORK_INTERFACE or add MAIN_PROCESS_IP to override." >&2
    exit 1
fi

# Allow explicit IP override
MAIN_NODE_IP="${MAIN_PROCESS_IP:-$MAIN_NODE_IP}"

#==================================================================================
# HELPER: cache environment variables
#==================================================================================

build_cache_exports() {
    local cb="$1"
    cat <<CACHEEOF
export HF_HOME="${cb}/huggingface"
export HF_HUB_CACHE="${cb}/huggingface/hub"
export HF_DATASETS_CACHE="${cb}/huggingface/datasets"
export HF_ASSETS_CACHE="${cb}/huggingface/assets"
export HF_MODULES_CACHE="${cb}/huggingface/modules"
export TRANSFORMERS_CACHE="${cb}/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${cb}/huggingface/hub"
export TORCH_HOME="${cb}/torch"
export TORCH_EXTENSIONS_DIR="${cb}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${cb}/torch_inductor"
export PYTORCH_KERNEL_CACHE_PATH="${cb}/torch/kernels"
export WANDB_DIR="${cb}/wandb"
export WANDB_CACHE_DIR="${cb}/wandb/cache"
export WANDB_CONFIG_DIR="${cb}/wandb/config"
export WANDB_DATA_DIR="${cb}/wandb/data"
export TIKTOKEN_CACHE_DIR="${cb}/tiktoken"
export TRITON_CACHE_DIR="${cb}/triton"
export MPLCONFIGDIR="${cb}/matplotlib"
export XDG_CACHE_HOME="${cb}/xdg_cache"
export XDG_CONFIG_HOME="${cb}/xdg_config"
export XDG_DATA_HOME="${cb}/xdg_data"
export CUDA_CACHE_PATH="${cb}/cuda_cache"
export FLASH_ATTENTION_CACHE_DIR="${cb}/flash_attn"
export VLLM_CACHE_DIR="${cb}/vllm"
export TMPDIR="${cb}/tmp"
export TEMP="${cb}/tmp"
export TMP="${cb}/tmp"
CACHEEOF
}

#==================================================================================
# JOB INFO
#==================================================================================

echo "=============================================="
echo "SLURM Multi-Node Training"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Nodes:           ${NODELIST[*]}"
echo "Num nodes:       $SLURM_NNODES"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Total processes: $TOTAL_PROCESSES"
echo "Main node:       $MAIN_NODE ($MAIN_NODE_IP)"
echo "Workspace:       $WORKSPACE"
echo "Accelerate cfg:  $ACCELERATE_CONFIG"
echo "Train args:      $TRAIN_ARGS"
echo "Cache base:      $CACHE_BASE (on /raid, node-local)"
echo "Log dir:         $LOG_DIR (on /lambdafs, shared)"
echo "Network iface:   $NETWORK_INTERFACE"
echo "Debug:           $ENABLE_DEBUG"
if [ -n "$RESUME_FROM_RUN_DIR" ]; then
    echo "Resume from:     $RESUME_FROM_RUN_DIR"
    [ -n "$RESUME_CHECKPOINT" ] && echo "Resume ckpt:     $RESUME_CHECKPOINT"
fi
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint dir:  $CHECKPOINT_DIR"
else
    echo "Checkpoint dir:  (Hydra output_dir on shared FS)"
fi
echo "Data root:       $PYINE_DATA_ROOT"
echo "Logs root:       $PYINE_LOGS_ROOT (Hydra outputs, checkpoints)"
echo "=============================================="
echo ""

#==================================================================================
# PRE-FLIGHT: create dirs, validate workspace
#==================================================================================

# Create shared log directory
mkdir -p "$LOG_DIR"

# Save this script's config for reproducibility
env | grep -E '^(SLURM_|TRAIN_|WORKSPACE|CACHE_|CHECKPOINT_|RESUME_|NETWORK_|ENABLE_|MODULE_|ACCELERATE_|LOG_DIR)' \
    | sort > "${LOG_DIR}/job_env.txt" 2>/dev/null || true

# Validate workspace exists (shared filesystem — same on all nodes)
if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: Workspace not found: $WORKSPACE" >&2
    echo "Ensure the repo is cloned on the shared filesystem (/lambdafs)." >&2
    exit 1
fi

# Node-local setup: create cache dirs, optionally recreate cache
echo "Setting up node-local directories..."
srun --ntasks-per-node=1 bash -c "
    if [ '${RECREATE_CACHE}' = true ]; then
        rm -rf '${CACHE_BASE}'
    fi
    mkdir -p '${CACHE_BASE}/tmp'
    if [ -n '${CHECKPOINT_DIR}' ]; then
        mkdir -p '${CHECKPOINT_DIR}'
    fi
    echo \"  Node \$(hostname): /raid ready\"
"

#==================================================================================
# BUILD TRAINING COMMAND
#==================================================================================

# Build Hydra overrides for checkpoint dir and resume
HYDRA_OVERRIDES=""
if [ -n "$CHECKPOINT_DIR" ]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} config.grpo_config.output_dir=${CHECKPOINT_DIR}"
fi
if [ -n "$RESUME_FROM_RUN_DIR" ]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} config.resume_from_run_dir=${RESUME_FROM_RUN_DIR}"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        HYDRA_OVERRIDES="${HYDRA_OVERRIDES} config.resume_checkpoint_name=${RESUME_CHECKPOINT}"
    fi
fi

# Build debug env exports
DEBUG_EXPORTS=""
if [ "$ENABLE_DEBUG" = true ]; then
    DEBUG_EXPORTS="
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
"
fi

# Build module load commands
MODULE_CMDS=""
if [ -n "$MODULE_LOADS" ]; then
    MODULE_CMDS="source /etc/profile.d/modules.sh"
    for mod in $MODULE_LOADS; do
        MODULE_CMDS="${MODULE_CMDS} && module load ${mod}"
    done
fi

# Write the cache exports to a temp file so srun can source it
CACHE_EXPORTS_FILE="${LOG_DIR}/.cache_exports.sh"
build_cache_exports "$CACHE_BASE" > "$CACHE_EXPORTS_FILE"

#==================================================================================
# LAUNCH TRAINING
#==================================================================================

echo ""
echo "Launching training across ${SLURM_NNODES} nodes..."
echo "  DeepSpeed Zero Stage 3"
echo "  vLLM colocated on training GPUs"
echo ""

# Export variables needed inside srun
export WORKSPACE ACCELERATE_CONFIG TRAIN_SCRIPT TRAIN_ARGS GPUS_PER_NODE
export TOTAL_PROCESSES MAIN_NODE_IP NETWORK_INTERFACE
export CACHE_EXPORTS_FILE HYDRA_OVERRIDES DEBUG_EXPORTS MODULE_CMDS LOG_DIR
export PYINE_PER_NODE_PREP=off
export PYINE_DATA_ROOT PYINE_LOGS_ROOT WANDB_PROJECT LOGLEVEL

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    # Load modules if configured
    if [ -n "$MODULE_CMDS" ]; then
        eval "$MODULE_CMDS"
    fi

    # Set cache environment (node-local /raid paths)
    source "$CACHE_EXPORTS_FILE"

    # Application paths (dataset reads from shared FS, outputs to shared FS)
    export PYINE_DATA_ROOT="$PYINE_DATA_ROOT"
    export PYINE_LOGS_ROOT="$PYINE_LOGS_ROOT"

    # NCCL network config
    export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"

    # Debug env vars (empty string if debug disabled)
    eval "$DEBUG_EXPORTS"

    # Accelerate env vars (used by accelerate + DeepSpeed for coordination)
    export ACCELERATE_MACHINE_RANK=$SLURM_PROCID
    export ACCELERATE_MAIN_PROCESS_IP=$MAIN_NODE_IP
    export ACCELERATE_NUM_MACHINES=$SLURM_NNODES
    export ACCELERATE_NUM_PROCESSES=$TOTAL_PROCESSES

    cd "$WORKSPACE"

    echo "[$(hostname)] Rank $SLURM_PROCID: launching accelerate (main=$MAIN_NODE_IP)"

    uv run --extra vllm --extra flash_attn --extra liger --extra gpu_monitoring accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --machine_rank "$SLURM_PROCID" \
        --main_process_ip "$MAIN_NODE_IP" \
        --num_machines "$SLURM_NNODES" \
        --num_processes "$TOTAL_PROCESSES" \
        $TRAIN_SCRIPT \
        $TRAIN_ARGS \
        $HYDRA_OVERRIDES \
        2>&1 | tee "${LOG_DIR}/train_$(hostname).log"
'

EXIT_CODE=$?

#==================================================================================
# POST-TRAINING
#==================================================================================

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training exited with code $EXIT_CODE."
fi
echo "Job ID:    $SLURM_JOB_ID"
echo "Logs:      $LOG_DIR"
echo "SLURM out: /lambdafs/users/a.palmas/logs/slurm/job_${SLURM_JOB_ID}.out"
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints: $CHECKPOINT_DIR (on /raid — collect from nodes)"
else
    echo "Checkpoints: Hydra output_dir (check config for path)"
fi
echo "=============================================="

exit $EXIT_CODE
