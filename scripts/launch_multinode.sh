#!/bin/bash
# shellcheck disable=SC2029
# shellcheck disable=SC2015
# shellcheck disable=SC2329
# Multi-node DeepSpeed Zero3 training launcher
# Uses local /scratch on each node for maximum performance
# Code must be manually synced before running (use sync_code_to_nodes.sh)

set -e

#==================================================================================
# USAGE
#==================================================================================

usage() {
    echo "Usage: $0 <node1> <node2> [node3] ..."
    echo ""
    echo "Launch multi-node DeepSpeed Zero3 training across specified GPU nodes."
    echo ""
    echo "Arguments:"
    echo "  node1, node2, ...   List of compute nodes to use (at least 2 required)"
    echo ""
    echo "Environment variables:"
    echo "  LOCAL_WORKSPACE     Workspace path on nodes (default: /scratch/a.palmas/code-interp-benchmark)"
    echo "  ACCELERATE_CONFIG   Accelerate config file (default: pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml)"
    echo "  TRAIN_ARGS          Additional training arguments"
    echo "  CHECKPOINT_DIR      Checkpoint output directory"
    echo "  LOG_DIR             Log directory on NAS"
    echo "  CACHE_BASE          Cache directory on nodes (default: /scratch/a.palmas/tmp/cache)"
    echo "  RECREATE_CACHE      Delete and recreate cache before training (default: false)"
    echo "  ENABLE_DEBUG        Enable NCCL/PyTorch debug logging (default: false, WARNING: 10-30%+ overhead)"
    echo "  RESUME_FROM_RUN_DIR Path to resume from (checkpoint parent dir, e.g., /scratch/.../run_xxx)"
    echo "  RESUME_CHECKPOINT   Specific checkpoint name to resume from (e.g., checkpoint-500)"
    echo ""
    echo "Examples:"
    echo "  $0 gpu05 gpu06"
    echo "  $0 gpu01 gpu02 gpu03 gpu04"
    echo "  TRAIN_ARGS='--config experiment.yaml' $0 gpu05 gpu06"
    echo "  RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250201_120000 $0 gpu05 gpu06"
    exit 1
}

# Check for help flag or no arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    usage
fi

#==================================================================================
# CONFIGURATION
#==================================================================================

COMPUTE_NODES=("$@")
NUM_NODES=${#COMPUTE_NODES[@]}
GPUS_PER_NODE=8
TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

# Local workspace on each node's /scratch (must be synced manually!)
LOCAL_WORKSPACE="${LOCAL_WORKSPACE:-/scratch/a.palmas/code-interp-benchmark}"

# Accelerate config (relative to LOCAL_WORKSPACE)
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml}"

# Training script and arguments
TRAIN_SCRIPT="pyine/apps/trainers/hf_trainer.py"
TRAIN_ARGS="${TRAIN_ARGS:-}"

# Cache directories (local scratch for performance)
# All caches are redirected to local scratch to avoid NFS issues (e.g., Kerberos expiration)
# Cache is reused across runs to avoid re-downloading models (prevents HF rate limits)
LOCAL_SCRATCH_BASE="${LOCAL_SCRATCH_BASE:-/scratch/a.palmas/tmp}"
CACHE_BASE="${CACHE_BASE:-${LOCAL_SCRATCH_BASE}/cache}"

# Option to force recreate cache (delete existing cache before training)
# Usage: RECREATE_CACHE=true ./scripts/launch_multinode.sh gpu01 gpu02
RECREATE_CACHE="${RECREATE_CACHE:-false}"

# Checkpoints - by default, use Hydra's output_dir (keeps checkpoints with configs for easy resume)
# Override with: export CHECKPOINT_DIR=/your/path (note: if overridden, resume requires manual config handling)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

#==================================================================================
# RESUME CONFIGURATION
#==================================================================================
# Resume from a previous run's checkpoint directory
# Usage: RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250201_120000 ./scripts/launch_multinode.sh gpu05 gpu06
# Optionally specify a specific checkpoint: RESUME_CHECKPOINT=checkpoint-500
#
# IMPORTANT: When resuming on the same nodes, each rank loads only its own shard.
# The checkpoint shards must be in the same location on each node as when they were saved.
# If resuming on different nodes, first collect all shards to a shared location (NFS).
RESUME_FROM_RUN_DIR="${RESUME_FROM_RUN_DIR:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

#==================================================================================
# PROFILING CONFIGURATION (Nsight Systems)
#==================================================================================
# Enable profiling: export ENABLE_PROFILING=true
# Profiling levels:
#   - minimal:  CUDA/NVTX tracing only (~1-5% overhead)
#   - moderate: + OS runtime, syscalls (~5-15% overhead)
#   - full:     + backtraces, sampling, memory tracking (>15% overhead)
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"
PROFILE_LEVEL="${PROFILE_LEVEL:-minimal}"

#==================================================================================
# DEBUG CONFIGURATION (NCCL/PyTorch Distributed)
#==================================================================================
# Enable debug mode: export ENABLE_DEBUG=true
# WARNING: Debug mode has significant performance overhead (10-30%+)
# Use only for diagnosing hangs, deadlocks, or collective operation failures.
#
# Debug flags enabled:
#   - NCCL_DEBUG=INFO              : Verbose NCCL logging
#   - NCCL_DEBUG_SUBSYS=ALL        : Log all NCCL subsystems
#   - TORCH_NCCL_TRACE_BUFFER_SIZE : Enable flight recorder for stack traces
#   - TORCH_DISTRIBUTED_DEBUG      : Detailed distributed debugging with barriers
#   - TORCH_SHOW_CPP_STACKTRACES   : Show C++ stack traces on errors
ENABLE_DEBUG="${ENABLE_DEBUG:-false}"

# Logs - saved to NAS (accessible from login node for monitoring)
# Training logs on NAS, but checkpoints/cache on scratch for performance
# Include "profiling" in folder name when profiling is enabled
if [ "$ENABLE_PROFILING" = true ]; then
    LOG_DIR="${LOG_DIR:-/nas/users/a.palmas/logs/multinode_profiling_${PROFILE_LEVEL}_$(date +%Y%m%d_%H%M%S)}"
else
    LOG_DIR="${LOG_DIR:-/nas/users/a.palmas/logs/multinode_$(date +%Y%m%d_%H%M%S)}"
fi

CLEANUP_ON_EXIT=true

# Build nsys profile command based on level
get_nsys_cmd() {
    local node="$1"
    local rank="$2"
    local output_file="${LOG_DIR}/profile_${node}_rank${rank}"

    case "$PROFILE_LEVEL" in
        minimal)
            # Minimal overhead: CUDA kernels and NVTX annotations only
            # --sample=none disables CPU sampling for lowest overhead
            echo "nsys profile -o ${output_file} --trace=cuda,nvtx --sample=none --cuda-memory-usage=false --cudabacktrace=none"
            ;;
        moderate)
            # Moderate overhead: + OS runtime and syscalls
            # Still no CPU sampling, but traces OS runtime calls
            echo "nsys profile -o ${output_file} --trace=cuda,nvtx,osrt --sample=none --cuda-memory-usage=false --cudabacktrace=none"
            ;;
        full)
            # Full profiling: + backtraces, CPU sampling, memory tracking
            # --sample=process-tree enables CPU sampling (default method)
            # --backtrace=dwarf provides detailed call stacks
            # Note: Requires perf_event_paranoid <= 2 (check with: nsys status -e)
            echo "nsys profile -o ${output_file} --trace=cuda,nvtx,osrt,cudnn,cublas --sample=process-tree --backtrace=dwarf --cuda-memory-usage=true --cudabacktrace=all"
            ;;
        *)
            error "Unknown PROFILE_LEVEL: $PROFILE_LEVEL (use: minimal, moderate, full)"
            exit 1
            ;;
    esac
}

#==================================================================================
# HELPER FUNCTIONS
#==================================================================================

log() {
    local msg
    msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    # Append to launcher log if directory exists
    if [ -d "$LOG_DIR" ]; then
        echo "$msg" >> "${LOG_DIR}/launcher.log" 2>/dev/null || true
    fi
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

cleanup() {
    log "Cleaning up training processes..."
    for node in "${COMPUTE_NODES[@]}"; do
        ssh "$node" "pkill -f 'accelerate.commands.launch' || true" 2>/dev/null || true
        ssh "$node" "pkill -f 'python.*${TRAIN_SCRIPT}' || true" 2>/dev/null || true
        ssh "$node" "pkill -f 'deepspeed' || true" 2>/dev/null || true
    done
    log "Cleanup complete"
}

if [ "$CLEANUP_ON_EXIT" = true ]; then
    trap cleanup EXIT INT TERM
fi

get_node_ip() {
    local node="$1"
    local ip
    ip=$(ssh "$node" "hostname -I | awk '{print \$1}'" 2>/dev/null)
    if [ -n "$ip" ]; then
        echo "$ip"
    else
        # Fallback to hostname if IP retrieval fails
        echo "$node"
    fi
}

# Build cache environment variables
# Redirects ALL known cache paths to local scratch to avoid NFS/Kerberos issues
get_cache_env_vars() {
    local cache_base="$1"
    cat << EOF
export HF_HOME="${cache_base}/huggingface" && \\
export HF_HUB_CACHE="${cache_base}/huggingface/hub" && \\
export HF_DATASETS_CACHE="${cache_base}/huggingface/datasets" && \\
export HF_ASSETS_CACHE="${cache_base}/huggingface/assets" && \\
export HF_MODULES_CACHE="${cache_base}/huggingface/modules" && \\
export TRANSFORMERS_CACHE="${cache_base}/huggingface/transformers" && \\
export HUGGINGFACE_HUB_CACHE="${cache_base}/huggingface/hub" && \\
export TORCH_HOME="${cache_base}/torch" && \\
export TORCH_EXTENSIONS_DIR="${cache_base}/torch_extensions" && \\
export TORCHINDUCTOR_CACHE_DIR="${cache_base}/torch_inductor" && \\
export PYTORCH_KERNEL_CACHE_PATH="${cache_base}/torch/kernels" && \\
export WANDB_DIR="${cache_base}/wandb" && \\
export WANDB_CACHE_DIR="${cache_base}/wandb/cache" && \\
export WANDB_CONFIG_DIR="${cache_base}/wandb/config" && \\
export WANDB_DATA_DIR="${cache_base}/wandb/data" && \\
export TIKTOKEN_CACHE_DIR="${cache_base}/tiktoken" && \\
export TRITON_CACHE_DIR="${cache_base}/triton" && \\
export MPLCONFIGDIR="${cache_base}/matplotlib" && \\
export XDG_CACHE_HOME="${cache_base}/xdg_cache" && \\
export XDG_CONFIG_HOME="${cache_base}/xdg_config" && \\
export XDG_DATA_HOME="${cache_base}/xdg_data" && \\
export CUDA_CACHE_PATH="${cache_base}/cuda_cache" && \\
export FLASH_ATTENTION_CACHE_DIR="${cache_base}/flash_attn" && \\
export VLLM_CACHE_DIR="${cache_base}/vllm" && \\
export TMPDIR="${cache_base}/tmp" && \\
export TEMP="${cache_base}/tmp" && \\
export TMP="${cache_base}/tmp" && \\
EOF
}

#==================================================================================
# VALIDATION
#==================================================================================

log "Multi-node training launcher (local /scratch mode)"
log "=================================================="
log "Nodes: ${COMPUTE_NODES[*]}"
log "Total processes: $TOTAL_PROCESSES ($NUM_NODES nodes × $GPUS_PER_NODE GPUs)"
log "Workspace: $LOCAL_WORKSPACE (on compute nodes - must be synced!)"
if [ -n "$CHECKPOINT_DIR" ]; then
    log "Checkpoints: $CHECKPOINT_DIR (custom path - collect after training)"
else
    log "Checkpoints: (using Hydra output_dir - checkpoints saved with configs)"
fi
log "Logs: $LOG_DIR (on NAS - accessible from login node)"
if [ "$ENABLE_PROFILING" = true ]; then
    log "Profiling: ENABLED (level: $PROFILE_LEVEL)"
else
    log "Profiling: disabled (set ENABLE_PROFILING=true to enable)"
fi
if [ "$ENABLE_DEBUG" = true ]; then
    log "Debug: ENABLED (NCCL_DEBUG=INFO, TORCH_DISTRIBUTED_DEBUG=DETAIL) - WARNING: performance overhead!"
else
    log "Debug: disabled (set ENABLE_DEBUG=true to enable)"
fi
log "Cache: $CACHE_BASE (reused across runs; set RECREATE_CACHE=true to clear)"
if [ -n "$RESUME_FROM_RUN_DIR" ]; then
    log "Resume: ENABLED from $RESUME_FROM_RUN_DIR"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        log "  Checkpoint: $RESUME_CHECKPOINT"
    else
        log "  Checkpoint: (auto-detect latest)"
    fi
else
    log "Resume: disabled (set RESUME_FROM_RUN_DIR to enable)"
fi
log ""

# Create log directory on NAS (login node has access)
log "Creating log directory on NAS..."
mkdir -p "$LOG_DIR" || {
    error "Cannot create log directory: $LOG_DIR"
    exit 1
}
log "  ✓ Created $LOG_DIR"

# Validate SSH connectivity
log "Validating nodes..."
for node in "${COMPUTE_NODES[@]}"; do
    if ! ssh -o ConnectTimeout=5 "$node" "echo 'OK'" &>/dev/null; then
        error "Cannot connect to $node"
        exit 1
    fi
    log "  ✓ $node accessible"
done

# Validate workspace exists on all nodes
log "Validating workspace..."
for node in "${COMPUTE_NODES[@]}"; do
    if ! ssh "$node" "test -d ${LOCAL_WORKSPACE}" 2>/dev/null; then
        error "Workspace missing on $node: ${LOCAL_WORKSPACE}"
        error "Run: ./scripts/sync_code_to_nodes.sh"
        exit 1
    fi
    log "  ✓ $node has ${LOCAL_WORKSPACE}"
done

# Define main node (first in list)
MAIN_NODE="${COMPUTE_NODES[0]}"

# Pull latest code on all nodes
log "Pulling latest code on all nodes..."
for node in "${COMPUTE_NODES[@]}"; do
    PULL_OUTPUT=$(ssh "$node" "cd ${LOCAL_WORKSPACE} && git pull" 2>&1) || {
        error "Git pull failed on $node: $PULL_OUTPUT"
        exit 1
    }
    # Show abbreviated output (first line or "Already up to date")
    PULL_SUMMARY=$(echo "$PULL_OUTPUT" | head -1)
    log "  ✓ $node: $PULL_SUMMARY"
done

# Basic sync check (compare Python files) - DISABLED
# Uncomment to enable automatic sync checking
# log "Checking code sync..."
# MAIN_CHECKSUM=$(ssh "$MAIN_NODE" "find ${LOCAL_WORKSPACE} -name '*.py' -type f -exec md5sum {} \; 2>/dev/null | sort | md5sum" 2>/dev/null || echo "")
# for node in "${COMPUTE_NODES[@]:1}"; do
#     NODE_CHECKSUM=$(ssh "$node" "find ${LOCAL_WORKSPACE} -name '*.py' -type f -exec md5sum {} \; 2>/dev/null | sort | md5sum" 2>/dev/null || echo "")
#     if [ "$MAIN_CHECKSUM" != "$NODE_CHECKSUM" ]; then
#         error "Code mismatch on $node! Run: ./scripts/sync_code_to_nodes.sh"
#         exit 1
#     fi
# done
# log "  ✓ Code synced across nodes"
log "⚠ Skipping code sync check (verify manually!)"

# Validate resume checkpoint directory exists on all nodes (if resuming)
if [ -n "$RESUME_FROM_RUN_DIR" ]; then
    if [ -n "$RESUME_CHECKPOINT" ]; then
        # Specific checkpoint provided - validate it exists with shards on all nodes
        log "Validating resume checkpoint directory..."
        RESUME_CHECK_PATH="${RESUME_FROM_RUN_DIR}/${RESUME_CHECKPOINT}"
        for node in "${COMPUTE_NODES[@]}"; do
            if ! ssh "$node" "test -d ${RESUME_CHECK_PATH}" 2>/dev/null; then
                error "Resume checkpoint directory missing on $node: ${RESUME_CHECK_PATH}"
                error "Ensure checkpoint shards are present on all nodes before resuming."
                exit 1
            fi
            # Check for global_step* folder with DeepSpeed shards (.pt files)
            # DeepSpeed Zero3 saves shards as: zero_pp_rank_*_model_states.pt and bf16_zero_pp_rank_*_optim_states.pt
            GLOBAL_STEP_DIR=$(ssh "$node" "ls -d ${RESUME_CHECK_PATH}/global_step* 2>/dev/null | head -1" 2>/dev/null || echo "")
            if [ -n "$GLOBAL_STEP_DIR" ]; then
                SHARD_COUNT=$(ssh "$node" "ls ${GLOBAL_STEP_DIR}/zero_pp_rank_*_model_states.pt 2>/dev/null | wc -l" 2>/dev/null || echo "0")
                log "  ✓ $node has checkpoint at ${RESUME_CHECK_PATH} (${SHARD_COUNT} DeepSpeed shards in global_step folder)"
            else
                # Fallback: check for HF-style shards or consolidated model (for non-DeepSpeed or inference)
                HF_SHARD_COUNT=$(ssh "$node" "ls ${RESUME_CHECK_PATH}/pytorch_model-rank-*.bin ${RESUME_CHECK_PATH}/model*.safetensors 2>/dev/null | wc -l" 2>/dev/null || echo "0")
                if [ "$HF_SHARD_COUNT" -eq 0 ]; then
                    error "No checkpoint shards found on $node in ${RESUME_CHECK_PATH}"
                    error "Expected global_step*/zero_pp_rank_*_model_states.pt (DeepSpeed) or model*.safetensors (HF format)."
                    exit 1
                fi
                log "  ✓ $node has checkpoint at ${RESUME_CHECK_PATH} (${HF_SHARD_COUNT} HF-format files)"
            fi
        done
    else
        # No specific checkpoint - skip validation, trainer will auto-detect latest
        log "⚠ RESUME_CHECKPOINT not specified - skipping checkpoint shard validation"
        log "  Trainer will auto-detect latest checkpoint in: $RESUME_FROM_RUN_DIR"
        log "  If checkpoint is missing or invalid, training will fail at startup"
    fi
fi

# Handle cache recreation if requested
if [ "$RECREATE_CACHE" = true ]; then
    log "Recreating cache directories on compute nodes (RECREATE_CACHE=true)..."
    for node in "${COMPUTE_NODES[@]}"; do
        ssh "$node" "rm -rf ${CACHE_BASE} && mkdir -p ${CACHE_BASE}" 2>/dev/null || {
            error "Cannot recreate cache on $node"
            exit 1
        }
        log "  ✓ $node cache recreated"
    done
fi

# Create directories on compute nodes
log "Creating directories on compute nodes..."
for node in "${COMPUTE_NODES[@]}"; do
    # Create scratch dirs, cache base, and optionally checkpoint dir; ensure NAS log dir is accessible
    MKDIR_CMD="mkdir -p ${LOCAL_SCRATCH_BASE} ${CACHE_BASE}"
    if [ -n "$CHECKPOINT_DIR" ]; then
        MKDIR_CMD="${MKDIR_CMD} ${CHECKPOINT_DIR}"
    fi
    ssh "$node" "${MKDIR_CMD} && test -d ${LOG_DIR}" 2>/dev/null || {
        error "Cannot create directories on $node or NAS not accessible"
        exit 1
    }
    log "  ✓ $node directories ready"
done

# Get main node IP for DeepSpeed coordination
MAIN_NODE_IP=$(get_node_ip "$MAIN_NODE")
if [ -z "$MAIN_NODE_IP" ]; then
    error "Failed to get IP for main node: $MAIN_NODE"
    exit 1
fi
log "Main node: $MAIN_NODE ($MAIN_NODE_IP)"
log ""

#==================================================================================
# LAUNCH
#==================================================================================

log "Launching training..."
log "  - DeepSpeed Zero Stage 3"
log "  - vLLM colocated on training GPUs"
log "  - Cache base: ${CACHE_BASE}"
log ""

# Array to track SSH process PIDs
SSH_PIDS=()

for i in "${!COMPUTE_NODES[@]}"; do
    node="${COMPUTE_NODES[$i]}"
    machine_rank=$i

    log "Starting $node (rank $machine_rank)..."

    # Build checkpoint output dir override if needed
    CHECKPOINT_OVERRIDE=""
    if [ -n "$CHECKPOINT_DIR" ]; then
        CHECKPOINT_OVERRIDE="config.grpo_config.output_dir=${CHECKPOINT_DIR}"
    fi

    # Build resume override if resuming from a previous run
    RESUME_OVERRIDE=""
    if [ -n "$RESUME_FROM_RUN_DIR" ]; then
        RESUME_OVERRIDE="config.resume_from_run_dir=${RESUME_FROM_RUN_DIR}"
        if [ -n "$RESUME_CHECKPOINT" ]; then
            RESUME_OVERRIDE="${RESUME_OVERRIDE} config.resume_checkpoint_name=${RESUME_CHECKPOINT}"
        fi
    fi

    # Build profiler prefix if enabled
    PROFILER_CMD=""
    if [ "$ENABLE_PROFILING" = true ]; then
        PROFILER_CMD="$(get_nsys_cmd "$node" "$machine_rank") "
    fi

    # Build debug environment variables if enabled
    DEBUG_ENV_VARS=""
    if [ "$ENABLE_DEBUG" = true ]; then
        # NCCL_TIMEOUT: seconds before collective ops timeout (default 1800=30min)
        # Use shorter timeout for faster debugging: NCCL_TIMEOUT=300 (5min)
        NCCL_TIMEOUT_VAL="${NCCL_TIMEOUT:-1800}"
        DEBUG_ENV_VARS="export NCCL_DEBUG=INFO && \
        export NCCL_DEBUG_SUBSYS=ALL && \
        export TORCH_NCCL_TRACE_BUFFER_SIZE=1000 && \
        export TORCH_DISTRIBUTED_DEBUG=DETAIL && \
        export TORCH_SHOW_CPP_STACKTRACES=1 && \
        export NCCL_TIMEOUT=${NCCL_TIMEOUT_VAL} && "
    fi

    # Build cache environment variables (all redirected to local scratch)
    CACHE_ENV_VARS=$(get_cache_env_vars "${CACHE_BASE}")

    # Launch training on the node with krenew for Kerberos ticket renewal
    ssh "$node" "krenew -K 60 -- bash -l -c '
        cd ${LOCAL_WORKSPACE} && \
        mkdir -p ${CACHE_BASE}/tmp && \
        ${CACHE_ENV_VARS}
        export ACCELERATE_MACHINE_RANK=${machine_rank} && \
        export ACCELERATE_MAIN_PROCESS_IP=${MAIN_NODE_IP} && \
        export ACCELERATE_NUM_MACHINES=${NUM_NODES} && \
        export ACCELERATE_NUM_PROCESSES=${TOTAL_PROCESSES} && \
        ${DEBUG_ENV_VARS}
        nohup ${PROFILER_CMD}uv run accelerate launch \
            --config_file ${ACCELERATE_CONFIG} \
            --machine_rank ${machine_rank} \
            --main_process_ip ${MAIN_NODE_IP} \
            --num_machines ${NUM_NODES} \
            --num_processes ${TOTAL_PROCESSES} \
            ${TRAIN_SCRIPT} \
            ${TRAIN_ARGS} \
            ${CHECKPOINT_OVERRIDE} \
            ${RESUME_OVERRIDE} \
            > ${LOG_DIR}/train_${node}.log 2>&1 &
        echo \$!
    '" > "${LOG_DIR}/train_${node}.pid" &

    # Capture SSH process PID
    SSH_PIDS+=($!)
    log "  ✓ Launched on $node"
done

#==================================================================================
# WAIT FOR SSH PROCESSES TO COMPLETE
#==================================================================================

log ""
log "Waiting for SSH processes to establish remote training..."

# Wait for all SSH processes to complete (they return after nohup detaches)
for pid in "${SSH_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || {
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            error "SSH process (PID: $pid) failed with exit code $exit_code"
        fi
    }
done

# Give remote processes a moment to fully initialize
sleep 2

# Verify training processes are running on each node
log "Verifying training processes started..."
ALL_RUNNING=true
for node in "${COMPUTE_NODES[@]}"; do
    if ssh "$node" "pgrep -f 'accelerate.commands.launch'" &>/dev/null; then
        log "  ✓ $node: training process running"
    else
        error "  ✗ $node: training process NOT found"
        ALL_RUNNING=false
    fi
done

if [ "$ALL_RUNNING" = false ]; then
    error "Some training processes failed to start. Check logs in $LOG_DIR"
    exit 1
fi

log ""
log "All training processes started successfully!"

log ""
log "Training launched on all nodes"
log "Logs: $LOG_DIR"
log ""
log "Monitor with:"
log "  tail -f ${LOG_DIR}/train_*.log"
log "  ssh <node> nvidia-smi"
log ""
log "Check training processes with:"
log "  for node in ${COMPUTE_NODES[*]}; do ssh \$node 'ps aux | grep accelerate | grep -v grep'; done"
log ""
log "Stop training with:"
log "  for node in ${COMPUTE_NODES[*]}; do ssh \$node 'pkill -f accelerate.commands.launch'; done"
log ""
if [ -n "$CHECKPOINT_DIR" ]; then
    log "Checkpoints will be saved to: $CHECKPOINT_DIR"
    log "To collect checkpoints from all nodes:"
    log "  mkdir -p /your/destination/"
    for node in "${COMPUTE_NODES[@]}"; do
        log "  scp -r $node:${CHECKPOINT_DIR}/* /your/destination/"
    done
else
    log "Checkpoints will be saved to Hydra output_dir (same location as configs)"
    log "Check your Hydra config for the exact output path."
fi
log ""
if [ "$ENABLE_PROFILING" = true ]; then
    log "PROFILING ENABLED (level: $PROFILE_LEVEL)"
    log "Profile files will be saved to: ${LOG_DIR}/profile_*.nsys-rep"
    log ""
    log "After training completes, view profiles with:"
    log "  nsys-ui ${LOG_DIR}/profile_<node>_rank<N>.nsys-rep"
    log ""
    log "Or generate a summary report:"
    log "  nsys stats ${LOG_DIR}/profile_<node>_rank<N>.nsys-rep"
    log ""
fi
if [ "$ENABLE_DEBUG" = true ]; then
    log "DEBUG MODE ENABLED"
    log "Debug flags active:"
    log "  - NCCL_DEBUG=INFO"
    log "  - NCCL_DEBUG_SUBSYS=ALL"
    log "  - TORCH_NCCL_TRACE_BUFFER_SIZE=1000"
    log "  - TORCH_DISTRIBUTED_DEBUG=DETAIL"
    log "  - TORCH_SHOW_CPP_STACKTRACES=1"
    log "  - NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}s"
    log ""
    log "Logs will contain detailed NCCL communication info and stack traces on failures."
    log "WARNING: Expect 10-30%+ performance overhead with debug enabled."
    log ""
    log "Tip: Use NCCL_TIMEOUT=300 for faster failure detection (5min instead of 30min)"
    log ""
fi
log "Script complete. Training is running in background on all nodes."

exit 0
