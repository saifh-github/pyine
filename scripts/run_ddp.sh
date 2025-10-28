#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------
# run_ddp.sh - Launch Hydra/Transformers experiments with torchrun, forwarding overrides.
#
# IMPORTANT:
#   - This script REQUIRES bash, not sh. SLURM job scripts sometimes default to /bin/sh.
#     Use `bash run_ddp.sh`, not `sh run_ddp.sh`.
#
#   - DO NOT pass secrets / tokens / passwords / access keys as command-line args after `--`.
#     This script logs the final torchrun command (including overrides) to disk for provenance.
#     Secrets MUST come from a trusted .env file or some secure mechanism, NOT via CLI args.
#
# This script is designed for robustness and debuggability:
#   - forwards everything after `--` to the target Python app (e.g. Hydra overrides);
#   - works on single-node and multi-node (including SLURM);
#   - auto-detects GPU count if you do not specify one;
#   - captures a clean launcher log (optionally via `tee`);
#   - sets a few safe environment defaults for NCCL and Hugging Face;
#   - chooses a non-conflicting rendezvous port for single-node jobs when possible.
#
# Usage examples are shown in `usage()` below.
# ------------------------------------------------------------------------------------------

# fail fast and make word-splitting predictable:
set -euo pipefail
IFS=$'\n\t'

###################################################
#    Early .env loading (BEFORE anything else)
###################################################
# We support an env file according to this priority:
#   1) `--env-file <path>` flag (first occurrence on CLI, parsed in this pre-pass)
#   2) $ENVFILE environment variable
#   3) ./.env in current working directory
#   4) <script_dir>/.env (useful when launching from elsewhere)
#
# The env file must be a trusted shell fragment with KEY=VALUE lines. We source it *before*
# parsing the rest of the flags so the rest of the script can rely on those env vars.
#
# SECURITY: we do NOT print env vars, to avoid leaking secrets.
ENVFILE="${ENVFILE:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# pre-parse ONLY --env-file from "$@" (stop at `--`)
if [[ -z "${ENVFILE}" ]]; then
  ARGV_FOR_ENV=("$@")
  for ((env_idx=0; env_idx < ${#ARGV_FOR_ENV[@]}; env_idx++)); do
    arg_name="${ARGV_FOR_ENV[env_idx]}"
    if [[ "${arg_name}" == "--" ]]; then
      break
    fi
    if [[ "${arg_name}" == "--env-file" ]]; then
      next_idx=$((env_idx + 1))
      if (( next_idx >= ${#ARGV_FOR_ENV[@]} )); then
        echo "Error: --env-file requires a path argument."
        exit 1
      fi
      ENVFILE="${ARGV_FOR_ENV[next_idx]}"
      break
    fi
  done
fi

# fallback to local/default .env if still unset
if [[ -z "${ENVFILE}" ]]; then
  if [[ -f ".env" ]]; then
    ENVFILE=".env"
  elif [[ -f "${SCRIPT_DIR}/.env" ]]; then
    ENVFILE="${SCRIPT_DIR}/.env"
  fi
fi

# load .env if present; enable "allexport" so all loaded names are exported automatically
SOURCED_ENVFILE=""
if [[ -n "${ENVFILE}" && -f "${ENVFILE}" ]]; then
  echo "[dotenv] loading ${ENVFILE}"
  set -a # export all variables assigned during source
  # shellcheck disable=SC1090
  . "${ENVFILE}"
  set +a
  SOURCED_ENVFILE="${ENVFILE}"
else
  echo "[dotenv] no .env file found (skip)"
fi

###################################################
#    Defaults (override via flags or env vars)
###################################################
APP_MODULE="${APP_MODULE:-pyine.apps.trainers.hf_trainer}" # Python entrypoint (defaults to hf trainer)
NNODES="${NNODES:-1}"                                      # number of nodes in the job
NODE_RANK="${NODE_RANK:-0}"                                # this node's rank among [0..NNODES-1].
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"                   # number of processes per node (GPUs)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"                    # rendezvous IP or hostname for rank 0 node
MASTER_PORT_DEFAULT="29500"
MASTER_PORT_PRESET=0
if [[ -n "${MASTER_PORT+x}" ]]; then MASTER_PORT_PRESET=1; fi
MASTER_PORT="${MASTER_PORT:-${MASTER_PORT_DEFAULT}}"       # rendezvous TCP port
_MASTER_PORT_IS_DEFAULT=1
if [[ "${MASTER_PORT_PRESET}" == "1" ]]; then _MASTER_PORT_IS_DEFAULT=0; fi
if [[ "${MASTER_PORT}" != "${MASTER_PORT_DEFAULT}" ]]; then _MASTER_PORT_IS_DEFAULT=0; fi
MASTER_PORT_FROM_FLAG=0

LOG_DIR="${LOG_DIR:-logs/ddp}"                             # where to write the launcher log
RUN_NAME="${RUN_NAME:-$(date +'%Y%m%d_%H%M%S')_rank${NODE_RANK}_$$}" # label appended to log filename
TEE_LOG="${TEE_LOG:-1}"                                    # 1 -> console+log, 0 -> console only, other -> log only
DEBUG="${DEBUG:-0}"                                        # 1 -> verbose NCCL / distributed debug (no shell trace)
DRY_RUN="${DRY_RUN:-0}"                                    # 1 -> print info and exit without launching

# optional environment knobs
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"   # avoid tokenizer thread storms
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}" # safer collective error handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" # future-proof
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"                            # quiet unless debugging
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"                     # keep CPU threads in check
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" # fewer OOM stalls
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"                   # full Hydra tracebacks (no trimming)
PYTHON_BIN="${PYTHON_BIN:-python}"                                 # which Python to use (override if needed)

###################################################
#                 Helper defs
###################################################

usage() {
  # Prints a concise help with examples; kept here instead of README for quick access.
  cat <<EOF
Usage:
  $(basename "$0") [options] -- [APP ARGS / HYDRA OVERRIDES]

NOTE: This script REQUIRES bash, not sh. If you are in SLURM, call it with bash.

SECURITY WARNING:
  Do NOT pass secrets (API keys, tokens, passwords, etc.) after the "--".
  This script logs the full launch command (including overrides) to disk
  for reproducibility. Secrets should come from a trusted .env file.

Options:
  --env-file <path>            Source KEY=VALUE pairs before parsing other flags
  --app <module_or_path>       Python entrypoint passed to \`-m\` (default: ${APP_MODULE})
  --nnodes <int>               Number of nodes (default: ${NNODES})
  --node_rank <int>            Rank of this node [0..nnodes-1] (default: ${NODE_RANK})
  --nproc_per_node <int|auto>  Processes per node (default: ${NPROC_PER_NODE})
  --master_addr <addr>         Master address (default: ${MASTER_ADDR})
  --master_port <port>         Master port (default: ${MASTER_PORT})
  --log_dir <path>             Directory for launcher logs (default: ${LOG_DIR})
  --run_name <name>            Label used in log filename (default: ${RUN_NAME})
  --dry-run                    Print the resolved command and exit
  --debug                      Enable NCCL / torch.distributed debug logging
  -h, --help                   Show this help

Environment knobs (via env vars):
  TEE_LOG=0                    Console only (disable file logging)
  PYTHON_BIN=python3           Select interpreter used for the target module
  DEBUG=1                      Verbose NCCL + TORCH_DISTRIBUTED_DEBUG=DETAIL
  OMP_NUM_THREADS=4            Increase from default (1) if dataloader is CPU-bound

Examples:
  # single-node, auto GPU count, with Hydra overrides forwarded after --
  $(basename "$0") -- --experiment_name=demo trainer.learning_rate=3e-5 data.path=/data

  # single-node, explicit 8 GPUs
  $(basename "$0") --nproc_per_node 8 -- --experiment_name=demo trainer.per_device_train_batch_size=2

  # multi-node (manual)
  $(basename "$0") --nnodes 2 --node_rank 0 --master_addr hostA --master_port 29501 -- --seed=1337

  # under SLURM inside a job script (env inferred automatically)
  # NOTE: In SLURM, request --ntasks-per-node=1 and call srun once, so this script
  #       runs once per node (not once per GPU). torchrun spawns GPU processes internally.
  srun $(basename "$0") -- --experiment_name=slurm_run

  # dry run (no launch, just summary + command)
  $(basename "$0") --dry-run -- --experiment_name=inspect_only
EOF
}

log() {
  # Small timestamped logger.
  echo "[$(date +'%F %T')] $*";
}

detect_gpus() {
  # Returns how many visible GPUs to use as processes-per-node.
  # Priority: explicit CUDA_VISIBLE_DEVICES -> nvidia-smi -> fallback 1.
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    local count
    # count comma-separated device IDs, filtering out empty entries
    count=$(awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print (n>0)?n:1}' <<<"${CUDA_VISIBLE_DEVICES}")
    echo "${count}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count=$(nvidia-smi -L | wc -l | awk '{print $1}')
    echo "$((count > 0 ? count : 1))"
  else
    echo 1
  fi
}

first_slurm_host() {
  # On SLURM: derive the master address as the first hostname in the job nodelist.
  # Use `|| true` to avoid `set -e` killing the script if scontrol fails.
  scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | head -n1 || true
}

free_port() {
  # Find a free TCP port on this machine. Uses Python for cross-platform socket probing.
  # Note: there's a small race condition between finding and binding the port; acceptable in practice.
  "${PYTHON_BIN}" - <<'PY'
import os, socket, contextlib
port = int(os.environ.get("MASTER_PORT","0")) or 0
with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(("", port))  # port=0 -> OS picks a free one; else try the given port
    print(s.getsockname()[1])
PY
}

derive_slurm_master_port() {
  # Try to pick a deterministic-but-job-specific port under SLURM to avoid collisions.
  # This function never fails; if Python execution fails, it falls back to MASTER_PORT_DEFAULT
  # to ensure all nodes use the same port even in edge cases.
  MASTER_PORT_DEFAULT="${MASTER_PORT_DEFAULT}" "${PYTHON_BIN}" - 2>/dev/null <<'PY' || echo "${MASTER_PORT_DEFAULT}"
import hashlib
import os

payload = ":".join(
    value.strip()
    for value in (
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("SLURM_STEP_ID", ""),
    )
    if value and value.strip()
)
default_port = int(os.environ.get("MASTER_PORT_DEFAULT", "29500"))
if not payload:
    print(default_port)
else:
    port_min = 15000
    port_span = 45000
    digest = int(hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest(), 16)
    print(port_min + (digest % port_span))
PY
}

###################################################
#      Parse args (stop at -- for app args)
###################################################
# We split our own flags from the app/Hydra arguments; everything after `--` is forwarded verbatim
# to Python (so arguments such as Hydra overrides pass through cleanly).
APP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      if [[ -z "${2-}" ]]; then
        echo "Error: --env-file requires a path argument."
        exit 1
      fi
      ENVFILE="$2"
      shift 2
      ;;
    --app) APP_MODULE="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --node_rank) NODE_RANK="$2"; shift 2 ;;
    --nproc_per_node) NPROC_PER_NODE="$2"; shift 2 ;;
    --master_addr) MASTER_ADDR="$2"; shift 2 ;;
    --master_port)
      MASTER_PORT="$2"
      MASTER_PORT_FROM_FLAG=1
      _MASTER_PORT_IS_DEFAULT=0
      shift 2
      ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; APP_ARGS=("$@"); break ;;   # all remaining args go to the app
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

###################################################
#   Validate that the sourced env file matches
###################################################
# We sourced .env *before* this full parse. If the user tried to change --env-file later,
# that means we'd be running with env vars from one file but *claiming* another.
# That's confusing / unsafe, so we refuse and ask the user to put --env-file first.
if [[ -n "${ENVFILE}" && "${ENVFILE}" != "${SOURCED_ENVFILE}" ]]; then
  # Case 1: we previously sourced a different file => mismatch.
  # Case 2: we sourced nothing (SOURCED_ENVFILE="") but now ENVFILE is set => too late.
  echo "Error: --env-file must be provided before any other options so it can be sourced exactly once."
  echo "       Sourced: '${SOURCED_ENVFILE:-<none>}' ; Requested: '${ENVFILE}'"
  exit 1
fi

###################################################
#   Re-evaluate MASTER_PORT default-ness
###################################################
if [[ "${MASTER_PORT}" != "${MASTER_PORT_DEFAULT}" ]]; then
  _MASTER_PORT_IS_DEFAULT=0
elif [[ "${MASTER_PORT_FROM_FLAG}" == "1" ]]; then
  _MASTER_PORT_IS_DEFAULT=0
elif [[ "${MASTER_PORT_PRESET}" == "1" ]]; then
  _MASTER_PORT_IS_DEFAULT=0
else
  _MASTER_PORT_IS_DEFAULT=1
fi

###################################################
#      SLURM integration (auto if present)
###################################################
# If running under SLURM, infer ranks, nodes, rendezvous info, and GPU counts:
if [[ -n "${SLURM_JOB_ID-}" ]]; then
  NNODES="${SLURM_NNODES:-$NNODES}"
  NODE_RANK="${SLURM_NODEID:-$NODE_RANK}"

  # override MASTER_ADDR with first SLURM node if still at default
  if [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    MASTER_ADDR="$(first_slurm_host)"
  fi

  if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Error: Unable to infer MASTER_ADDR from SLURM (empty host list?)"
    exit 1
  fi

  # if port was not explicitly set by user, pick a job-specific port to reduce conflicts across jobs
  if [[ "${_MASTER_PORT_IS_DEFAULT}" == "1" ]]; then
    MASTER_PORT="$(derive_slurm_master_port)"
    _MASTER_PORT_IS_DEFAULT=0
  fi

  # prefer SLURM_GPUS_ON_NODE when auto-detecting per-node processes
  # NOTE: we expect SLURM_GPUS_ON_NODE to be a plain integer like "8".
  #       Some clusters expose strings like "gpu:4"; that will fail numeric
  #       validation later, which is intentional (fail loud).
  if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-auto}"
  fi
fi

###################################################
#            Finalize derived values
###################################################
# resolve "auto" -> actual GPU count (using CUDA_VISIBLE_DEVICES or nvidia-smi)
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  NPROC_PER_NODE="$(detect_gpus)"
fi

# validate numeric parameters
if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NNODES must be a positive integer (got: ${NNODES})"
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "Error: NODE_RANK must be a non-negative integer (got: ${NODE_RANK})"
  exit 1
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NPROC_PER_NODE must be a positive integer (got: ${NPROC_PER_NODE})"
  exit 1
fi
if [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "Error: NODE_RANK (${NODE_RANK}) must be less than NNODES (${NNODES})"
  exit 1
fi
if ! [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "Error: MASTER_PORT must be an integer in [1,65535] (got: ${MASTER_PORT})"
  exit 1
fi

# sanity for multi-node when not under SLURM:
if [[ "${NNODES}" != "1" && -z "${SLURM_JOB_ID-}" ]]; then
  # user is claiming multi-node manually; MASTER_ADDR must NOT stay 127.0.0.1
  if [[ "${MASTER_ADDR}" == "127.0.0.1" || -z "${MASTER_ADDR}" ]]; then
    echo "Error: multi-node launch requires a reachable --master_addr (rank0 hostname/IP)."
    echo "       MASTER_ADDR is '${MASTER_ADDR}'."
    exit 1
  fi
fi

# debug mode: louder NCCL / torch.distributed for diagnosing hangs/timeouts
if [[ "${DEBUG}" == "1" ]]; then
  # We intentionally DO NOT `set -x`, to avoid leaking secrets from the sourced .env
  export NCCL_DEBUG=INFO
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

###################################################
#          Assemble torchrun command
###################################################
# If we're single-node (no SLURM, NNODES==1) and the master port is still the default,
# pick an ephemeral free port so that two concurrent jobs on the same machine don't collide.
if [[ "${NNODES}" == "1" && -z "${SLURM_JOB_ID-}" ]]; then
  if [[ "${_MASTER_PORT_IS_DEFAULT}" == "1" ]]; then
    MASTER_PORT="$(free_port)"
    _MASTER_PORT_IS_DEFAULT=0
  fi
fi

# build commands using bash arrays to preserve correct quoting and avoid word-splitting bugs
TORCHRUN_BASE=(torchrun --nproc_per_node "${NPROC_PER_NODE}")

# for single-node (non-SLURM), --standalone simplifies rendezvous;
# we still include master_port for clarity / safety.
if [[ "${NNODES}" == "1" && -z "${SLURM_JOB_ID-}" ]]; then
  TORCHRUN_BASE+=(--standalone --master_port "${MASTER_PORT}")
else
  TORCHRUN_BASE+=(--nnodes "${NNODES}" --node_rank "${NODE_RANK}" \
                  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}")
fi

# use `python -m <module>` rather than a script path to keep import semantics clean
PY_ENTRY=("${PYTHON_BIN}" -m "${APP_MODULE}")

# final command: torchrun launcher + python module + forwarded app or Hydra args
CMD=("${TORCHRUN_BASE[@]}" "${PY_ENTRY[@]}" "${APP_ARGS[@]}")

if ! command -v torchrun >/dev/null 2>&1; then
  echo "Error: torchrun not found on PATH. Install PyTorch with distributed support." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: PYTHON_BIN='${PYTHON_BIN}' not found on PATH." >&2
  exit 1
fi

COMMAND_PRETTY="$(printf '%q ' "${CMD[@]}")"
COMMAND_PRETTY="${COMMAND_PRETTY% }"

###################################################
#     Pre-run summary (before dry-run/files)
###################################################
# gather extra provenance info (git commit, world size, etc.)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
# mark commit as +dirty if there are ANY unstaged OR staged-but-uncommitted changes
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  GIT_DIRTY="+dirty"
else
  GIT_DIRTY=""
fi

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
LAUNCH_LOG="${LOG_DIR}/launch_${RUN_NAME}.log"

log "=== Launch Summary ==="
log "date:       $(date +'%Y-%m-%d %H:%M:%S %Z')"
log "app:        ${APP_MODULE}"
log "nodes:      ${NNODES}  (node_rank=${NODE_RANK})"
log "per-node:   ${NPROC_PER_NODE} processes"
log "world size: ${WORLD_SIZE} total processes"
log "master:     ${MASTER_ADDR}:${MASTER_PORT}"
if [[ "${TEE_LOG}" == "0" ]]; then
  log "log dir:    (disabled by TEE_LOG=0)"
else
  log "log dir:    ${LOG_DIR}"
fi
log "run name:   ${RUN_NAME}"
log "git:        ${GIT_COMMIT}${GIT_DIRTY}"
if [[ -n "${SLURM_JOB_ID-}" ]]; then
  log "SLURM:      job_id=${SLURM_JOB_ID} nodelist=${SLURM_JOB_NODELIST:-N/A}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
  log "CUDA:       visible_devices=${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  log "CUDA:       $(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1) GPUs available"
fi
log "env:        OMP_NUM_THREADS=${OMP_NUM_THREADS} TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
if [[ "${DEBUG}" == "1" ]]; then
  log "env:        DEBUG=1 (NCCL_DEBUG=${NCCL_DEBUG}, TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-})"
fi
log "python:     ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
log "command:    ${COMMAND_PRETTY}"
log "REMINDER:   Never pass secrets via CLI overrides. They WILL appear in logs."

# dry-run mode: show exactly what would be executed, then exit without touching logs/dirs:
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] not executing."
  exit 0
fi

###################################################
#     Prepare log dir and write header (if any)
###################################################
if [[ "${TEE_LOG}" != "0" ]]; then
  mkdir -p "${LOG_DIR}"
  # write launch header to log; keep minimal for provenance and reproducibility:
  {
    echo "===== $(date) : torchrun launch ====="
    echo "APP=${APP_MODULE}"
    echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "WORLD_SIZE=${WORLD_SIZE}"
    echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID-}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
    echo "NCCL_DEBUG=${NCCL_DEBUG}"
    echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG-}"
    echo "GIT_COMMIT=${GIT_COMMIT}${GIT_DIRTY}"
    # SECURITY NOTE:
    # COMMAND below includes all overrides after `--`.
    # DO NOT put secrets/tokens/passwords in CLI overrides.
    echo "COMMAND=${COMMAND_PRETTY}"
    echo "====================================="
  } >> "${LAUNCH_LOG}"
fi

###################################################
#         Signal handling and execution
###################################################
# Ensure any child torchrun workers are terminated if this script is interrupted.
# We SIGTERM the whole process group so no orphaned ranks linger on GPUs.
# NOTE: In the TEE_LOG!="1" branch where we 'exec', this trap won't persist (intentional).
trap 'echo "signal caught, terminating..."; kill -TERM -$$ 2>/dev/null || true' INT TERM

###################################################
#              Launch and log handling
###################################################
# If tee is enabled, stream stdout/stderr to console and append to the launcher log.
# console-only mode runs without touching the log file.
# any other TEE_LOG value -> "log-only": exec torchrun and redirect to file.
# IMPORTANT: we capture exit code from torchrun even when piped through tee.
if [[ "${TEE_LOG}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | tee -a "${LAUNCH_LOG}"
  exit_code=${PIPESTATUS[0]}   # exit code of "${CMD[@]}" side of the pipe
  exit "${exit_code}"
elif [[ "${TEE_LOG}" == "0" ]]; then
  "${CMD[@]}"
  exit $?
else
  exec "${CMD[@]}" >> "${LAUNCH_LOG}" 2>&1
fi
