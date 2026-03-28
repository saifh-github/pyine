#!/usr/bin/env bash
# run_probe_sweep.sh: launches probe training jobs sequentially on specific GPUs
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DDP="${SCRIPT_DIR}/run_ddp.sh"

JOBS=(
    "--app pyine.apps.trainers.probe_trainer --nproc_per_node 4 -- +experiment=probes/v0_probe_moderate"
    "--app pyine.apps.trainers.probe_trainer --nproc_per_node 4 -- +experiment=probes/v0_probe_strong"
    "--app pyine.apps.trainers.probe_trainer --nproc_per_node 4 -- +experiment=probes/v0_probe_weak"
    # "--app pyine.apps.trainers.probe_trainer --nproc_per_node 2 -- +experiment=probes/TODO_job4"
)

for job_idx in "${!JOBS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Job $((job_idx + 1))/${#JOBS[@]}"
    echo "========================================"
    echo ""
    # shellcheck disable=SC2086
    bash "${RUN_DDP}" ${JOBS[$job_idx]}
    job_exit=$?
    echo ""
    echo "[sweep] Job $((job_idx + 1))/${#JOBS[@]} finished with exit code ${job_exit}."
done

echo ""
echo "[sweep] All ${#JOBS[@]} jobs complete."
