#!/bin/bash
# Generic container job: runs  $CMD  inside the pmfoundation image with the same binds as run.sbatch.
#SBATCH --job-name=pmf-cmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=main
#SBATCH --account=ncps-team
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p logs
REPO="$SLURM_SUBMIT_DIR"
IMAGE="${IMAGE:-$HOME/hpc_training/pmfoundation.sif}"
DATA_ROOT="${DATA_ROOT:-$HOME/hpc_training/data}"
OUT="${OUT:-$REPO/outputs}"
NV=""; [ "${USE_GPU:-0}" = "1" ] && NV="--nv"
echo "[run_cmd.sbatch] CMD=$CMD  USE_GPU=${USE_GPU:-0}  node=$(hostname)"
srun apptainer exec $NV \
  --bind "$REPO":/workspace --bind "$DATA_ROOT":/workspace/data --bind "$OUT":/workspace/outputs \
  --pwd /workspace --env PYTHONPATH=/workspace/src --env PROJECT_ROOT=/workspace --env PYTHONUNBUFFERED=1 \
  "$IMAGE" bash -c "$CMD"
