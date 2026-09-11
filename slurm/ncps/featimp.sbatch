#!/bin/bash
# Generic frozen-cascade job: runs  python train.py $ARGS  in the container.
# Binds the PARENT data/ dir so ../very-raw symlinks (BPI18/19/11/HospitalBilling/RoadTraffic) resolve.
#SBATCH --job-name=pmf-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --account=ncps-team
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p logs
REPO="$SLURM_SUBMIT_DIR"
IMAGE="${IMAGE:-$HOME/hpc_training/pmfoundation.sif}"
DATA_ROOT="${DATA_ROOT:-$HOME/hpc_training/data}"    # parent (has raw/ + very-raw/ symlink targets)
OUT="${OUT:-$REPO/outputs}"
mkdir -p "$OUT"
NV=""; [ "${USE_GPU:-0}" = "1" ] && NV="--nv"
echo "[bench.sbatch] ARGS=$ARGS  USE_GPU=${USE_GPU:-0}  node=$(hostname)"
srun apptainer exec $NV \
  --bind "$REPO":/workspace --bind "$DATA_ROOT":/workspace/data --bind "$OUT":/workspace/outputs \
  --pwd /workspace --env PYTHONPATH=/workspace/src --env PROJECT_ROOT=/workspace \
  --env DATA_DIR=/workspace/data/raw --env OUTPUT_DIR=/workspace/outputs --env PYTHONUNBUFFERED=1 --env WANDB_MODE=offline \
  "$IMAGE" python bench_feat_importance.py $ARGS
