#!/bin/bash
# Native (containerless) AR pretraining on oland (Uni Kiel) - single fat node, Slurm, uv env.
# oland has NO container runtime, so we run in the project's uv venv directly (built once with
# `uv sync --extra tracking`). GPUs are requested via Slurm gres and exposed as CUDA devices.
#   make submit_oland EXPERIMENT=role DATASET=mimic_transfers \
#       EXTRA='ar.outcome_labeler=mimic_mortality' TAG=mimic_role
#SBATCH --job-name=pmf-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    ##EDIT## any GPU; or gpu:H100:1 / gpu:L40:1 to pin a type
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00                 ##EDIT## walltime (partition normal=1d, long=7d)
#SBATCH --partition=normal              ##EDIT## normal | long
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"     # uv
# Live code wins over the installed wheel (rsync a code change -> effective without reinstall).
export PYTHONPATH="$REPO/src"
export PROJECT_ROOT="$REPO"
export DATA_DIR="$REPO/data/raw"
export OUTPUT_DIR="$REPO/outputs"

EXPERIMENT="${EXPERIMENT:-baseline}"
DATASET="${DATASET:-bpi12}"
LOGGER="${LOGGER:-wandb}"
EXTRA="${EXTRA:-}"

# W&B: oland has internet, so log ONLINE (no offline-sync step needed). The API key is read
# from ~/.netrc (`make login_native` once) or WANDB_API_KEY. LOGGER=none disables logging.
ONLINE=""
[ "$LOGGER" = "wandb" ] && ONLINE="logger.mode=online"
# DataLoader workers. Default 0 (in-process): the pretrain dataset is a large IN-MEMORY list of
# pydantic traces, so num_workers>0 forks a copy per worker and stalls startup - verified to hang
# on the 382k-trace MIMIC set. Only set WORKERS>0 for a lighter/streamed dataset.
WORKERS="${WORKERS:-0}"

# Multi-dataset training: DATASET may be a comma list (mimic_transfers,bpi12,bpi17) → ONE backbone
# trained on the concatenated 70% train split of each corpus. A single name keeps prior behaviour.
if [[ "$DATASET" == *,* ]]; then
    LIST="${DATASET// /}"                              # strip stray spaces
    DATA_ARG="data=multi ++data.datasets=[$LIST]"
    NAME_ARG="name=${LIST//,/+}-${EXPERIMENT}\${tag}"  # e.g. mimic_transfers+bpi12-role<TAG>
else
    DATA_ARG="data=$DATASET"
    NAME_ARG=""
fi

echo "[oland] pretrain experiment=$EXPERIMENT data=$DATASET logger=$LOGGER  gpu=$CUDA_VISIBLE_DEVICES  workers=$WORKERS"
srun uv run python train.py \
    experiment="$EXPERIMENT" $DATA_ARG trainer=local logger="$LOGGER" $NAME_ARG \
    "+data.num_workers=$WORKERS" $ONLINE $EXTRA
