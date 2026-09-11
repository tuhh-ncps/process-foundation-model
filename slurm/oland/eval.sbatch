#!/bin/bash
# Native (containerless) downstream evaluation on oland. Covers all eval modes via $EVAL:
#   label_efficiency (single BACKBONE), compare / zero_shot (BACKBONES map).
#   make eval_oland EVAL=compare  BACKBONES='{random:random,role:<id>}' SEEDS=3 \
#       EXTRA='evaluate.eval_log.path=/home/trinh/hpc_training/data/raw/mimic_transfers.csv evaluate.outcome.labeler=mimic_mortality'
#   make eval_oland EVAL=zero_shot BACKBONES='{role:<id>}' \
#       EXTRA='evaluate.eval_log.path=/home/trinh/hpc_training/data/raw/mimic_transfers.csv'
#SBATCH --job-name=pmf-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00                 ##EDIT## walltime
#SBATCH --partition=normal              ##EDIT## normal | long
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ulimit -n 65536 2>/dev/null || true      # many DataLoaders on large corpora (Errno 24 guard)
mkdir -p logs
REPO="$SLURM_SUBMIT_DIR"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PROJECT_ROOT="$REPO"
export DATA_DIR="$REPO/data/raw"
export OUTPUT_DIR="$REPO/outputs"

EVAL="${EVAL:-compare}"                   # label_efficiency | compare | zero_shot
EXTRA="${EXTRA:-}"
SEED_OVERRIDE=""
[ -n "${SEEDS:-}" ] && { _s=$(seq -s, 0 $((SEEDS-1))); SEED_OVERRIDE="evaluate.seeds=[${_s%,}]"; }
DS_OVERRIDE=""
[ -n "${EVAL_DATASET:-}" ] && DS_OVERRIDE="evaluate.eval_dataset=$EVAL_DATASET"

# label_efficiency takes a single BACKBONE; compare/zero_shot take a BACKBONES map.
BB_OVERRIDE=""
if [ -n "${BACKBONES:-}" ]; then
    BB_OVERRIDE="++evaluate.backbones=$BACKBONES"
elif [ -n "${BACKBONE:-}" ]; then
    BB_OVERRIDE="evaluate.backbones.ar=$BACKBONE"
else
    echo "set BACKBONES='{alias:id,...}' (compare/zero_shot) or BACKBONE=<id> (label_efficiency)"; exit 1
fi

echo "[oland] evaluate=$EVAL  $BB_OVERRIDE  gpu=$CUDA_VISIBLE_DEVICES"
srun uv run python train.py task=evaluate evaluate="$EVAL" \
    "$BB_OVERRIDE" $SEED_OVERRIDE $DS_OVERRIDE $EXTRA
