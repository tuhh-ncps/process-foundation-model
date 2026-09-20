#!/bin/bash
# Containerized (Apptainer) downstream evaluation on ncps. Covers all eval modes via $EVAL:
#   label_efficiency (single BACKBONE), compare / zero_shot (BACKBONES map).
#   make eval_ncps EVAL=compare          BACKBONES='{random:random,role:<id>}' SEEDS=3 EXTRA='...'
#   make eval_ncps EVAL=zero_shot        BACKBONES='{role:<id>}' EXTRA='...'
#   make eval_ncps EVAL=label_efficiency BACKBONE=<id>
#SBATCH --job-name=pmf-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl_1g.18gb:1
#SBATCH --cpus-per-task=8                ##EDIT## eval is light (num_workers=0); 4 cores is plenty
#SBATCH --mem=64G
#SBATCH --time=12:00:00                 ##EDIT## walltime - full event_tasks grid is long; runs resume from checkpoint on re-submit
#SBATCH --partition=main                ##EDIT## GPU partition
#SBATCH --account=ncps-team             ##EDIT## allocation/account (sacctmgr show assoc user=$USER)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ulimit -n 65536 2>/dev/null || true      # many DataLoaders on large corpora (Errno 24 guard)
mkdir -p logs

REPO="$SLURM_SUBMIT_DIR"
IMAGE="${IMAGE:-$REPO/pmfoundation.sif}"
DATA="${DATA:-$REPO/data/raw}"
OUT="${OUT:-$REPO/outputs}"
mkdir -p "$OUT"

EVAL="${EVAL:-compare}"                    # label_efficiency | compare | zero_shot
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

srun apptainer exec --nv \
    --bind "$REPO":/workspace --bind "$DATA":/workspace/data/raw --bind "$OUT":/workspace/outputs \
    --pwd /workspace \
    --env PYTHONPATH=/workspace/src --env DATA_DIR=/workspace/data/raw --env OUTPUT_DIR=/workspace/outputs \
    "$IMAGE" \
    python train.py task=evaluate evaluate="$EVAL" "$BB_OVERRIDE" $SEED_OVERRIDE $DS_OVERRIDE $EXTRA
