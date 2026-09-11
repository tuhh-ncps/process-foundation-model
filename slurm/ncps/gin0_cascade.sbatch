#!/bin/bash
# Self-driving featureless-GIN (gin0) cascade: Phase 1 (role, no features) -> Phase 2 (backbone,
# frozen role) -> auto-submits the 4 Phase-3 label-efficiency evals. Mirrors run.sbatch container setup.
#SBATCH --job-name=gin0-cascade
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
REPO="$SLURM_SUBMIT_DIR"; cd "$REPO"
IMAGE="${IMAGE:-$HOME/hpc_training/pmfoundation.sif}"
DATA_ROOT="${DATA_ROOT:-$HOME/hpc_training/data}"
OUT="$REPO/outputs"; mkdir -p "$OUT"
run () { srun apptainer exec --nv \
  --bind "$REPO":/workspace --bind "$DATA_ROOT":/workspace/data --bind "$OUT":/workspace/outputs \
  --pwd /workspace --env PYTHONPATH=/workspace/src --env PROJECT_ROOT=/workspace \
  --env DATA_DIR=/workspace/data/raw --env OUTPUT_DIR=/workspace/outputs --env PYTHONUNBUFFERED=1 \
  --env WANDB_MODE=offline "$IMAGE" python train.py "$@"; }

echo "=== PHASE 1: featureless role embedding ==="
run task=role_pretrain role=frozen role.model.role_arch=gin role.model.role_feature_subset=none
ROLE=$(ls -1dt outputs/role_encoders/*/ | head -1 | xargs basename)
echo "[phase1] role run: $ROLE"
grep -o '"role_feature_subset":[^,]*' outputs/role_encoders/$ROLE/manifest.json || true

echo "=== PHASE 2: backbone reading the FROZEN featureless role ==="
run model=role_gin0 data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11] \
    role_init_from=$ROLE freeze_role=true tag=-frozen-gin0
BB=$(ls -1dt outputs/backbones/*frozen-gin0*/ | head -1 | xargs basename)
echo "[phase2] backbone: $BB"
test -f "outputs/backbones/$BB/backbone.pt" || { echo "NO backbone.pt - aborting phase 3"; exit 1; }

echo "=== PHASE 3: submit the 4 eval grids ==="
TASKS="[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
for EV in BPI12 BPI17 BPI20ID helpdesk; do
  EXTRA=""; [ "$EV" = "helpdesk" ] && EXTRA="evaluate.eval_log.path=/workspace/data/raw/helpdesk.csv"
  ARGS="task=evaluate evaluate=label_efficiency model=role_gin0 ~evaluate.backbones.ar +evaluate.backbones.gin0=$BB evaluate.eval_dataset=$EV evaluate.tasks=$TASKS $EXTRA" \
  USE_GPU=1 sbatch --job-name=p3-gin0-$EV --time=03:00:00 --gres=gpu:nvidia_h200_nvl_1g.18gb:1 --export=ALL slurm/ncps/run.sbatch
done
echo "=== gin0 cascade complete; evals queued ==="
