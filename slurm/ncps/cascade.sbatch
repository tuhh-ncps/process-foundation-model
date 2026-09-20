#!/bin/bash
# Generic frozen-cascade runner. Env:
#   CAND            short alias (backbone tag + eval alias), e.g. fusedcand
#   MODEL           configs/model name for Phase 2/3, e.g. role_fused_cand
#   ROLE_ID         reuse an existing Phase-1 role run (skips Phase 1) - optional
#   ROLE_OVERRIDES  Phase-1 hydra overrides (used when ROLE_ID is empty)
#   P2_EXTRA        extra Phase-2 overrides, e.g. ar.jepa_weight=1.0
# Runs Phase 1 (unless reusing) -> Phase 2 -> submits the 4 Phase-3 eval grids.
#SBATCH --job-name=cascade
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

if [ -n "${ROLE_ID:-}" ]; then
  ROLE="$ROLE_ID"; echo "=== PHASE 1 SKIPPED - reusing role: $ROLE ==="
  test -f "outputs/role_encoders/$ROLE/role_encoder.pt" || { echo "missing role_encoder.pt"; exit 1; }
else
  echo "=== PHASE 1: role embedding ($CAND) ==="
  run task=role_pretrain role=frozen ${ROLE_OVERRIDES:-}
  # Pick the newest role run MATCHING this variant (not just newest) - other cascades may run
  # concurrently and would otherwise be picked up by a bare newest-dir lookup.
  ROLE=""
  for d in $(ls -1dt outputs/role_encoders/*/); do
    if grep -q "${ROLE_GREP:-role_arch}" "$d/manifest.json" 2>/dev/null; then ROLE=$(basename "$d"); break; fi
  done
  test -n "$ROLE" || { echo "no role run matched ${ROLE_GREP:-}"; exit 1; }
  echo "[phase1] role run: $ROLE"
fi

echo "=== PHASE 2: backbone ($CAND) with FROZEN role ==="
run model=$MODEL data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11] \
    role_init_from=$ROLE freeze_role=true tag=-frozen-$CAND ${P2_EXTRA:-}
BB=$(ls -1dt outputs/backbones/*frozen-$CAND-*/ | head -1 | xargs basename)
echo "[phase2] backbone: $BB"
test -f "outputs/backbones/$BB/backbone.pt" || { echo "NO backbone.pt - aborting phase 3"; exit 1; }

echo "=== PHASE 3: submitting 4 eval grids ==="
TASKS="[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
for EV in BPI12 BPI17 BPI20ID helpdesk; do
  EXTRA=""; [ "$EV" = "helpdesk" ] && EXTRA="evaluate.eval_log.path=/workspace/data/raw/helpdesk.csv"
  ARGS="task=evaluate evaluate=label_efficiency model=$MODEL ~evaluate.backbones.ar +evaluate.backbones.$CAND=$BB evaluate.eval_dataset=$EV evaluate.tasks=$TASKS $EXTRA" \
  USE_GPU=1 sbatch --job-name=p3-$CAND-$EV --time=03:00:00 --gres=gpu:nvidia_h200_nvl_1g.18gb:1 --export=ALL slurm/ncps/run.sbatch
done
echo "=== cascade $CAND complete; evals queued ==="
