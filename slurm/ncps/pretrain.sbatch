#!/bin/bash
# NCPS cluster: single-GPU AR pretraining, full HPC path (container -> data -> train -> artifacts).
# EDIT the ##EDIT## lines, then:  sbatch slurm/ncps/pretrain.sbatch  (or: make submit_ncps)
#
#SBATCH --job-name=pmf-pretrain-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200_nvl_1g.18gb:1
#SBATCH --cpus-per-task=8                ##EDIT## ~1 core actually used; 4 leaves DataLoader headroom
#SBATCH --mem=64G                        ##EDIT## multi-domain corpus peaks ~13G RAM (streaming reader); 32G is safe
#SBATCH --time=20:00:00                 ##EDIT## walltime - 15 epochs on the small 1g.18gb slice is slow
#SBATCH --partition=main                ##EDIT## GPU partition
#SBATCH --account=ncps-team             ##EDIT## allocation/account (sacctmgr show assoc user=$USER)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Self-contained: everything is referenced from the project dir you submit from ($REPO).
REPO="$SLURM_SUBMIT_DIR"                           # the hpc_training project on the cluster
IMAGE="${IMAGE:-$REPO/pmfoundation.sif}"           # container built into the project (make build)
DATA="${DATA:-$REPO/data/raw}"                     # bundled data  (override w/ DATA=/shared/... for a shared FS)
OUT="${OUT:-$REPO/outputs}"                        # run artifacts stay inside the project
EXPERIMENT="${EXPERIMENT:-role_rope}"              # Hydra experiment (configs/experiment/*.yaml)
DATASET="${DATASET:-bpi12}"                     # training data (configs/data/*): bpi12 | bpi12_bpi17
LOGGER="${LOGGER:-none}"                            # none | wandb  (W&B logs offline -> sync on login node)
EXTRA="${EXTRA:-}"                                # any extra Hydra overrides (e.g. model.position=rope)
mkdir -p "$OUT"

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

# Single GPU -> trainer=local (no DDP). Same code path as the multi-node run.
# WANDB_MODE=offline: compute nodes have no internet; runs land in outputs/wandb/ for later sync.
srun apptainer exec --nv \
    --bind "$REPO":/workspace \
    --bind "$DATA":/workspace/data/raw \
    --bind "$OUT":/workspace/outputs \
    --pwd /workspace \
    --env PYTHONPATH=/workspace/src \
    --env DATA_DIR=/workspace/data/raw \
    --env OUTPUT_DIR=/workspace/outputs \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/workspace/outputs \
    "$IMAGE" \
    python train.py experiment="$EXPERIMENT" $DATA_ARG trainer=local logger="$LOGGER" $NAME_ARG $EXTRA
