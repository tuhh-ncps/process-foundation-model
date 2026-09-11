#!/bin/bash
# TUHH cluster: multi-node, multi-GPU AR pretraining under Lightning DDP inside Apptainer.
# One task per GPU. EDIT the ##EDIT## lines, then:  sbatch slurm/tuhh/pretrain.sbatch  (or: make submit_tuhh)
#
#SBATCH --job-name=pmf-pretrain
#SBATCH --nodes=2                      # 2 nodes
#SBATCH --ntasks-per-node=1            # 1 GPU per node -> 1 task per node (world size = 2)
#SBATCH --gres=gpu:1                   # 1 GPU per node
#SBATCH --cpus-per-task=16              ##EDIT## dataloader workers headroom (<= cores/node)
#SBATCH --mem=16G                        # all memory on the node (or set e.g. 64G)
#SBATCH --time=07:00:00                ##EDIT## walltime
#SBATCH --partition=gpu                ##EDIT## GPU partition (from `sinfo -o "%P %G %l"`)
#SBATCH --account=ctt3141               ##EDIT## allocation/account
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# ---- paths: self-contained, everything referenced from the project dir you submit from ($REPO) ----
REPO="$SLURM_SUBMIT_DIR"                        # the hpc_training project on the cluster
IMAGE="${IMAGE:-$REPO/pmfoundation.sif}"        # container built into the project (make build)
DATA="${DATA:-$REPO/data/raw}"                  # bundled BPI12.xes / BPI17.xes (override w/ DATA=/shared/... for a shared FS)
OUT="${OUT:-$REPO/outputs}"                     # run artifacts stay inside the project
EXPERIMENT="${EXPERIMENT:-role_rope}"           # Hydra experiment (configs/experiment/*.yaml)
DATASET="${DATASET:-bpi12}"                     # training data (configs/data/*): bpi12 | bpi12_bpi17
LOGGER="${LOGGER:-none}"                         # none | wandb  (W&B logs offline -> sync on login node)
EXTRA="${EXTRA:-}"                                # any extra Hydra overrides (e.g. model.position=rope)
mkdir -p "$OUT"

# ---- DDP rendezvous (Lightning reads these under srun) ----
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=ib0        ##EDIT## if your cluster needs a specific interface

echo "job $SLURM_JOB_ID | nodes=$SLURM_NNODES tasks/node=$SLURM_NTASKS_PER_NODE | master=$MASTER_ADDR"

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

# One process per GPU. --nv exposes the GPUs; binds put staged data + live code + outputs in place.
# train.py reads the DDP topology (devices/nodes/world) straight from the Slurm env vars.
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
    python train.py experiment="$EXPERIMENT" $DATA_ARG trainer=ddp logger="$LOGGER" $NAME_ARG $EXTRA
