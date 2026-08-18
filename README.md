# HPC training (self-contained, Hydra-driven)

> This folder is a **standalone copy** of the PM-Foundation project, adapted for the HPC
> cluster. It bundles its own copy of the `pm_foundation` package (`src/`) plus the container,
> Slurm scripts, and a full Hydra config tree — so it deploys and runs **independently** of the
> local checkout. Nothing here touches the local training code. Trade-off: it is a copy — sync
> model/data improvements from the main project when you want them. Stage the event logs from
> the main project's `data/raw/` (see below).

The same idea as the local project, made portable: **where it runs (laptop / 1 GPU / 2 nodes)
is a config choice, not a code fork.** One entrypoint (`train.py`), one Hydra tree, driving the
**real** pm_foundation flows.

```bash
# local — AR pretrain on CPU/MPS/1 GPU (BPI12 is the default dataset)
python train.py experiment=role_rope trainer=local

# cluster — SAME code; only trainer (device/strategy) + where-it-runs change
make submit_tuhh                                    # 2 nodes x 1 GPU, DDP, inside Apptainer (make submit_ncps = 1 GPU)
```

## Layout

```
hpc_training/
├── train.py                  # the ONE Hydra entrypoint — dispatches task=pretrain | evaluate
├── configs/                  # Hydra tree — compose a run from swappable groups
│   ├── train.yaml            #   root defaults + task selector
│   ├── paths/default.yaml    #   data_dir / output_dir  (ENV-driven — never hardcoded)
│   ├── data/                 #   bpi12.yaml, bpi12_bpi17.yaml   (which logs, vocab, batching)
│   ├── model/                #   transformer_small.yaml, transformer_large.yaml
│   ├── ar/                   #   default.yaml   (objective, loss weight, optimizer)
│   ├── trainer/              #   local | ddp | fsdp   (device + strategy)
│   ├── evaluate/             #   label_efficiency.yaml   (downstream probe)
│   ├── experiment/           #   role_rope.yaml, no_role.yaml   (architecture; one word = a full run)
│   └── hydra/launcher/       #   slurm.yaml   (submitit — submit from Python, no sbatch)
├── src/pm_foundation/        # bundled copy of the package (DDP-safe pretrain + sampler)
├── containers/pmfoundation.def   # Apptainer image (build once on a login node)
├── slurm/{ncps,tuhh,oland}/*.sbatch   # per-cluster pretrain + eval jobs (robust with Apptainer)
└── tests/test_samplers.py    # DDP-sharding sampler test
```

**What `train.py` does:** composes the Hydra config into the exact dict `pm_foundation`
expects, fills the DDP topology (`devices`/`num_nodes`/`strategy`) from the **Slurm environment**,
then calls the real flow — `pretrain_autoregressive_ddp` (in
`src/pm_foundation/training/ar_pretrain_hpc.py`, kept separate from the untouched local
`pretrain_autoregressive`) for `task=pretrain`, or `run_label_efficiency` for `task=evaluate`.
One process per GPU; only rank 0 prints and writes artifacts.

## 1. One-time setup on the login node

Build the container (needs internet; compute nodes run it offline):
```bash
cd hpc_training
apptainer build $SCRATCH/pmfoundation.sif containers/pmfoundation.def
```

Stage the data to a **shared** fast filesystem (all nodes must read it — never node-local `/tmp`):
```bash
mkdir -p $SCRATCH/pm_foundation/data/raw
rsync -av /path/to/main/data/raw/BPI12.xes /path/to/main/data/raw/BPI17.xes \
    $SCRATCH/pm_foundation/data/raw/
```

`configs/paths/default.yaml` reads `DATA_DIR` / `OUTPUT_DIR` from the environment, so nothing is
hardcoded. The sbatch scripts export them into the container for you.

## 2. Run

### Quick single-GPU test (proves container → data → train → artifacts)
```bash
# edit the ##EDIT## lines (partition, account) in slurm/ncps/pretrain.sbatch once, then:
make submit_ncps                          # or: sbatch slurm/ncps/pretrain.sbatch
```

### Multi-node DDP (2 nodes × 1 GPU)
```bash
make submit_tuhh                          # or: sbatch slurm/tuhh/pretrain.sbatch
# multi-log pretraining — pass a comma-separated DATASET (data=multi under the hood):
make submit_ncps DATASET=bpi12,bpi17
```
`train.py` reads `SLURM_NNODES` / `SLURM_NTASKS_PER_NODE` / `SLURM_NTASKS` and sets
`devices=GPUs/node`, `num_nodes=nodes`, `strategy=ddp` automatically — no numbers to hand-edit in
Python. Batch size in the data config is **per GPU**; global batch = `batch_size × world_size`
(scale LR accordingly).

### Downstream label-efficiency evaluation
```bash
# after a pretraining run, take its run-id from $OUTPUT_DIR/backbones/<id>/
python train.py task=evaluate evaluate=label_efficiency evaluate.backbones.ar=<backbone_run_id>
```

## Two clusters (central + institute)

Same container and code; only the launcher/partition differs. Either submit each cluster's sbatch
from that cluster's login node, or pass the partition/account on the CLI. Each cluster has its own
filesystem, so **stage the data and build the `.sif` once per cluster**.

## Submitting from Python instead of sbatch (optional)

`configs/hydra/launcher/slurm.yaml` wires the **submitit** launcher — submit the same run without
a hand-written sbatch:
```bash
pip install '.[hpc]'    # login-node env: adds hydra-submitit-launcher
python train.py -m experiment=role_rope trainer=ddp hydra/launcher=slurm \
    hydra.launcher.partition=gpu hydra.launcher.nodes=2 hydra.launcher.gpus_per_node=1
```
Note: submitit launches into the **login-node Python env**. On this air-gapped + Apptainer setup
the **sbatch scripts are the robust path** (they run `train.py` *inside* the container); use
submitit only if compute nodes can see the same installed env. The sbatch route is recommended.

## Scaling to a bigger model — change the `trainer`, not the code

A model too big for one GPU needs **sharding** (plain DDP replicates the whole model per GPU):
```bash
# via submitit (or set model=transformer_large + trainer=fsdp in an sbatch):
python train.py -m model=transformer_large trainer=fsdp hydra/launcher=slurm \
    hydra.launcher.nodes=4 hydra.launcher.gpus_per_node=8
```
`trainer=fsdp` shards params/grads/optimizer (Lightning's FSDP; or `strategy: deepspeed_stage_3`).
Add activation checkpointing, `accumulate_grad_batches`, and ≥80 GB GPUs on InfiniBand for the
truly large regime. Entrypoint, launcher, and container are unchanged.

## Keeping in sync with the main project

This is a copy. When the local `pm_foundation` improves, re-sync `src/pm_foundation/` (the DDP
pretrain lives only here in `ar_pretrain_hpc.py`; the rest mirrors main). The Hydra configs here
track the real config schema — update them if the model/AR/data schema changes upstream.
