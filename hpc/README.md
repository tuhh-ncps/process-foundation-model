# Cluster orchestration scripts

Everything in this directory ran on the NCPS cluster from the working tree `~/hpc_training_frozen`,
which is a copy of this repository. The library code there is byte-identical to `src/` here; these
scripts are the only part that lived solely on the cluster until the 2026-09-11 code freeze.

**They produced every number reported in the ICPM paper**, so they are kept as the reproducibility
record rather than as maintained code.

## Layout

| Directory | What it holds |
|---|---|
| `submit/` | Slurm submitters: one per experiment grid (`submit_v2.py` is the main label-efficiency grid) |
| `collect/` | Result collectors that walk `outputs/label_efficiency/*/manifest.json` and emit one CSV |
| `bench/` | Wall-clock and feature-importance benchmarks built on cached frozen features |
| `analysis/` | Ad-hoc one-off checks written during the work; unmaintained |
| `slurm/` | The sbatch runners. The repo's top-level `slurm/` is gitignored, so they are duplicated here |

## How they were invoked

All paths inside these scripts are **relative to the repository root**, not to the script. On the
cluster they sat at the root of `~/hpc_training_frozen` and were run from there:

```bash
cd ~/hpc_training_frozen
python submit_v2.py main              # submit a grid
python collect_v2.py > results.csv    # collect it
```

To run them from this layout, invoke them from the repository root so the relative paths still
resolve:

```bash
python hpc/submit/submit_v2.py main
python hpc/collect/collect_v2.py > results.csv
```

`bench/bench_feat_importance.py` imports `bench_cached_pfm`, so those two must stay together.

The sbatch runners take the hydra overrides through the `ARGS` environment variable and a
`USE_GPU=1` flag, for example:

```bash
USE_GPU=1 ARGS="task=evaluate evaluate=label_efficiency ..." sbatch --gres=gpu:1 slurm/ncps/run.sbatch
```

## Provenance of the reported results

| Run tag | Script | What it produced |
|---|---|---|
| r14 | `submit/submit_v2.py` | v2 main grid and the backbone ablation |
| r19, r22 | `submit/submit_seeds.py`, `submit/submit_seedprobes*.py` | pretraining-seed replication |
| r23 | `submit/submit_rft.py` | PFM-RFT, role encoder fine-tuned with the backbone frozen |
| r16, r17, r24 | `submit/submit_timing*.py` | pinned wall-clock measurements |
| r25 | `bench/bench_cached_pfm.py` | cached-feature wall-clock |
| r26 | `submit/submit_linhead.py`, `collect/collect_linhead.py` | linear vs MLP regression heads |
| r27 | `bench/bench_feat_importance.py` | permutation importance of the 15 fingerprint features |
