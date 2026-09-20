# Reproducing the experiments

Every number, table and figure in the paper comes from this repository. This guide goes from a fresh
clone to regenerated figures.

**Contents:** [Setup](#1-setup) · [Data](#2-data) · [Phase 1a](#3-phase-1a--role-encoder) ·
[Phase 1b](#4-phase-1b--backbone) · [Evaluation](#5-evaluation-grids) ·
[Collecting results](#6-collecting-results) · [Figures](#7-figures-and-tables) ·
[Paper map](#8-which-command-produced-which-result) · [Costs](#9-compute-cost) ·
[Deviations](#known-deviations)

---

## 1. Setup

```bash
git clone <this repo> && cd hpc_training
uv sync                     # or: python -m venv .venv && pip install -e .
```

Python 3.11, PyTorch 2.x, Lightning 2.x. **An accelerator is assumed**: a CUDA GPU, or Apple silicon
(M-series) through PyTorch MPS. `trainer=local` sets `accelerator: auto`, which picks CUDA or MPS
on its own, so nothing needs changing per machine. CPU-only runs are not supported: pretraining is
impractical and the probes are far too slow to be useful.

The figures in section 7 are the exception - they read the committed CSVs and need no accelerator.

Everything is driven by one Hydra entrypoint:

```bash
python train.py task=<pretrain|role_pretrain|evaluate> [overrides...]
```

Paths come from `configs/paths/default.yaml` and honour `DATA_DIR` and `OUTPUT_DIR`. Nothing is
hard-coded.

### On a cluster

Every run is one `train.py` invocation with Hydra overrides, so any scheduler works: submit the
same command however your site does it. The cluster submitters we used are internal and not part
of this release; the commands below are exactly what they pass to `train.py`.

---

## 2. Data

All logs go **directly in `data/raw/`**, flat, under the exact names below. The code never reads the
original download filenames, so renaming is required, not optional. This is the step people get
wrong, so there is a checker:

```bash
python scripts/check_data.py       # names any file that is missing or misnamed, and how to fix it
```

Ten logs are public and come from [4TU.ResearchData](https://data.4tu.nl/). Search the **exact
title** in the middle column; the download contains the file in the third column, which you rename.

| Rename to | 4TU dataset title | File inside the download | Used for |
|---|---|---|---|
| `BPI12.xes` | BPI Challenge 2012 | `BPI_Challenge_2012.xes` | pretrain + in-domain eval |
| `BPI18.xes` | BPI Challenge 2018 | `BPI Challenge 2018.xes` | pretrain |
| `BPI19.xes` | BPI Challenge 2019 | `BPI_Challenge_2019.xes` | pretrain |
| `RoadTraffic.xes` | Road Traffic Fine Management Process | `Road_Traffic_Fine_Management_Process.xes` | pretrain |
| `BPI11.xes` | Real-life event logs - Hospital log | `Hospital_log.xes` | pretrain |
| `HospitalBilling.xes` | Hospital Billing - Event Log | `Hospital Billing - Event Log.xes` | pretrain |
| `BPI17.xes` | BPI Challenge 2017 | `BPI Challenge 2017.xes` | held-out eval |
| `BPI20ID.xes` | BPI Challenge 2020: International Declarations | `InternationalDeclarations.xes` | held-out eval |
| `BPI13.xes` | BPI Challenge 2013, incidents | `BPI_Challenge_2013_incidents.xes` | held-out eval |
| `helpdesk.csv` | Helpdesk | `helpdesk.csv` | held-out eval |
| `SepsisCases_Event_Log.xes` | Sepsis Cases - Event Log | `Sepsis Cases - Event Log.xes` | Phase 1a validation only |
| `berti_receipt.xes` | Receipt phase of an environmental permit application | `Receipt phase of an environmental permit application process.xes` | Phase 1a validation only |

Two direct links, as a sanity check that you have the right datasets:
[BPI Challenge 2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204) ·
[BPI Challenge 2017](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884).

`helpdesk.csv` must have the header `case_id,activity,timestamp`. Some mirrors ship a wider CSV or
an XES; only those three columns are read.

Expect roughly **3.9 GB** once unpacked; BPI18 alone is 1.9 GB.

### MIMIC

MIMIC-IV is not redistributable and needs credentialed PhysioNet access, so it cannot be scripted for
you. With access:

```bash
python scripts/build_mimic_log.py \
  --mimic data/raw/mimic-iv-v3.1/physionet.org/files/mimiciv/3.1 \
  --out   data/raw/mimic_transfers.csv
```

Each admission becomes a trace of care-unit transfers with an outcome terminal at discharge; the
paper uses a fixed subset of 5,000 admissions. **Without MIMIC everything else still runs** - you
reproduce four of the five held-out logs, and `check_data.py` exits 0 to say so.

### Verify before spending GPU time

```bash
python scripts/check_data.py --content   # recomputes statistics from your copies
git diff results/log_stats.csv           # no diff means they are equivalent to ours
```

`results/log_stats.csv` is committed, so this catches a truncated or wrong-variant download in
minutes instead of after a training run.

## 3. Phase 1a - role encoder

Trains the vocabulary-free activity encoder on its own, selecting the checkpoint on **held-out** logs
(Sepsis and Receipt) that are never trained on.

```bash
python train.py task=role_pretrain role=frozen trainer=local
```

`role=frozen` is the configuration the paper's encoder uses: it trains on the six pretraining logs
and validates on Sepsis and Receipt. (`role=default` is an older variant that validates on two of
the held-out evaluation logs, so do not use it for a paper run.)

120 epochs, Adam at 1e-3, loss weights `(w_v, w_s, w_c) = (0.5, 0.5, 1.0)`, `τ_s = 0.2`. Writes
`outputs/role_encoders/<run_id>/role_encoder.pt`.

The paper's encoder is `role-encoder-20260906-151732-role-frozen-bd3b8c`.

## 4. Phase 1b - backbone

Pretrains the causal backbone on the six-log corpus, on top of a frozen Phase 1a encoder.

```bash
python train.py task=pretrain \
  model=role_gin15 \
  data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11] \
  freeze_role=true model.id_dropout=1.0 \
  ar.time_weight=1.0 ar.remaining_time_weight=1.0 ar.jepa_weight=1.0 \
  role_init_from=<phase-1a-run-id> tag=-v2-gin15
```

`model.id_dropout=1.0` is what makes the model vocabulary-free: activity IDs are dropped with
probability 1, so only role embeddings and time carry information. 15 epochs, AdamW at 5e-4, weight
decay 0.01, batch 96, gradient clipping 1.0.

All seven ablation variants at once:

```bash
python train.py task=pretrain model=role_gin15 \
    data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11] \
    freeze_role=true ar.time_weight=1.0 ar.remaining_time_weight=1.0 model.id_dropout=1.0 \
    ar.jepa_weight=1.0 role_init_from=<role-encoder run id> tag=-v2-gin15 seed=0

# the six ablation backbones: model=role_mlp15 | role_raw15 | role_gin11 | role_gin0 (tag -v2-<name>),
# latent0 = role_gin15 with ar.jepa_weight=0, norole = role_gin15 with model.role_dim=0
```

The paper's backbone is `backbone-20260906-153102-multi-none-v2-gin15-17fc3c`.

## 5. Evaluation grids

The main grid: three arms × five held-out logs × eight label budgets × three seeds × seven tasks.

Every cell is one `train.py` run; the arms differ only in their backbone/finetune overrides:

| Arm | Overrides |
|---|---|
| PFM (frozen) | `+evaluate.backbones.pfm=<backbone run id>` |
| PFM-FT | `+evaluate.backbones.pfm_ft=<id>` `+evaluate.finetune=[pfm_ft]` |
| Frozen Random | `+evaluate.backbones.random_role=random_role` |
| PFM-Scratch | `+evaluate.backbones.pfm_scratch=random_role` `+evaluate.finetune=[pfm_scratch]` |

MIMIC adds `+evaluate.eval_log.max_traces=5000`; the ablation grid uses
`evaluate.label_sizes=[null]` (full budget only).

A single cell, if you want to run one by hand:

```bash
python train.py task=evaluate evaluate=label_efficiency model=role_gin15 \
  ~evaluate.backbones.ar ~evaluate.backbones.random \
  +evaluate.max_trace_len=64 evaluate.role_corpus=budget \
  evaluate.probe.max_epochs=100 evaluate.probe.early_stop_patience=10 \
  evaluate.tasks=[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set] \
  evaluate.eval_dataset=helpdesk evaluate.eval_log.path=data/raw/helpdesk.csv \
  +evaluate.backbones.pfm=<backbone-run-id> \
  evaluate.label_sizes=[0,10,30,100,300,1000,5000,null] evaluate.seeds=[0]
```

Protocol knobs that matter, all leak-relevant:

| Override | Meaning |
|---|---|
| `max_trace_len=64` | cases longer than 64 events are excluded, not truncated |
| `role_corpus=budget` | the DFG and fingerprints are built **only** from the sampled labelled cases |
| `label_sizes` | `0` is not zero-shot for MLP heads; exclude it from budget curves |
| `probe.head_hidden` | `128` gives regression heads an MLP; `0` makes them linear |
| `finetune=[alias]` | trains that arm end to end (this is PFM-FT) |
| `finetune_role=[alias]` | trains **only** the role encoder with the head, backbone frozen |
| `backbones.<a>=random_role` + `finetune=[<a>]` | PFM's architecture from random init, trained end to end (PFM-Scratch) |

Other grids:

```bash
# probe and backbone variants are the same command with one override changed:
#   linear regression heads       evaluate.probe.head_hidden=0
#   role-encoder-only finetuning  +evaluate.finetune_role=[pfm_rft]
#   seed-2 backbone               +evaluate.backbones.pfm_s2=<seed-2 backbone id>
python scripts/bench_cached_pfm.py <log>            # cached-feature wall-clock
python scripts/bench_feat_importance.py <log>       # fingerprint permutation importance
```

### Baselines

```bash
# one call per log; --max-traces mirrors the eval config (MIMIC: 5000)
python scripts/export_splits.py  --log helpdesk --path data/raw/helpdesk.csv --out exports/helpdesk_splits.csv
python scripts/export_queries.py --log helpdesk --path data/raw/helpdesk.csv --max-trace-len 64 \
    --splits-csv exports/helpdesk_splits.csv --out exports/helpdesk_queries.csv
# SuTraN: build its tensors from our splits, train it in its own repository, then score with our adapter
python scripts/sutran_build.py   --splits exports/helpdesk_splits.csv --queries exports/helpdesk_queries.csv \
    --out <SuTraN_Plus>/HELPDESK --log HELPDESK --window 64
python scripts/sutran_metrics.py --results <SuTraN_Plus>/HELPDESK/.../TEST_SET_RESULTS \
    --data <SuTraN_Plus>/HELPDESK --log-prefix HELPDESK --splits exports/helpdesk_splits.csv \
    --log helpdesk --seed 1 --out results/baselines/sutran_v2/helpdesk_s1.csv

# FM-v2: retrieval evaluation against its released checkpoint
python scripts/fmv2_eval.py --repo <events-transf> --checkpoint-dir <fmv2 checkpoints> \
    --splits exports/helpdesk_splits.csv --queries exports/helpdesk_queries.csv --log helpdesk \
    --tasks next_activity,remaining_time --k 5 10 20 --out results/baselines/fmv2_v2/helpdesk.csv
```

Both baselines are scored on the **same exported prefixes** as PFM, which is what makes the
comparison fair.

## 5b. Feature-budget ladder (pre-registered)

`protocols/feature_ladder.md` freezes the protocol before the runs; amendments A1 and A2 are appended
at the end of that file, and `protocols/feature_ladder_waivers.json` records the waived C2 check.

```bash
python scripts/feature_ladder.py --data-dir data/raw \
    --out results/feature_ladder.json --fingerprints results/feature_ladder_fingerprints.npz
python scripts/feature_ladder.py --selftest          # synthetic checks, no data needed
# phases B and C are 15 role encoders + 15 backbones + 240 evaluation runs: same commands as above
# with model.role_feature_mask=<mask from results/feature_ladder.json> and tag=-v2-fb<k>,
# then scripts/bench_cached_pfm.py --backbone <fb-k backbone> --tasks next_activity per (log, seed)
python scripts/feature_ladder_analysis.py c2         # -> results/feature_ladder_c2.json
python scripts/feature_ladder_analysis.py report     # -> results/feature_ladder_summary.{csv,json}
python scripts/feature_ladder_tasks_analysis.py      # -> results/feature_ladder_tasks_summary.{csv,json}
```

Phase A is CPU-only and reads the six pretraining logs; phases B and C are 270 GPU runs. The frozen
order is `CHFNJBMLDOKGIEA`, and the C2 gate stands at `PASSED_WITH_WAIVER` (A1: the cached evaluator
disagrees with the standard path by up to 0.043 accuracy, so ladder budgets are compared only with
each other, never with the main result tables).

## 6. Collecting results

Collectors walk `outputs/label_efficiency/*/manifest.json`, keep only runs matching the protocol, and
emit one tidy CSV:

These **overwrite the committed CSVs**, so run them only once `outputs/` holds your own runs.
Collect to a temporary file and move it into place, so a run that collects nothing cannot truncate
the shipped results (the collectors exit non-zero and print nothing when they match no run):

```bash
python scripts/collect_v2.py      > /tmp/v2_all.csv      && mv /tmp/v2_all.csv      results/v2_all.csv
python scripts/collect_linhead.py > /tmp/linhead_all.csv && mv /tmp/linhead_all.csv results/linhead_all.csv
```

Columns: `log, arm, task, n_labels, n_train_samples, seed, value, run`. One row per
(log, arm, task, budget, seed). Read a full-budget number like this:

```python
import pandas as pd
d = pd.read_csv("results/v2_all.csv")
d = d[(d.n_labels.astype(str) == "all") & (d.task == "next_activity")]
print(d.groupby(["log", "arm"]).value.mean().unstack("arm"))
```

Arms: `pfm` frozen, `pfm_ft` fine-tuned, `pfm_scratch` the same architecture trained end to end from
random init, `pfm_rft` role encoder only, `random_role` the floor, plus the ablation variants
`mlp15 raw15 gin11 gin0 latent0 norole` and the seed replicas `gin15_s1 gin15_s2 latent0_s1
latent0_s2 pfm_s2 pfm_ft_s2` (pretraining seed 1/2 of the same backbone).

## 7. Figures and tables

Every script in `figures/` reads only from `results/`, so these run on a fresh clone with **no GPU
and no event logs**. Run them from the repository root:

```bash
python figures/frozen_agg_merged.py       # Figure 3, label-efficiency panels (seed-0 backbone)
python figures/frozen_agg_merged.py --seed2   # same panels from the seed-2 backbone (needs r28 collected)
python figures/sota_agg_plot.py           # writes results/sota_agg_data.csv, then
python figures/sota_wall_plot.py          # Figure 4, baselines + adaptation cost
python figures/make_table_ablation_v2.py  # Table 6 and the seed-replication table
python figures/make_datasets_table.py     # dataset statistics table (needs results/log_stats.csv)
python figures/feats_importance_plot.py   # fingerprint non-redundancy
python figures/frozen_aggregate_panels.py # the same curves as seven standalone panels + a LaTeX block
PFM_SEED=2 python figures/frozen_aggregate_panels.py   # those panels from the seed-2 backbone
python figures/feature_ladder_plot.py     # feature-budget ladder, next activity (Phase D)
python figures/feature_ladder_tasks_plot.py           # ladder, six further tasks (amendment A2)
python figures/feature_ladder_reconstruction_plot.py  # reconstruction vs downstream, one file per task
python figures/feature_ladder_descriptor_heatmap.py   # per-descriptor R^2 vs budget
```

Output lands next to the scripts and is gitignored; the committed copies are in `assets/`.

This is the cheapest way to check our numbers: regenerate a figure and compare it against the one in
`assets/`. Those committed copies were produced by these exact scripts in the environment `uv.lock`
pins, so inside that environment they come out byte-identical. **Across matplotlib versions they will
not** - canvas dimensions shift by a few pixels - so compare what the figure *says*, not its
checksum. The numbers themselves come from `results/` and are version-independent.

Three of the eight images in `assets/` are regenerated this way (`frozen_agg_all`, `sota_wall`,
`feats_importance`). The architecture diagrams and the role-space t-SNE are built from the LaTeX and
TikZ sources under `docs/`, which is not published.

> The role-space t-SNE (`gin15_seen_unseen.py`) is **not** in `figures/`, because it needs the raw
> logs and the Phase 1a encoder rather than a results CSV. It stays with the LaTeX sources under
> `docs/`, which `.gitignore` excludes.

## 8. Which command produced which result

| Paper element | Produced by |
|---|---|
| Dataset statistics table | `scripts/log_stats.py` (needs `data/raw/`) → `make_datasets_table.py` |
| Table 5, full-budget results | `submit_v2.py main` → `collect_v2.py` |
| Table 6, component ablation | `submit_v2.py ablation` → `make_table_ablation_v2.py` |
| Figure 3, label efficiency | `submit_v2.py main` → `frozen_agg_merged.py` |
| Figure 4a–b, baseline comparison | `submit_sutran.py`, `submit_fmv2.py` → `sota_wall_plot.py` |
| Figure 4c, adaptation cost | `submit_timing3.py`, `bench_cached_pfm.py` → `sota_wall_plot.py` |
| Role-space t-SNE | `gin15_seen_unseen.py` |
| Latent-objective seed replication | `submit_seeds.py`, `submit_seedprobes_remaining.py` |

Run tags (r14, r28, r31 ...) in the commit history map each batch to the grid it produced.

## 9. Compute cost

Measured on NVIDIA H200.

| Stage | Cost |
|---|---|
| Phase 1a, role encoder | ~20 min |
| Phase 1b, one backbone | 33–80 min |
| Main grid, 5 logs × 3 arms × 3 seeds | ~12 GPU-hours, BPI17 dominates |
| Ablation, 6 variants full budget | ~4 GPU-hours |
| Adapting one log, two tasks, cached | 0.19–1.52 min |

The full paper is roughly 40 GPU-hours including baselines. Pretraining is **loader-bound**, not
GPU-bound: MIG slices run as fast as a full H200.

---

## Known deviations

Recorded so results can be checked rather than taken on trust.

**Outcome pretext term.** The released backbone was trained with a fifth loss term, a BPI'12
application-outcome head at weight 0.3, in addition to the four documented objectives. A control
backbone without it was pretrained and probed on four held-out logs: differences are within
pretraining-seed noise, the largest being −0.035 next-activity accuracy on BPI13, with PFM-FT and
BPI20ID unchanged. The paper reports the released model and discloses the extra term.

**Inert role-contrast term.** The backbone config carries `role_contrast_weight=0.3`, but
`freeze_role=true` means the role encoder receives no gradient, so the term never contributes.

**Zero-label point.** The `0` budget is not zero-shot for the regression tasks, because an MLP head
with random initialisation has no meaningful zero-shot behaviour. Exclude it from budget curves.

**Seed variance.** Single-backbone differences of 1–3 percentage points are **not** meaningful. Three
pretraining seeds on five logs put the spread at roughly ±0.006 on aggregate next-activity accuracy.
Treat any ablation gap smaller than that as noise, including the latent objective.

**MIMIC variant count.** The dataset table's variant count for MIMIC does not reproduce from the current
`mimic_transfers.csv`; the measured value is 42,673 against a published 42,594. Every other column
of every other log reproduces exactly.
