# .deadcode — removed from the live code path

These 15 modules are **not used by anything in the paper**. They are kept here rather than deleted
so the history of the project stays legible; git preserves them either way.

Nothing in `src/`, `train.py`, `hpc/`, `scripts/` or `tests/` imports or calls any of them.

## What is here, and why it is dead

### The pre-PFM DINO pipeline

The project originally used a DINO-style self-distillation objective with a Lightning DataModule.
PFM replaced it with the autoregressive multi-objective module (`ssl/autoregressive.py`) driven by
Hydra through `train.py`. The whole old path came along for the ride until now.

| File | Was |
|---|---|
| `ssl/dino_loss.py` | DINO loss |
| `ssl/dino_module.py` | DINO Lightning module and encoder |
| `models/heads/projection.py` | DINO projection head, used only by `dino_module` |
| `training/pretrain.py` | the old pretrain entrypoint, built a DINO module |
| `training/finetune.py` | the old finetune entrypoint |
| `data/datamodule.py` | `ProcessMiningDataModule`, used only by the three above |

### Superseded evaluation

`evaluation/label_efficiency.py` is the protocol the paper uses. These were earlier attempts:

| File | Was |
|---|---|
| `evaluation/probing.py` | standalone linear probing |
| `evaluation/benchmark.py` | earlier task benchmark runner |
| `evaluation/report.py` | report rendering for that runner |
| `evaluation/rollout.py` | autoregressive suffix rollout for remaining time |

### Never finished, or never used

| File | Was |
|---|---|
| `cli.py` | a `pmf` console script whose `main()` only raised `NotImplementedError`. The `[project.scripts]` entry was removed from `pyproject.toml`, so installing the package no longer ships a command that crashes |
| `ssl/masked_event.py` | a masked-event objective that was implemented but never adopted |
| `models/heads/sequence.py` | `SequenceHead`, never referenced |
| `utils/config.py` | dataclass config layer, obsoleted by Hydra |
| `utils/logging.py` | `get_logger`, never referenced |

## How this was determined

Reachability from every entry point, then per-symbol reference checks across the whole repository
including configs and sbatch files. Automated analysis alone was not trusted: it reported
`training/pretrain.py` and `training/finetune.py` as live because the *words* `pretrain` and
`finetune` appear in Hydra overrides such as `task=pretrain` and `evaluate.finetune=[...]`, and
`cli.main` as live because sbatch files contain `--partition=main`. Every candidate was confirmed by
reading the actual reference.

After removal the package imports cleanly, the test suite passes, `train.py` loads all three
dispatch paths, and an end-to-end smoke test builds the paper's backbone, installs a role catalogue,
runs a forward pass and trains one probe step for both Setting-1 tasks with the backbone verifiably
frozen.

## Known leftovers

Removing the DINO path orphaned three things that live inside files still in use, so they were left
alone rather than risk editing live code:

- `data/dataset.py` — `DinoTraceDataset` and `collate_dino`
- `data/augmentations.py` — `TraceAugmentation`, `SubTraceCrop`, `EventMasking`,
  `AttributeDropout`, `TemporalJitter`; only `MultiCropTraceAugmenter` and `TraceView` are still
  referenced, and only by `DinoTraceDataset`
- `data/datamodule.py` also defined `OutcomeDataModule`, which had no users even before this change

Removing those means editing two live modules and re-running the verification above.
