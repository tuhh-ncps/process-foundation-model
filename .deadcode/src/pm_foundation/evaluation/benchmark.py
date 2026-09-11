"""Clean multi-task benchmark harness.

Evaluates one or more backbones (e.g. ``random`` vs a pretrained checkpoint) across
the downstream tasks, in **probe** (frozen) and/or **finetune** modes, on a single
aligned temporal split, averaged over seeds. This is the reusable-backbone test:
"does this pretrained backbone help, across tasks, vs a random one?"

Leakage notes: the split is whole-case temporal (no case in two splits). Per-event
tasks (next-activity, remaining-time) are *forecast-from-prefix* tasks — for an honest
number use a **causal** backbone (``model_cfg["causal"]=True``), so per-event states
do not see the future. Outcome uses the leak-free stripped prefix via its labeler.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from pm_foundation.data.dataset import (
    OutcomeDataset,
    SupervisedTraceDataset,
    collate_outcome,
    collate_supervised,
)
from pm_foundation.data.labeling import OutcomeLabeler
from pm_foundation.data.preprocessing import (
    FeatureSpec,
    SplitStrategy,
    build_traces,
    split_log,
)
from pm_foundation.data.schema import EventLog
from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.tasks.multitask_module import MultiTaskLitModule
from pm_foundation.tasks.next_activity import NextActivityHead
from pm_foundation.tasks.outcome import OutcomeHead
from pm_foundation.tasks.remaining_time import RemainingTimeHead

Report = dict[str, dict[str, float]]
PER_EVENT_TASKS = ("next_activity", "remaining_time")


def _make_backbone(
    model_cfg: dict[str, Any], spec: FeatureSpec, weights: str | None
) -> TraceBackbone:
    backbone = TraceBackbone.from_config(model_cfg, spec)
    if weights is not None:
        backbone.load_state_dict(torch.load(weights, map_location="cpu"))
    return backbone


def _trainer(max_epochs: int, accelerator: str) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
    )


def _fit_test(
    backbone: TraceBackbone,
    heads: dict[str, Any],
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    *,
    freeze: bool,
    lr: float,
    max_epochs: int,
    accelerator: str,
) -> dict[str, float]:
    module = MultiTaskLitModule(backbone, heads, freeze_backbone=freeze, optimizer_cfg={"lr": lr})
    trainer = _trainer(max_epochs, accelerator)
    trainer.fit(module, train_loader, val_loader)
    return {k: float(v) for k, v in trainer.test(module, test_loader, verbose=False)[0].items()}


def run_task_benchmark(
    event_log: EventLog,
    feature_spec: FeatureSpec,
    model_cfg: dict[str, Any],
    *,
    backbones: dict[str, str | None],
    labeler: OutcomeLabeler | None = None,
    tasks: Sequence[str] = ("next_activity", "remaining_time", "outcome"),
    modes: Sequence[str] = ("probe",),
    seeds: Sequence[int] = (0,),
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_seed: int = 0,
    min_trace_len: int = 2,
    batch_size: int = 128,
    max_epochs: int = 12,
    accelerator: str = "cpu",
    lr: float = 1e-3,
) -> Report:
    """Run the benchmark; return ``report[f"{backbone}/{mode}"] = {metric: mean_over_seeds}``."""
    d_model = int(model_cfg.get("d_model", 256))
    log = build_traces(event_log, min_trace_len=min_trace_len)
    splits = split_log(log, SplitStrategy.TEMPORAL, split, seed=split_seed)
    per_event = [t for t in tasks if t in PER_EVENT_TASKS]
    do_outcome = "outcome" in tasks
    if do_outcome and labeler is None:
        raise ValueError("outcome task requires a labeler")
    out_pairs = (
        {s: labeler.apply(getattr(splits, s)) for s in ("train", "val", "test")}
        if do_outcome and labeler is not None
        else {}
    )

    def sup_loader(split_name: str, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            SupervisedTraceDataset(getattr(splits, split_name), feature_spec),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_supervised,
            drop_last=shuffle,
        )

    def out_loader(split_name: str, shuffle: bool) -> DataLoader[Any]:
        pair = out_pairs[split_name]
        return DataLoader(
            OutcomeDataset(pair[0], pair[1], feature_spec),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_outcome,
            drop_last=shuffle,
        )

    report: Report = {}
    for name, weights in backbones.items():
        for mode in modes:
            freeze = mode == "probe"
            per_seed: list[dict[str, float]] = []
            for seed in seeds:
                metrics: dict[str, float] = {}
                if per_event:
                    L.seed_everything(seed, workers=True)
                    heads: dict[str, Any] = {}
                    if "next_activity" in per_event:
                        heads["next_activity"] = NextActivityHead(
                            d_model, feature_spec.n_activities
                        )
                    if "remaining_time" in per_event:
                        heads["remaining_time"] = RemainingTimeHead(d_model)
                    metrics |= _fit_test(
                        _make_backbone(model_cfg, feature_spec, weights),
                        heads,
                        sup_loader("train", True),
                        sup_loader("val", False),
                        sup_loader("test", False),
                        freeze=freeze,
                        lr=lr,
                        max_epochs=max_epochs,
                        accelerator=accelerator,
                    )
                if do_outcome and labeler is not None:
                    L.seed_everything(seed, workers=True)
                    metrics |= _fit_test(
                        _make_backbone(model_cfg, feature_spec, weights),
                        {"outcome": OutcomeHead(d_model, labeler.n_classes)},
                        out_loader("train", True),
                        out_loader("val", False),
                        out_loader("test", False),
                        freeze=freeze,
                        lr=lr,
                        max_epochs=max_epochs,
                        accelerator=accelerator,
                    )
                per_seed.append(metrics)
            keys = per_seed[0].keys()
            report[f"{name}/{mode}"] = {
                k: sum(d[k] for d in per_seed) / len(per_seed) for k in keys
            }
    return report


def load_event_log(path: str | Path, fmt: str = "xes", **reader_kwargs: Any) -> EventLog:
    """Convenience: read an event log for benchmarking."""
    from pm_foundation.data.readers import get_reader

    return get_reader(fmt, **reader_kwargs).read(path)
