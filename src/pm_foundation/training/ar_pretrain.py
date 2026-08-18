"""Autoregressive ("process-GPT") backbone pretraining, with tracked outputs.

Trains a causal AR backbone (see :mod:`pm_foundation.ssl.autoregressive`) and records
everything needed to reuse and trace it: the backbone weights, the (possibly
multi-log) shared feature spec, a per-epoch **learning curve** (CSV + PNG), and a
run manifest. Every run lives under ``outputs/backbones/<run_id>/`` via
:class:`~pm_foundation.experiments.provenance.RunRegistry`.

Config schema (a plain dict; see ``configs/pretrain/ar_bpi12.yaml``)::

    vocab_logs:  [{path, format?, max_traces?}, ...]   # fit the shared spec on these
    train_logs:  [{path, format?, max_traces?}, ...]   # train the backbone on these
                                                        # (defaults to vocab_logs)
    strip_to_control_flow: true        # keep only activity + timestamp per event
    min_trace_len: 2
    split: [0.70, 0.15, 0.15]
    batch_size: 96
    model:   {...}                     # TraceBackbone config (causal forced on)
    ar:      {time_weight, optimizer}  # AutoregressiveLitModule config
    trainer: {max_epochs, accelerator, devices, precision, gradient_clip_val}
    seed: 0
    name: "bpi12+bpi17"                # optional human label for the run
    output_dir: outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from pm_foundation.data.dataset import TraceDataset, collate_traces
from pm_foundation.data.preprocessing import (
    FeatureSpec,
    SplitStrategy,
    build_traces,
    fit_feature_spec,
    split_log,
)
from pm_foundation.data.readers import get_reader
from pm_foundation.data.samplers import LengthBucketedSampler
from pm_foundation.data.schema import Event, EventLog, Trace
from pm_foundation.experiments import (
    RunRegistry,
    plot_learning_curve,
    write_learning_curve,
)
from pm_foundation.ssl import build_autoregressive_module
from pm_foundation.training.callbacks import LearningCurveRecorder


def _strip_to_control_flow(log: EventLog) -> EventLog:
    """Drop every event attribute except activity + timestamp (control-flow view)."""
    traces = [
        Trace(
            case_id=t.case_id,
            events=[
                Event(case_id=e.case_id, activity=e.activity, timestamp=e.timestamp)
                for e in t.events
            ],
        )
        for t in log.traces
    ]
    vocab = sorted({e.activity for t in traces for e in t.events})
    return EventLog(traces=traces, activity_vocab=vocab)


def _read_log(spec: dict[str, Any], *, strip: bool) -> EventLog:
    path = spec["path"]
    fmt = str(spec.get("format") or Path(path).suffix.lstrip("."))
    reader_kwargs = {"max_traces": spec["max_traces"]} if spec.get("max_traces") else {}
    log = get_reader(fmt, **reader_kwargs).read(path)
    return _strip_to_control_flow(log) if strip else log


def _train_traces(
    log: EventLog, *, min_trace_len: int, split: tuple[float, float, float]
) -> list[Trace]:
    splits = split_log(
        build_traces(log, min_trace_len=min_trace_len), SplitStrategy.TEMPORAL, split, seed=0
    )
    return list(splits.train.traces)


def pretrain_autoregressive(config: dict[str, Any]) -> Path:
    """Pretrain an AR backbone and persist artifacts. Returns the run directory."""
    seed = int(config.get("seed", 0))
    L.seed_everything(seed, workers=True)

    strip = bool(config.get("strip_to_control_flow", True))
    min_len = int(config.get("min_trace_len", 2))
    split = tuple(config.get("split", (0.70, 0.15, 0.15)))
    vocab_logs = list(config["vocab_logs"])
    train_logs = list(config.get("train_logs") or vocab_logs)

    # Shared vocabulary + feature spec fit over the TRAIN split of every vocab log,
    # so backbones trained on different corpora remain directly comparable downstream.
    vocab_train: list[Trace] = []
    data_summary: dict[str, Any] = {
        "strip_to_control_flow": strip,
        "vocab_logs": [],
        "train_logs": [],
    }
    for spec in vocab_logs:
        traces = _train_traces(_read_log(spec, strip=strip), min_trace_len=min_len, split=split)
        vocab_train.extend(traces)
        data_summary["vocab_logs"].append({"path": spec["path"], "n_train_traces": len(traces)})
    shared_vocab = sorted({e.activity for t in vocab_train for e in t.events})
    feature_spec = fit_feature_spec(
        EventLog(traces=vocab_train, activity_vocab=shared_vocab),
        max_seq_len=int(config["model"].get("max_seq_len", 64)),
    )

    train_traces: list[Trace] = []
    for spec in train_logs:
        traces = _train_traces(_read_log(spec, strip=strip), min_trace_len=min_len, split=split)
        train_traces.extend(traces)
        data_summary["train_logs"].append({"path": spec["path"], "n_train_traces": len(traces)})
    data_summary["n_activities"] = feature_spec.n_activities
    data_summary["n_train_traces_total"] = len(train_traces)

    registry = RunRegistry(config.get("output_dir", "outputs"))
    ctx = registry.start("backbone", config, name=config.get("name"), data=data_summary)

    dataset = TraceDataset(EventLog(traces=train_traces, activity_vocab=shared_vocab), feature_spec)
    batch_size = int(config.get("batch_size", 96))
    if bool(config.get("length_bucketing", False)):
        # Group similar-length traces so uncapped (max_seq_len=0) batches stay tightly padded.
        cap = feature_spec.max_seq_len
        lengths = [min(len(t.events), cap) if cap else len(t.events) for t in train_traces]
        loader = DataLoader(
            dataset,
            batch_sampler=LengthBucketedSampler(lengths, batch_size, shuffle=True, drop_last=True),
            collate_fn=collate_traces,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_traces, drop_last=True
        )
    module = build_autoregressive_module(
        dict(config["model"]), dict(config.get("ar") or {}), feature_spec
    )

    trainer_cfg = dict(config.get("trainer") or {})
    recorder = LearningCurveRecorder()
    trainer = L.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 15)),
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        logger=False,
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", False)),
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[recorder],
        default_root_dir=str(ctx.dir),
    )
    trainer.fit(module, loader)

    # Artifacts: backbone weights, shared spec, learning curve (CSV always, PNG if viz).
    torch.save(module.backbone.state_dict(), ctx.dir / "backbone.pt")
    # Persist the pretext AR heads: they let a downstream head whose task matches a pretext
    # (next-activity, next-time) be *warm-started* from them for near-zero-label transfer
    # (see tasks/warmstart.py, docs §15), and they are also needed for AR rollout. The
    # log-Δt standardization stats travel with them so the next-time affine is self-contained.
    ar_heads: dict[str, Any] = {
        "activity_head": module.activity_head.state_dict(),
        "time_head": module.time_head.state_dict(),
        "n_activities": module.n_activities,
        "end_id": module.end_id,
        "time_dist": module.time_dist,
        "objective": module.objective,
        "log_delta_mean": feature_spec.time_mean["log_delta"],
        "log_delta_std": feature_spec.time_std["log_delta"],
    }
    if module.time_cond_emb is not None:  # joint objective conditions Δt on the next activity
        ar_heads["time_cond_emb"] = module.time_cond_emb.state_dict()
    torch.save(ar_heads, ctx.dir / "ar_heads.pt")
    feature_spec.save(ctx.dir / "feature_spec.json")
    write_learning_curve(ctx.dir / "learning_curve.csv", recorder.history)
    plot_learning_curve(
        ctx.dir / "learning_curve.csv",
        ctx.dir / "learning_curve.png",
        title=f"AR backbone {ctx.run_id}",
    )

    final = recorder.history[-1] if recorder.history else {}
    registry.finish(ctx, metrics={"final_epoch": final, "n_epochs": len(recorder.history)})
    return ctx.dir


def load_backbone_spec(run_dir: str | Path) -> FeatureSpec:
    """Load the feature spec saved alongside a backbone run (for downstream eval)."""
    return FeatureSpec.load(Path(run_dir) / "feature_spec.json")
