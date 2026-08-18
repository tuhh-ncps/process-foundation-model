"""DDP / Slurm variant of AR pretraining — kept SEPARATE from the local entrypoint.

This is the multi-node / multi-GPU version used only by ``scripts/train_hpc.py`` on a cluster.
It reuses the *pure* data-prep helpers from :mod:`pm_foundation.training.ar_pretrain` (reading,
stripping, splitting, spec fit) but has its own training/save orchestration so the local
``pretrain_autoregressive`` stays byte-for-byte unchanged. Differences from the local version:

- only **global rank 0** creates the run directory and writes artifacts (other ranks just do
  their share of the DDP compute and return),
- the ``Trainer`` receives ``num_nodes`` / ``strategy`` / ``use_distributed_sampler``,
- the length-bucketed sampler is DDP-sharded (disjoint, equal-count per rank).

Everything is driven by the Slurm environment (one process per GPU); see ``docs/hpc.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pm_foundation.data.dataset import (
    OutcomeDataset,
    SupervisedTraceDataset,
    TraceDataset,
    collate_outcome,
    collate_supervised,
    collate_traces,
)
from pm_foundation.data.labeling import (
    bpi12_application_outcome,
    bpi17_application_outcome,
    bpi20id_declaration_outcome,
    mimic_mortality,
    sepsis_admission_outcome,
)
from pm_foundation.data.preprocessing import fit_feature_spec
from pm_foundation.data.roles import fit_role_graph
from pm_foundation.data.samplers import LengthBucketedSampler
from pm_foundation.data.schema import EventLog, Trace
from pm_foundation.experiments import RunRegistry, plot_learning_curve, write_learning_curve
from pm_foundation.ssl import build_autoregressive_module
from pm_foundation.training.ar_pretrain import _read_log, _train_traces  # pure, shared helpers
from pm_foundation.training.callbacks import LearningCurveRecorder

# Outcome labelers usable as a pretext head (name -> factory).
_OUTCOME_LABELERS = {
    "bpi12_application_outcome": bpi12_application_outcome,
    "bpi17_application_outcome": bpi17_application_outcome,
    "bpi20id_declaration_outcome": bpi20id_declaration_outcome,
    "sepsis_admission_outcome": sepsis_admission_outcome,
    "mimic_mortality": mimic_mortality,
}


def global_rank() -> int:
    """Global process rank under Slurm/torch DDP (0 for single-process)."""
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))


def world_size() -> int:
    """Total number of DDP processes (1 for single-process)."""
    return int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))


def pretrain_autoregressive_ddp(config: dict[str, Any]) -> Path:
    """DDP-safe AR pretraining. Returns the run directory (rank 0) / output dir (other ranks)."""
    seed = int(config.get("seed", 0))
    L.seed_everything(seed, workers=True)

    strip = bool(config.get("strip_to_control_flow", True))
    min_len = int(config.get("min_trace_len", 2))
    split = tuple(config.get("split", (0.70, 0.15, 0.15)))
    vocab_logs = list(config["vocab_logs"])
    train_logs = list(config.get("train_logs") or vocab_logs)
    rank, world = global_rank(), world_size()
    is_main = rank == 0

    # All ranks build the identical spec + data (deterministic), so DDP replicas agree.
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

    # Role channel: fit activity fingerprints + DFG on the TRAIN split ONLY (same leakage
    # contract as fit_feature_spec — never traces that will be predicted/scored).
    role_graph = None
    if int(config["model"].get("role_dim", 0)) > 0:
        role_graph = fit_role_graph(train_traces, feature_spec.activity_vocab)
        data_summary["role_corpus"] = {"split": "train", "n_traces": len(train_traces)}

    # Only rank 0 owns the run dir + artifacts (else ranks race / mint different run ids).
    output_dir = Path(config.get("output_dir", "outputs"))
    registry = RunRegistry(output_dir)
    ctx = (
        registry.start("backbone", config, name=config.get("name"), data=data_summary)
        if is_main
        else None
    )
    run_dir = ctx.dir if ctx is not None else output_dir

    # Auxiliary pretext heads (e.g. remaining-time) need per-event targets in the batch, so use
    # the supervised dataset/collate when enabled; otherwise the plain AR features path.
    ar_cfg = dict(config.get("ar") or {})
    aux_targets = float(ar_cfg.get("remaining_time_weight", 0.0)) > 0
    event_log = EventLog(traces=train_traces, activity_vocab=shared_vocab)
    dataset = (SupervisedTraceDataset if aux_targets else TraceDataset)(event_log, feature_spec)
    collate = collate_supervised if aux_targets else collate_traces
    batch_size = int(config.get("batch_size", 96))
    trainer_cfg = dict(config.get("trainer") or {})
    # Multi-worker data loading. KEEP AT 0 for the main pretrain path: the dataset is a large
    # in-memory list of pydantic traces, so num_workers>0 forks a heavy copy per worker and stalls
    # DataLoader startup (observed to hang on the 382k-trace MIMIC set). >0 only suits a light dataset.
    n_workers = int(config.get("num_workers", 0))
    dl_kw: dict[str, Any] = {"num_workers": n_workers, "pin_memory": n_workers > 0}
    if n_workers > 0:
        dl_kw["persistent_workers"] = True

    # Outcome pretext head: a SEPARATE data path over the labeler's stripped prefixes (leak-free).
    outcome_loader = None
    if float(ar_cfg.get("outcome_weight", 0.0)) > 0:
        name = str(ar_cfg.get("outcome_labeler", "bpi12_application_outcome"))
        if name not in _OUTCOME_LABELERS:
            raise ValueError(f"unknown outcome_labeler {name!r}; known: {list(_OUTCOME_LABELERS)}")
        o_traces, o_labels = _OUTCOME_LABELERS[name]().apply(event_log)
        if o_traces:
            ar_cfg["outcome_classes"] = int(max(o_labels)) + 1
            o_ds = OutcomeDataset(o_traces, o_labels, feature_spec)
            if world > 1:
                o_samp = DistributedSampler(
                    o_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True
                )
                outcome_loader = DataLoader(
                    o_ds,
                    batch_size=batch_size,
                    sampler=o_samp,
                    collate_fn=collate_outcome,
                    drop_last=True,
                    **dl_kw,
                )
            else:
                outcome_loader = DataLoader(
                    o_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_outcome,
                    drop_last=True,
                    **dl_kw,
                )
        elif is_main:
            print(
                f"[ar] outcome pretext: labeler {name!r} produced 0 labeled traces — head disabled"
            )

    if bool(config.get("length_bucketing", False)):
        # Custom batch_sampler disables Lightning's auto DistributedSampler → we shard it.
        cap = feature_spec.max_seq_len
        lengths = [min(len(t.events), cap) if cap else len(t.events) for t in train_traces]
        loader = DataLoader(
            dataset,
            batch_sampler=LengthBucketedSampler(
                lengths,
                batch_size,
                shuffle=True,
                drop_last=True,
                num_replicas=world,
                rank=rank,
                seed=seed,
            ),
            collate_fn=collate,
            **dl_kw,
        )
        use_distributed_sampler = False
    elif outcome_loader is not None and world > 1:
        # CombinedLoader path shards manually, so the AR loader needs its own DistributedSampler.
        ar_samp = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=ar_samp,
            collate_fn=collate,
            drop_last=True,
            **dl_kw,
        )
        use_distributed_sampler = False
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
            **dl_kw,
        )
        use_distributed_sampler = outcome_loader is None  # CombinedLoader shards manually

    # Pair the AR and outcome loaders; the shorter (outcome) cycles to match the AR loader.
    if outcome_loader is not None:
        loader = CombinedLoader({"ar": loader, "outcome": outcome_loader}, mode="max_size_cycle")

    module = build_autoregressive_module(dict(config["model"]), ar_cfg, feature_spec)

    # Continue-pretraining ("seen data" transfer): warm-start the backbone from a prior run's
    # weights before adapting on this corpus. Only SHAPE-MATCHING tensors transfer — the transformer
    # blocks and the vocab-free role encoder (GIN + projection) carry over, while vocab-sized tensors
    # (id embedding, output projections) reinitialize for THIS corpus's activity set. This runs
    # BEFORE set_role_graph so the new corpus's fingerprints/DFG install on the transferred encoder.
    init_from = config.get("init_from")
    if init_from:
        src = torch.load(
            registry.run_dir("backbone", init_from) / "backbone.pt", map_location="cpu"
        )
        own = module.backbone.state_dict()
        transfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        reinit = sorted(k for k in own if k not in transfer)
        module.backbone.load_state_dict(transfer, strict=False)
        if is_main:
            print(
                f"[ar] init_from {init_from}: transferred {len(transfer)}/{len(own)} backbone "
                f"tensors; reinitialized for this corpus: {reinit}",
                flush=True,
            )

    if role_graph is not None:  # install on student + JEPA teacher copy
        module.set_role_graph(role_graph)
    recorder = LearningCurveRecorder()
    trainer = L.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 15)),
        accelerator=str(trainer_cfg.get("accelerator", "gpu")),
        devices=trainer_cfg.get("devices", 1),
        num_nodes=int(trainer_cfg.get("num_nodes", 1)),
        strategy=str(trainer_cfg.get("strategy", "auto")),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        use_distributed_sampler=use_distributed_sampler,
        logger=config.get("logger")
        or False,  # e.g. a WandbLogger (offline on the cluster); rank-0 safe
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", False)),
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[recorder],
        default_root_dir=str(run_dir),
    )
    trainer.fit(module, loader)

    if not (is_main and ctx is not None):
        return run_dir  # non-main ranks: DDP work done in fit(); rank 0 owns the artifacts

    torch.save(module.backbone.state_dict(), run_dir / "backbone.pt")
    ar_heads: dict[str, Any] = {
        # With candidate matching the fixed linear head does not exist; the matching
        # query projection is saved instead (the candidate bank lives in backbone.pt).
        **(
            {"activity_head": module.activity_head.state_dict()}
            if module.activity_head is not None
            else {"match_query": module.match_query.state_dict()}
        ),
        "time_head": module.time_head.state_dict(),
        "n_activities": module.n_activities,
        "end_id": module.end_id,
        "time_dist": module.time_dist,
        "objective": module.objective,
        "log_delta_mean": feature_spec.time_mean["log_delta"],
        "log_delta_std": feature_spec.time_std["log_delta"],
    }
    if module.time_cond_emb is not None:
        ar_heads["time_cond_emb"] = module.time_cond_emb.state_dict()
    if module.remaining_head is not None:
        ar_heads["remaining_head"] = module.remaining_head.state_dict()
    if module.outcome_head is not None:
        ar_heads["outcome_head"] = module.outcome_head.state_dict()
    if module.jepa_predictor is not None:
        ar_heads["jepa_predictor"] = module.jepa_predictor.state_dict()
        ar_heads["jepa_proj"] = module.jepa_proj.state_dict()
    torch.save(ar_heads, run_dir / "ar_heads.pt")
    feature_spec.save(run_dir / "feature_spec.json")
    write_learning_curve(run_dir / "learning_curve.csv", recorder.history)
    plot_learning_curve(
        run_dir / "learning_curve.csv",
        run_dir / "learning_curve.png",
        title=f"AR backbone {ctx.run_id}",
    )
    final = recorder.history[-1] if recorder.history else {}
    registry.finish(ctx, metrics={"final_epoch": final, "n_epochs": len(recorder.history)})
    return run_dir
