"""Label-efficiency evaluation across multiple heads, with tracked outputs.

For each downstream task, each backbone, each label budget, and each seed, this trains
a probe on a random subsample of the training traces and measures the test metric.
Probes are **frozen** by default (backbone frozen; only the head learns — the classic
"how much do the pretrained features buy you" setting). With ``finetune`` a backbone is
trained **end-to-end** instead; a ``random`` backbone + finetune is a NORMAL TRANSFORMER
trained from scratch — the honest supervised baseline for the label-efficiency hypothesis
(*does a pretrained foundation backbone reach the target with fewer labels than a
from-scratch transformer?*). ``labels_to_target.csv`` reports exactly that: the smallest
budget each backbone needs to hit the per-task target. Results for every task
are written together under ``outputs/label_efficiency/<run_id>/`` as one long-format
CSV, a mean-over-seeds CSV, and one PNG per task. For the classification tasks
(``next_activity`` and ``outcome``) a confusion matrix — PNG heatmap (row-normalized
recall) plus a raw-count CSV — is also written per backbone, computed at the largest
label budget and the first seed (the best-case probe). The run manifest names the
backbone run(s) it consumed, and each backbone directory is back-linked to this
evaluation, so curves and backbones are traceable both ways.

Backbones are given as ``{alias: backbone_run_id | "random"}``. The ``"random"`` alias
builds a freshly-initialised backbone (the honest untrained baseline). The feature spec
and model architecture are taken from the first real backbone so every variant is scored
on an identical encoding.

Config schema (a plain dict; see ``configs/experiment/label_efficiency.yaml``)::

    backbones: {ar_multi: <run_id>, ar_bpi12: <run_id>, random: "random"}
    eval_log:  {path, format?, max_traces?}
    strip_to_control_flow: true
    tasks:      [next_activity, outcome, remaining_time, next_time]
    label_sizes: [100, 300, 1000, null]     # null -> use all training traces
    seeds:       [0, 1]
    probe:      {max_epochs: 15, lr: 1.0e-3, batch_size: 128, early_stop_patience: 3}
                # each probe early-stops on the val split (monitors val/<task>/<metric>) and the
                # best-val weights are restored; max_epochs is the cap, patience the plateau window.
    outcome:    {labeler: bpi12_application_outcome}   # required iff "outcome" in tasks
    pooling:    trace      # case-head pooling for ALL outcome-kind tasks (shared adapter over the
                           # per-event states H): trace(P0) | mean(P1) | last(P2) | attention(P3).
    finetune:   []         # [] frozen probes (default) | true (all end-to-end) | [alias,...].
                           # A `random` backbone + finetune = a normal transformer from scratch.
    target:     {frac: 0.95}   # "labels required" = smallest budget reaching frac of the best
                               # per-task mean; or {abs: {outcome: 0.7}} for absolute thresholds.
    warm_start: false      # DEFAULT: fresh random heads. Opt-in true warm-starts
                           # next-activity/next-time heads from the pretext heads (docs §15).
    holdout:    true       # DEFAULT: 70/15/15 train/val/test split. false => NO split (cross-domain):
                           # probe trains on subsamples of the FULL log and tests on the FULL log
                           # (stable full-domain metric; largest-budget point is an upper bound).
    split: [0.70, 0.15, 0.15]
    min_trace_len: 2
    fresh:      false      # DEFAULT: crash-recovery ON. Each probe is checkpointed to
                           # ${output_dir}/label_efficiency/.ckpt/<config-hash>/; re-submitting an
                           # identical run resumes (skips done probes). The checkpoint is deleted once
                           # the grid finishes. true => ignore any checkpoint and start clean.
    model: {...}    # only used when every backbone is "random"
    name: "bpi12-label-efficiency"
    output_dir: outputs
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from pm_foundation.data.dataset import (
    NEXT_ACTIVITY_IGNORE_INDEX,
    OutcomeDataset,
    SupervisedTraceDataset,
    collate_outcome,
    collate_supervised,
)
from pm_foundation.data.labeling import (
    bpi12_application_outcome,
    bpi17_application_outcome,
    bpi20id_declaration_outcome,
    mimic_icu_24h,
    mimic_icu_48h,
    mimic_icu_ever_24h,
    mimic_icu_ever_48h,
    mimic_los_24h,
    mimic_los_48h,
    mimic_los_long_24h,
    mimic_los_long_48h,
    mimic_mortality,
    mimic_mortality_24h,
    mimic_mortality_48h,
    mimic_readmission_30d,
    sepsis_admission_outcome,
)
from pm_foundation.data.preprocessing import (
    FeatureSpec,
    Splits,
    SplitStrategy,
    build_traces,
    fit_feature_spec,
    split_log,
)
from pm_foundation.data.readers import get_reader
from pm_foundation.data.roles import fit_role_graph
from pm_foundation.data.samplers import LengthBucketedSampler
from pm_foundation.data.schema import Event, EventLog, Trace
from pm_foundation.data.vocabulary import RESERVED_TOKENS, Vocabulary
from pm_foundation.evaluation.confusion import build_confusion, plot_confusion, write_confusion_csv
from pm_foundation.experiments import (
    RunManifest,
    RunRegistry,
    plot_label_efficiency,
    write_label_efficiency,
)
from pm_foundation.experiments.curves import (
    aggregate_label_efficiency,
    write_label_efficiency_means,
)
from pm_foundation.models import TraceBackbone
from pm_foundation.tasks import (
    FutureActivitySetHead,
    MultiTaskLitModule,
    NextActivityHead,
    NextKActivitiesHead,
    NextTimeHead,
    OutcomeHead,
    RemainingCountHead,
    RemainingTimeHead,
    TaskHead,
)
from pm_foundation.tasks import warmstart as ws

# Evaluation spins up many short-lived DataLoaders (one Trainer per task x backbone x size x seed).
# PyTorch's default "file_descriptor" tensor-sharing opens an FD per shared tensor, which exhausts
# the OS limit (Errno 24: too many open files) on large corpora like MIMIC. "file_system" names
# shared segments instead, so FDs don't accumulate across loaders.
with contextlib.suppress(RuntimeError, AttributeError):  # platform without this strategy
    torch.multiprocessing.set_sharing_strategy("file_system")

_ALL = 10_000_000  # sentinel used when a label budget is null/None ("use everything")
_RANDOM = "random"  # frozen, vocab-blind untrained floor (knows NO dataset's vocabulary)
_SCRATCH = "scratch"  # random-init, eval-vocab, trained end-to-end (the from-scratch transformer)
# Labelers selectable as the `outcome` task's labeler (evaluate.outcome.labeler=<name>).
# The mimic labelers below are ALSO wired as first-class task names via _TASK_LABELERS, so a
# single run can evaluate several of them at once (tasks: [mortality, los, icu, readmission]).
_LABELERS: dict[str, Callable[[], Any]] = {
    "bpi12_application_outcome": bpi12_application_outcome,
    "bpi17_application_outcome": bpi17_application_outcome,
    "bpi20id_declaration_outcome": bpi20id_declaration_outcome,
    "sepsis_admission_outcome": sepsis_admission_outcome,
    "mimic_mortality": mimic_mortality,
    "mimic_mortality_24h": mimic_mortality_24h,
    "mimic_mortality_48h": mimic_mortality_48h,
    "mimic_los_24h": mimic_los_24h,
    "mimic_los_48h": mimic_los_48h,
    "mimic_los_long_24h": mimic_los_long_24h,
    "mimic_los_long_48h": mimic_los_long_48h,
    "mimic_icu_24h": mimic_icu_24h,
    "mimic_icu_48h": mimic_icu_48h,
    "mimic_icu_ever_24h": mimic_icu_ever_24h,
    "mimic_icu_ever_48h": mimic_icu_ever_48h,
    "mimic_readmission_30d": mimic_readmission_30d,
}

# First-class outcome-kind TASK names -> their fixed labeler. Each is an independent
# classification task with its OWN train/val/test outcome splits, so `tasks` may list any
# subset and they all run in one grid (unlike the single config-driven `outcome` task).
# mortality/los/icu predict from a FIXED early observation window (see labeling.py) — the
# leak-free, standardized-prediction-time formulation.
_TASK_LABELERS: dict[str, Callable[[], Any]] = {
    "mortality_24h": mimic_mortality_24h,
    "mortality_48h": mimic_mortality_48h,
    "los_24h": mimic_los_24h,
    "los_48h": mimic_los_48h,
    "los_long_24h": mimic_los_long_24h,
    "los_long_48h": mimic_los_long_48h,
    "icu_24h": mimic_icu_24h,
    "icu_48h": mimic_icu_48h,
    "icu_ever_24h": mimic_icu_ever_24h,
    "icu_ever_48h": mimic_icu_ever_48h,
    "readmission": mimic_readmission_30d,
}


@dataclass(frozen=True)
class _TaskDef:
    kind: str  # "per_event" | "outcome"
    metric: str  # metric suffix, e.g. "acc" / "mae" / "macro_f1"
    higher_is_better: bool


# Per-event head factories: (d_model, spec, head_hidden). head_hidden>0 gives the time-regression
# probes a nonlinear MLP head (d_model -> head_hidden -> GELU -> 1) so they can access the
# backbone's NON-linear time representation; a linear probe only sees linearly-decodable signal.
# Classification heads (next_activity) ignore head_hidden.
_PER_EVENT_HEADS: dict[str, Callable[[int, FeatureSpec, int], TaskHead]] = {
    "next_activity": lambda d, s, h: NextActivityHead(d, s.n_activities),
    "remaining_time": lambda d, s, h: RemainingTimeHead(d, hidden_dim=h or None),
    "next_time": lambda d, s, h: NextTimeHead(d, hidden_dim=h or None),
    "remaining_count": lambda d, s, h: RemainingCountHead(d, hidden_dim=h or None),
}
_TASKS: dict[str, _TaskDef] = {
    "next_activity": _TaskDef("per_event", "acc", True),
    "next_3_activities": _TaskDef("per_event", "acc", True),  # multi-step (K=3) horizon probe
    "next_5_activities": _TaskDef("per_event", "acc", True),  # multi-step (K=5) horizon probe
    "remaining_time": _TaskDef("per_event", "mae", False),
    "next_time": _TaskDef("per_event", "mae", False),
    "remaining_count": _TaskDef("per_event", "mae", False),  # #events until case end
    "future_activity_set": _TaskDef(
        "per_event", "micro_f1", True
    ),  # suffix activity set (multi-label)
    "outcome": _TaskDef("outcome", "macro_f1", True),
    # First-class outcome-kind tasks (labeler-backed; see _TASK_LABELERS), from a fixed 24h/48h
    # observation window (leak-free prediction point). The imbalanced BINARY tasks report AUROC
    # (threshold-free) as their primary metric — macro-F1 collapses to the majority floor under
    # 1-3% prevalence even when the model ranks well (a strong LR baseline hits AUROC ~0.79 on
    # mortality). The 3-class LOS bucket keeps macro-F1. All tasks also compute auroc/auprc/f1/acc.
    "mortality_24h": _TaskDef("outcome", "auroc", True),
    "mortality_48h": _TaskDef("outcome", "auroc", True),
    "los_24h": _TaskDef("outcome", "macro_f1", True),  # 3-class, balanced — macro-F1 is fine
    "los_48h": _TaskDef("outcome", "macro_f1", True),
    "los_long_24h": _TaskDef("outcome", "auroc", True),  # binary LOS>=7d (~30% pos)
    "los_long_48h": _TaskDef("outcome", "auroc", True),
    "icu_24h": _TaskDef("outcome", "auroc", True),
    "icu_48h": _TaskDef("outcome", "auroc", True),
    "icu_ever_24h": _TaskDef("outcome", "auroc", True),  # ICU ever (~4% pos), less floor-prone
    "icu_ever_48h": _TaskDef("outcome", "auroc", True),
    "readmission": _TaskDef("outcome", "auroc", True),
}


class _KeepBestState(L.Callback):
    """Restore the best-validation weights at the end of ``fit``, in memory.

    ``EarlyStopping`` decides *when* to stop but leaves the model at the last (post-plateau)
    epoch; this keeps a CPU snapshot of the best-scoring epoch and reloads it, so the probe is
    evaluated at its best-val point rather than its final one. Kept in RAM (not on disk) because
    the grid runs one short-lived Trainer per task x backbone x size x seed — hundreds of tiny
    checkpoint files would otherwise churn.
    """

    def __init__(self, monitor: str, mode: str) -> None:
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.state: dict[str, torch.Tensor] | None = None

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:  # the pre-train sanity pass is not a real epoch
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:  # metric absent (e.g. empty val split) — nothing to select on
            return
        value = float(current)
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.state is not None:
            pl_module.load_state_dict(self.state)


def _strip_to_control_flow(log: EventLog) -> EventLog:
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
    return EventLog(
        traces=traces, activity_vocab=sorted({e.activity for t in traces for e in t.events})
    )


def _encoded_len(trace: Trace, spec: FeatureSpec) -> int:
    """Number of events after truncation — the padded length the collate will produce."""
    n = len(trace.events)
    return min(n, spec.max_seq_len) if spec.max_seq_len else n


def _subsample(items: list[Any], size: int, seed: int) -> list[Any]:
    if size >= len(items):
        return list(items)
    gen = torch.Generator().manual_seed(1000 + seed)
    idx = torch.randperm(len(items), generator=gen)[:size].tolist()
    return [items[i] for i in idx]


@torch.no_grad()
def _collect_confusion(
    module: L.LightningModule, test_dl: DataLoader[Any], task: str, class_ids: torch.Tensor
) -> torch.Tensor:
    """Run the trained frozen probe over the test set and return a (C, C) confusion matrix.

    ``next_activity``: per-position, with the argmax restricted to real-activity columns
    (a next activity is never a reserved token — same convention as the zero-shot head).
    ``outcome``: per-trace over all outcome classes. Reuses the backbone's own device.
    """
    module.eval()
    device = module.device
    head = module.heads[task]
    real = class_ids.to(device=device, dtype=torch.long)
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in test_dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = head(module.backbone.forward_batch(batch), batch["padding_mask"])
        if task == "next_activity":
            tgt = batch["next_activity"]
            valid = tgt != NEXT_ACTIVITY_IGNORE_INDEX
            pred = real[
                logits[..., real].argmax(dim=-1)
            ]  # map real-column argmax back to vocab ids
            preds.append(pred[valid].cpu())
            targets.append(tgt[valid].cpu())
        else:  # outcome (trace-level)
            preds.append(logits.argmax(dim=-1).cpu())
            targets.append(batch["outcome"].cpu())
    if not preds:
        n = int(class_ids.shape[0])
        return torch.zeros(n, n, dtype=torch.long)
    return build_confusion(torch.cat(preds), torch.cat(targets), class_ids.cpu())


def _resolve_per_backbone(
    registry: RunRegistry, backbones: dict[str, str], config: dict[str, Any], train_log: EventLog
) -> dict[str, tuple[FeatureSpec, dict[str, Any]]]:
    """Per-alias (feature spec, model config) so backbones with DIFFERENT architectures/specs
    (e.g. learned+capped vs RoPE+uncapped) can be compared on one plot: each backbone is built
    and its eval data encoded with its OWN spec.

    Two untrained baselines are handled specially (no weights loaded), both a PLAIN transformer
    (role_dim=0) sized like the first real backbone (or ``config.model`` if all-untrained):

      - ``random`` — a FROZEN, **vocab-blind** floor: its activity vocabulary is empty, so every
        activity (from either dataset) maps to UNK. It genuinely knows NO vocabulary — just
        position + random features + a trained head. The honest bottom line, on equal (dead-ID)
        footing with disjoint-vocab cross-domain backbones.
      - ``scratch`` — a random-init model trained END-TO-END on the eval labels; it uses the eval
        dataset's own vocabulary (so it can learn activity embeddings). The "normal transformer
        from scratch" baseline (auto-finetuned in the sweep)."""
    per: dict[str, tuple[FeatureSpec, dict[str, Any]]] = {}
    ref_cfg: dict[str, Any] | None = None  # architecture of the first real backbone
    for alias, run_id in backbones.items():
        if run_id in (_RANDOM, _SCRATCH):
            continue
        run_dir = registry.run_dir("backbone", run_id)
        spec = FeatureSpec.load(run_dir / "feature_spec.json")
        model_cfg = dict(RunManifest.load(run_dir / "manifest.json").config["model"])
        model_cfg["causal"] = True  # probes read last-event state; keep leak-free pooling
        per[alias] = (spec, model_cfg)
        ref_cfg = ref_cfg or model_cfg
    if any(r in (_RANDOM, _SCRATCH) for r in backbones.values()):
        base_cfg = {**(ref_cfg or dict(config["model"])), "causal": True, "role_dim": 0}
        eval_spec = fit_feature_spec(
            train_log, max_seq_len=int(config["model"].get("max_seq_len", 64))
        )
        blind_spec = replace(
            eval_spec, activity_vocab=Vocabulary.build([])
        )  # every activity -> UNK
        for alias, run_id in backbones.items():
            if run_id == _RANDOM:
                per[alias] = (blind_spec, base_cfg)
            elif run_id == _SCRATCH:
                per[alias] = (eval_spec, base_cfg)
    return per


def _labels_to_target(
    records: list[dict[str, Any]], *, frac: float = 0.95, absolute: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """The crux of the label-efficiency hypothesis: for each (task, backbone), the SMALLEST label
    budget whose mean metric meets the task target — a pretrained backbone should reach it at far
    fewer labels than a from-scratch transformer.

    Target = ``absolute[task]`` if given, else ``frac`` of the best mean achieved for that task by
    ANY backbone at ANY budget (for a lower-is-better metric, the best is the min and the target is
    ``min / frac``). "Meets" respects the metric direction. ``labels_needed`` is "not reached" when
    no budget qualifies (a strong result in itself — the backbone never gets there).
    """
    agg = aggregate_label_efficiency(records)
    mode_of = {r["backbone_alias"]: r.get("mode", "frozen") for r in records}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for g in agg:
        by_task.setdefault(g["task"], []).append(g)
    rows: list[dict[str, Any]] = []
    for task, gs in by_task.items():
        hib = bool(gs[0]["higher_is_better"])
        best = max(g["mean_value"] for g in gs) if hib else min(g["mean_value"] for g in gs)
        if absolute and task in absolute:
            target = float(absolute[task])
        else:
            target = frac * best if hib else best / max(frac, 1e-9)
        by_bb: dict[str, list[dict[str, Any]]] = {}
        for g in gs:
            by_bb.setdefault(g["backbone_alias"], []).append(g)
        for alias, bg in by_bb.items():
            bg.sort(key=lambda g: (g["n_train_samples"] or 0))  # ascending real training-set size
            hit = next(
                (
                    g
                    for g in bg
                    if (g["mean_value"] >= target if hib else g["mean_value"] <= target)
                ),
                None,
            )
            bb_best = max(g["mean_value"] for g in bg) if hib else min(g["mean_value"] for g in bg)
            rows.append(
                {
                    "task": task,
                    "metric": bg[0]["metric"],
                    "backbone_alias": alias,
                    "mode": mode_of.get(alias, "frozen"),
                    "higher_is_better": hib,
                    "target": round(target, 4),
                    "labels_needed": (hit["n_labels"] if hit else "not reached"),
                    "n_train_needed": (hit["n_train_samples"] if hit else None),
                    "best_mean": round(bb_best, 4),
                }
            )
    return rows


def _checkpoint_key(config: dict[str, Any], backbones: dict[str, str]) -> str:
    """Stable hash over the fields that DEFINE this grid — so re-running an identical, crashed run
    resumes its checkpoint, while any real change (backbone, dataset, budgets, probe settings)
    starts a fresh one. Excludes cosmetic fields (name/output_dir)."""
    payload = {
        "backbones": dict(sorted(backbones.items())),
        "eval_log": (config.get("eval_log") or {}).get("path"),
        "tasks": list(config.get("tasks", [])),
        "label_sizes": list(config.get("label_sizes", [])),
        "seeds": list(config.get("seeds", [0, 1])),
        "split": list(config.get("split", (0.70, 0.15, 0.15))),
        "warm_start": bool(config.get("warm_start", False)),
        "holdout": bool(config.get("holdout", True)),
        "finetune": config.get("finetune", []),
        "pooling": config.get("pooling", "trace"),  # case-head pooling (P0-P3) is grid-defining
        "outcome": (config.get("outcome") or {}).get("labeler"),
        "probe": {
            k: (config.get("probe") or {}).get(k)
            for k in (
                "max_epochs",
                "lr",
                "backbone_lr",
                "batch_size",
                "head_hidden",
                "early_stop_patience",
                "normalize_features",
            )
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_label_efficiency(config: dict[str, Any]) -> Path:
    """Run the multi-head label-efficiency sweep and persist artifacts + provenance."""
    backbones: dict[str, str] = dict(config["backbones"])
    tasks: list[str] = list(config["tasks"])
    unknown = [t for t in tasks if t not in _TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}; known: {list(_TASKS)}")
    sizes = [int(s) if s is not None else _ALL for s in config["label_sizes"]]
    seeds = list(config.get("seeds", [0, 1]))
    probe_cfg = dict(config.get("probe") or {})
    split = tuple(config.get("split", (0.70, 0.15, 0.15)))
    min_len = int(config.get("min_trace_len", 2))
    strip = bool(config.get("strip_to_control_flow", True))
    warm_start = bool(config.get("warm_start", False))  # default: fresh heads (opt-in only)
    length_bucketing = bool(config.get("length_bucketing", False))  # keep padding tight if uncapped
    # Shared case-head pooling for ALL outcome-kind tasks (Template B): trace (P0, fixed-summary
    # linear probe) | mean/last/max (parameter-free aggregations of the event states) | attention
    # (P3, learned event-state aggregation). Run the same grid with different pooling to separate
    # "using all event states" (mean/last/max) from "learning event importance" (attention).
    pooling = str(config.get("pooling", "trace"))
    # Optional trainable LayerNorm on the frozen backbone features before the head (probe-side,
    # backbone untouched) — accelerates the low-budget end of a linear probe. Default off.
    normalize_features = bool(probe_cfg.get("normalize_features", False))

    # End-to-end (unfrozen) training. `finetune` is a bool (all backbones) or a list of aliases.
    # A `random` backbone + finetune = a NORMAL TRANSFORMER trained from scratch (the honest
    # supervised baseline); a pretrained backbone + finetune = full finetuning. Default: [] (every
    # backbone is a frozen linear probe, the classic label-efficiency-of-features setting).
    ft = config.get("finetune", [])
    finetune_aliases: set[str] = set(backbones) if ft is True else set(ft or [])
    # A `scratch` backbone is a from-scratch transformer → always trained end-to-end.
    finetune_aliases |= {a for a, r in backbones.items() if r == _SCRATCH}
    # "Labels required" target: fraction of the best-achievable mean per task (default), or an
    # absolute per-task threshold (target.abs.{task}). Drives labels_to_target.csv.
    tgt_cfg = dict(config.get("target") or {})
    target_frac = float(tgt_cfg.get("frac", 0.95))
    target_abs = tgt_cfg.get("abs") or None

    log = get_reader(
        str(
            config["eval_log"].get("format") or Path(config["eval_log"]["path"]).suffix.lstrip(".")
        ),
        **(
            {"max_traces": config["eval_log"]["max_traces"]}
            if config["eval_log"].get("max_traces")
            else {}
        ),
    ).read(config["eval_log"]["path"])
    if strip:
        log = _strip_to_control_flow(log)
    built = build_traces(log, min_trace_len=min_len)
    # holdout=false: NO train/val/test split — the probe trains on subsamples of the FULL eval log
    # and is tested on the FULL log. Intended for CROSS-DOMAIN eval (the frozen backbone never saw
    # this domain, so there is no backbone-leakage). At small budgets this is ~leak-free few-shot
    # label efficiency; at the largest budget train == test, so THAT point is an optimistic upper
    # bound. Descriptors (feature spec + role graph) are then fit over all traces too — fine for
    # cross-domain aggregate stats, but do NOT use holdout=false for a leakage-strict in-domain claim.
    holdout = bool(config.get("holdout", True))
    if holdout:
        splits = split_log(built, SplitStrategy.TEMPORAL, split, seed=0)
    else:
        splits = Splits(train=built, val=built, test=built)
        print(
            f"[label-efficiency] holdout=false: training pool = test = FULL eval log "
            f"({len(built.traces)} traces); largest-budget metric is an upper bound (train==test).",
            flush=True,
        )

    registry = RunRegistry(config.get("output_dir", "outputs"))
    bb_specs = _resolve_per_backbone(registry, backbones, config, splits.train)
    ref_spec = next(iter(bb_specs.values()))[0]  # for provenance (all share the same vocab)

    # Next-activity TARGET label space = the EVAL dataset's OWN activity vocabulary (fit here on the
    # eval train split), DECOUPLED from each backbone's input-encoding vocab. Inputs are encoded with
    # the backbone's training vocab — cross-dataset activities land on UNK there — but the target is
    # the eval dataset's REAL next activity, so accuracy honestly measures transfer instead of
    # collapsing to "predict UNK" (a false ~100%). Same-dataset eval is unchanged (identical vocab,
    # and accuracy is invariant to id ordering).
    eval_label_spec = fit_feature_spec(
        splits.train, max_seq_len=int(config["model"].get("max_seq_len", 64))
    )
    eval_activity_vocab = eval_label_spec.activity_vocab
    eval_n_activities = eval_label_spec.n_activities

    # Confusion matrices (next_activity / outcome) are collected at the LARGEST label budget and
    # the first seed. The next-activity matrix is over REAL activities only (reserved PAD/UNK/…
    # excluded — a next activity is never a reserved token).
    reserved = set(RESERVED_TOKENS)
    _eval_vocab_list = eval_activity_vocab.to_list()
    na_real_ids = torch.tensor(
        [i for i, nm in enumerate(_eval_vocab_list) if nm not in reserved], dtype=torch.long
    )
    na_real_names = [_eval_vocab_list[i] for i in na_real_ids.tolist()]

    # Role channel (backbones with role_dim>0): fit the EVAL catalogue's fingerprints/DFG
    # on the eval dataset's TRAIN SPLIT ONLY — never the val/test traces being scored
    # (data/roles.py leakage contract). The frozen ActivityEncoder maps this new catalogue
    # into the shared role space with no retraining.
    eval_role_graph = None
    if any(int(cfg.get("role_dim", 0)) > 0 for _, cfg in bb_specs.values()):
        eval_role_graph = fit_role_graph(list(splits.train.traces), eval_activity_vocab)

    # Preload backbone weights once per alias (state dicts kept on CPU).
    weights: dict[str, dict[str, torch.Tensor] | None] = {}
    for alias, run_id in backbones.items():
        weights[alias] = (
            None
            if run_id in (_RANDOM, _SCRATCH)
            else torch.load(
                registry.run_dir("backbone", run_id) / "backbone.pt", map_location="cpu"
            )
        )

    # Outcome supervision (each labeler strips decision-leaking events itself). Every outcome-kind
    # task gets its OWN labeler + train/val/test splits, so one grid can score several at once
    # (mortality/los/icu/readmission), plus the config-driven generic `outcome` task.
    task_labelers: dict[str, Any] = {}
    task_outcome_splits: dict[str, dict[str, tuple[list[Trace], list[int]]]] = {}
    for task in [t for t in tasks if _TASKS[t].kind == "outcome"]:
        if task == "outcome":
            name = str((config.get("outcome") or {}).get("labeler", "bpi12_application_outcome"))
            if name not in _LABELERS:
                raise ValueError(f"Unknown outcome labeler {name!r}; known: {list(_LABELERS)}")
            lab = _LABELERS[name]()
        else:
            name = task
            lab = _TASK_LABELERS[task]()
        splits_t = {
            part: lab.apply(subset)
            for part, subset in (
                ("train", splits.train),
                ("val", splits.val),
                ("test", splits.test),
            )
        }
        # A labeler that matches nothing on this dataset (e.g. a mimic labeler on a BPI log, or the
        # wrong `outcome.labeler`) yields empty splits -> empty loaders -> Trainer.test() returns []
        # -> IndexError, aborting the whole grid. Skip that task gracefully instead (mirrors the
        # pretrain path's "0 labeled traces -> disabled"); other tasks in the grid still run.
        n_lab = {p: len(v[0]) for p, v in splits_t.items()}
        if min(n_lab.values()) == 0:
            print(
                f"[label-efficiency] labeler {name!r} for task {task!r} produced empty splits on this "
                f"dataset (train/val/test = {n_lab['train']}/{n_lab['val']}/{n_lab['test']}) — SKIPPING "
                f"task {task!r}.",
                flush=True,
            )
            tasks = [t for t in tasks if t != task]
            continue
        task_labelers[task] = lab
        task_outcome_splits[task] = splits_t

    batch_size = int(probe_cfg.get("batch_size", 128))

    # Multi-worker dataloading (use the job's CPUs; single-process by default to stay unchanged).
    n_workers = int(probe_cfg.get("num_workers", 0))
    dl_kw: dict[str, Any] = {"num_workers": n_workers, "pin_memory": n_workers > 0}
    if n_workers > 0:
        dl_kw["persistent_workers"] = True  # avoid re-spawning workers for every probe's Trainer

    def make_loaders(
        spec: FeatureSpec,
    ) -> tuple[Callable[..., DataLoader[Any]], Callable[..., DataLoader[Any]]]:
        """Build per-event / outcome loaders that encode with THIS backbone's spec."""
        vocab = spec.activity_vocab.to_list()

        def _loader(ds: Any, traces: list[Trace], shuffle: bool, collate: Any) -> DataLoader[Any]:
            if length_bucketing:  # keep padding tight when traces are uncapped (docs §18)
                lengths = [_encoded_len(t, spec) for t in traces]
                sampler = LengthBucketedSampler(lengths, batch_size, shuffle=shuffle)
                return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, **dl_kw)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate,
                drop_last=False,
                **dl_kw,
            )

        # role_vocab: only when a role backbone is present — role_ids encoded with the EVAL
        # catalogue so its e(a) table (installed via set_graph) is indexed consistently.
        role_vocab = eval_activity_vocab if eval_role_graph is not None else None

        def per_event_loader(traces: list[Trace], shuffle: bool) -> DataLoader[Any]:
            # Encode inputs with THIS backbone's spec; encode next-activity TARGETS with the eval
            # dataset's vocab (honest cross-dataset accuracy — see eval_activity_vocab above).
            ds = SupervisedTraceDataset(
                EventLog(traces=list(traces), activity_vocab=vocab),
                spec,
                target_activity_vocab=eval_activity_vocab,
                role_vocab=role_vocab,
            )
            return _loader(ds, list(traces), shuffle, collate_supervised)

        def outcome_loader(pair: tuple[list[Trace], list[int]], shuffle: bool) -> DataLoader[Any]:
            traces, labels = pair
            ds = OutcomeDataset(traces, labels, spec, role_vocab=role_vocab)
            return _loader(ds, list(traces), shuffle, collate_outcome)

        return per_event_loader, outcome_loader

    def trainer(callbacks: list[L.Callback] | None = None) -> L.Trainer:
        tcfg = dict(config.get("trainer") or {})
        return L.Trainer(
            max_epochs=int(probe_cfg.get("max_epochs", 15)),
            accelerator=str(tcfg.get("accelerator", "auto")),
            devices=tcfg.get("devices", 1),
            precision=tcfg.get("precision", "32-true"),
            gradient_clip_val=1.0,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

    def probe(
        task: str, alias: str, size: int, seed: int, collect: bool = False
    ) -> tuple[float, int, torch.Tensor | None]:
        L.seed_everything(seed, workers=True)
        spec, model_cfg = bb_specs[alias]  # this backbone's own spec + architecture
        tdef = _TASKS[task]
        head = _build_head(
            task,
            tdef,
            int(model_cfg["d_model"]),
            spec,
            task_labelers.get(task),
            head_hidden=int(probe_cfg.get("head_hidden", 0)),
            next_activity_n=eval_n_activities,
            pooling=pooling,
        )
        # Warm-start heads whose task matches a pretext (next-activity/next-time) from the
        # backbone's persisted pretext heads — near-zero-label transfer. Random backbones
        # and non-pretext tasks (outcome, remaining-time) fall through to a fresh head.
        if warm_start and backbones[alias] not in (_RANDOM, _SCRATCH) and task in ws.WARM_STARTABLE:
            ws.warm_start(head, task, registry.run_dir("backbone", backbones[alias]))
        bb = TraceBackbone.from_config(model_cfg, spec)
        if weights[alias] is not None:
            bb.load_state_dict(weights[alias])
        # Role backbones: install the EVAL catalogue (train-split-only graph) AFTER loading,
        # so the frozen encoder scores this dataset's activities in the shared role space.
        role_encoder = getattr(bb.embedding, "role_encoder", None)
        if role_encoder is not None and eval_role_graph is not None:
            role_encoder.set_graph(eval_role_graph)
        # Frozen probe (default) vs end-to-end. For end-to-end, backbone_lr defaults to the head lr
        # (a from-scratch transformer must learn its whole stack, not gently adapt); override
        # probe.backbone_lr (e.g. 0.1x) to gently finetune a PRETRAINED backbone instead.
        finetune = alias in finetune_aliases
        module = MultiTaskLitModule(
            bb,
            {task: head},
            freeze_backbone=not finetune,
            optimizer_cfg={
                "lr": float(probe_cfg.get("lr", 1e-3)),
                "backbone_lr": float(probe_cfg.get("backbone_lr", probe_cfg.get("lr", 1e-3))),
            },
            feature_norm=normalize_features,
            d_model=int(model_cfg["d_model"]),
        )
        per_event_loader, outcome_loader = make_loaders(spec)
        train_dl, val_dl, test_dl, n_train = _task_loaders(
            task,
            tdef,
            size,
            seed,
            splits,
            task_outcome_splits.get(task, {}),
            per_event_loader,
            outcome_loader,
        )
        # Early-stop on the held-out val split and restore the best-val weights (in memory).
        # `max_epochs` is the cap; patience lets a plateaued probe stop sooner. `strict=False`
        # tolerates an occasionally-empty val split (metric absent -> no stop, no crash).
        monitor = f"val/{task}/{tdef.metric}"
        es_mode = "max" if tdef.higher_is_better else "min"
        patience = int(probe_cfg.get("early_stop_patience", 3))
        callbacks: list[L.Callback] = [
            EarlyStopping(monitor=monitor, mode=es_mode, patience=patience, strict=False),
            _KeepBestState(monitor, es_mode),
        ]
        t = trainer(callbacks)
        t.fit(module, train_dl, val_dl)
        result = t.test(module, test_dl, verbose=False)[0]
        # Confusion matrix for the classification tasks (best-case: largest budget, first seed).
        cm = None
        if collect and (task == "next_activity" or tdef.kind == "outcome"):
            class_ids = (
                na_real_ids
                if task == "next_activity"
                else torch.arange(task_labelers[task].n_classes)
            )
            cm = _collect_confusion(module, test_dl, task, class_ids)
        return float(result[f"test/{task}/{tdef.metric}"]), n_train, cm

    total = len(tasks) * len(backbones) * len(sizes) * len(seeds)
    done = 0
    records: list[dict[str, Any]] = []
    # next_activity + every labeler-backed outcome task gets a confusion matrix.
    cm_tasks = [t for t in tasks if t == "next_activity" or _TASKS[t].kind == "outcome"]
    max_size, first_seed = max(sizes), seeds[0]
    cm_store: dict[tuple[str, str], torch.Tensor] = {}  # (task, alias) -> confusion matrix

    # --- crash-recovery checkpoint ------------------------------------------------------------
    # Each probe is independent, but records/cm only hit disk at the very end — a crash mid-grid
    # (OOM, walltime, a labeler mismatch) loses ALL completed probes. Persist every finished probe
    # to a config-keyed checkpoint and skip already-done probes on re-run, so re-submitting the same
    # command resumes instead of restarting. `fresh: true` forces a clean run; the checkpoint is
    # deleted once the full grid succeeds (it exists only to recover a crash).
    ckpt_dir = (
        Path(config.get("output_dir", "outputs"))
        / "label_efficiency"
        / ".ckpt"
        / _checkpoint_key(config, backbones)
    )
    if bool(config.get("fresh", False)):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prog_path = ckpt_dir / "progress.jsonl"
    completed: set[tuple[str, str, int, int]] = set()
    if prog_path.exists():
        for line in prog_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["task"], rec["backbone_alias"], int(rec["_size_raw"]), int(rec["seed"]))
            if key in completed:
                continue
            completed.add(key)
            records.append({k: v for k, v in rec.items() if not k.startswith("_")})
        for pt in sorted(ckpt_dir.glob("cm_*.pt")):
            d = torch.load(pt, map_location="cpu", weights_only=False)
            cm_store[(d["task"], d["alias"])] = d["cm"]
        if completed:
            print(
                f"[label-efficiency] resuming checkpoint {ckpt_dir.name}: "
                f"{len(completed)}/{total} probes already done — skipping them",
                flush=True,
            )

    for task in tasks:
        tdef = _TASKS[task]
        for alias in backbones:
            for size in sizes:
                for seed in seeds:
                    done += 1
                    if (task, alias, size, seed) in completed:
                        continue  # recovered from checkpoint
                    print(
                        f"[label-efficiency] probe {done}/{total}  task={task} "
                        f"backbone={alias} n_labels={'all' if size >= _ALL else size} seed={seed}",
                        flush=True,
                    )
                    collect = task in cm_tasks and size == max_size and seed == first_seed
                    value, n_train, cm = probe(task, alias, size, seed, collect=collect)
                    if cm is not None:
                        cm_store[(task, alias)] = cm
                    rec = {
                        "task": task,
                        "metric": tdef.metric,
                        "higher_is_better": tdef.higher_is_better,
                        "backbone_alias": alias,
                        "backbone_run_id": backbones[alias],
                        "mode": "finetune" if alias in finetune_aliases else "frozen",
                        "n_labels": ("all" if size >= _ALL else size),
                        "n_train_samples": n_train,  # ACTUAL #training cases (budget capped by availability)
                        "seed": seed,
                        "value": value,
                    }
                    records.append(rec)
                    # durably checkpoint this probe (append+close so a crash keeps prior probes)
                    with prog_path.open("a") as fh:
                        fh.write(json.dumps({**rec, "_size_raw": size}) + "\n")
                    if cm is not None:
                        torch.save(
                            {"task": task, "alias": alias, "cm": cm}, ckpt_dir / f"cm_{done}.pt"
                        )

    # Activity-vocabulary overlap between each backbone (input vocab) and the eval dataset — the
    # ceiling on how much ACTIVITY-IDENTITY knowledge can transfer. ~1.0 = same/shared vocab (full
    # transfer possible); ~0.0 = disjoint vocab (only structural/temporal signal can transfer, and
    # next-activity accuracy will be near a time-only baseline). Recorded so cross-dataset runs are
    # self-documenting.
    eval_acts = {a for a in eval_activity_vocab.to_list() if a not in reserved}
    vocab_overlap = {}
    for alias, (spec, _) in bb_specs.items():
        bb_acts = {a for a in spec.activity_vocab.to_list() if a not in reserved}
        shared = len(eval_acts & bb_acts)
        vocab_overlap[alias] = {
            "shared_activities": shared,
            "eval_activities": len(eval_acts),
            "fraction": round(shared / max(len(eval_acts), 1), 4),
        }

    ctx = registry.start(
        "label_efficiency",
        config,
        name=config.get("name"),
        data={
            "eval_log": config["eval_log"]["path"],
            "n_activities": ref_spec.n_activities,
            "eval_n_activities": eval_n_activities,
            "vocab_overlap": vocab_overlap,
            # Auditable descriptor provenance: role fingerprints/DFG come from
            # the eval TRAIN split only (never scored traces).
            "role_corpus": (
                {"split": "train", "n_traces": len(splits.train.traces)}
                if eval_role_graph is not None
                else None
            ),
            "tasks": tasks,
            "sizes": [("all" if s >= _ALL else s) for s in sizes],
            "seeds": seeds,
            "holdout": holdout,
            # Confusion matrices for the classification tasks (best-case probe).
            "confusion_matrices": (
                {"tasks": cm_tasks, "n_labels": "all", "seed": first_seed} if cm_tasks else None
            ),
            # Label-efficiency hypothesis: which backbones trained end-to-end,
            # and the "labels required" target definition.
            "finetune": sorted(finetune_aliases),
            "target": {"frac": target_frac, "abs": target_abs},
        },
    )
    write_label_efficiency(ctx.dir / "curves.csv", records)
    write_label_efficiency_means(ctx.dir / "curves_mean.csv", records)
    plot_label_efficiency(records, ctx.dir, x_scale=str(config.get("x_scale", "log")))

    # Confusion matrices (PNG + raw-count CSV) per backbone for next_activity / outcome, at the
    # largest label budget and first seed — see cm_store above.
    for (task, alias), cm in cm_store.items():
        names = na_real_names if task == "next_activity" else list(task_labelers[task].classes)
        stem = ctx.dir / f"confusion_{task}_{alias}"
        write_confusion_csv(cm, names, stem.with_suffix(".csv"))
        plot_confusion(
            cm,
            names,
            stem.with_suffix(".png"),
            f"{alias}: {task} confusion (n_labels=all, seed={first_seed}) — row=recall",
        )
        print(
            f"[label-efficiency] confusion {task}/{alias}: {int(cm.sum()):,} predictions "
            f"-> {stem.name}.png",
            flush=True,
        )

    # "How many labels required?" — smallest budget each backbone needs to hit the per-task target.
    # This is the headline of the foundation-vs-from-scratch hypothesis: a pretrained backbone
    # should reach the target at a far smaller budget than a normal transformer trained from scratch.
    lt_rows = _labels_to_target(records, frac=target_frac, absolute=target_abs)
    lt_fields = (
        "task",
        "metric",
        "backbone_alias",
        "mode",
        "higher_is_better",
        "target",
        "labels_needed",
        "n_train_needed",
        "best_mean",
    )
    with (ctx.dir / "labels_to_target.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lt_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(lt_rows)
    print(
        "[label-efficiency] labels required to reach target (per task, per backbone):", flush=True
    )
    for r in lt_rows:
        need = (
            r["labels_needed"]
            if r["labels_needed"] == "not reached"
            else f"{r['labels_needed']} labels ({r['n_train_needed']} cases)"
        )
        print(
            f"    {r['task']:<14} {r['backbone_alias']:<12} target {r['metric']}"
            f"{'>=' if r['higher_is_better'] else '<='}{r['target']}: {need}  "
            f"(best={r['best_mean']})",
            flush=True,
        )

    real_backbones = {a: r for a, r in backbones.items() if r not in (_RANDOM, _SCRATCH)}
    for alias, run_id in real_backbones.items():
        registry.link_backbone_to_eval(run_id, ctx.run_id, info={"alias": alias, "tasks": tasks})
    registry.finish(ctx, links={"backbones": backbones})
    shutil.rmtree(
        ckpt_dir, ignore_errors=True
    )  # grid finished cleanly — drop the recovery checkpoint
    return ctx.dir


def _build_head(
    task: str,
    tdef: _TaskDef,
    d_model: int,
    spec: FeatureSpec,
    labeler: Any,
    head_hidden: int = 0,
    *,
    next_activity_n: int | None = None,
    pooling: str = "trace",
) -> TaskHead:
    if tdef.kind == "outcome":
        assert labeler is not None
        return OutcomeHead(d_model, labeler.n_classes, pooling=pooling)
    # Activity-prediction heads output over the eval dataset's activity vocab (matches the decoupled
    # target), NOT the backbone's input vocab — so the metric is honest under cross-dataset transfer.
    n = next_activity_n if next_activity_n is not None else spec.n_activities
    if task == "next_activity":
        return NextActivityHead(d_model, n)
    if task == "next_3_activities":
        return NextKActivitiesHead(d_model, n, k=3)
    if task == "next_5_activities":
        return NextKActivitiesHead(d_model, n, k=5)
    if task == "future_activity_set":
        return FutureActivitySetHead(d_model, n)
    return _PER_EVENT_HEADS[task](d_model, spec, head_hidden)


def _task_loaders(
    task: str,
    tdef: _TaskDef,
    size: int,
    seed: int,
    splits: Any,
    outcome_splits: dict[str, tuple[list[Trace], list[int]]],
    per_event_loader: Callable[[list[Trace], bool], DataLoader[Any]],
    outcome_loader: Callable[[tuple[list[Trace], list[int]], bool], DataLoader[Any]],
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int]:
    """Returns (train_dl, val_dl, test_dl, n_train) where n_train is the ACTUAL number of
    training cases used (the label budget capped by what's available) — used to place the
    label-efficiency x-axis proportional to real sample count and to record it in the CSVs."""
    if tdef.kind == "outcome":
        train_pairs = list(zip(*outcome_splits["train"], strict=True))
        sub = _subsample(train_pairs, size, seed)
        train_traces = [p[0] for p in sub]
        train_labels = [p[1] for p in sub]
        return (
            outcome_loader((train_traces, train_labels), True),
            outcome_loader(outcome_splits["val"], False),
            outcome_loader(outcome_splits["test"], False),
            len(sub),
        )
    train = _subsample(list(splits.train.traces), size, seed)
    return (
        per_event_loader(train, True),
        per_event_loader(list(splits.val.traces), False),
        per_event_loader(list(splits.test.traces), False),
        len(train),
    )
