"""Wall-clock benchmark: FROZEN PFM adapted to a held-out log with CACHED features.

Because the backbone is frozen, its forward pass does not depend on the head weights, so it only
has to run ONCE per log instead of once per epoch per task. This script measures that:

  1. build the frozen role encoder + backbone, install the eval role catalogue,
  2. encode train/val/test ONCE and cache the per-event states h_i (and the pooled trace embedding),
  3. train BOTH Setting-1 heads (next activity, remaining time) straight from the cached tensors,
  4. time from the start of encoding until every test prediction is done.

For a fair speed-up it also runs the STANDARD path (one Lightning probe per task, backbone re-run
every epoch) inside the same process, on the same GPU, under the SAME timer definition - the timer
starts after data preparation in both cases, so log parsing and splitting are excluded from both.

Everything else is the v2 protocol, identical to the main grid: chronological 70/15/15 split,
max_trace_len 64, full budget, train-partition role graph (what role_corpus=budget resolves to at
full budget), head_hidden 128, lr 1e-3, <=100 epochs, early stopping patience 10 on the validation
partition, best-val weights restored. The two paths must reach the SAME test metrics; the script
prints both so any divergence is visible.

Usage: python bench_cached_pfm.py <log> [--seed 0] [--skip-standard]
       [--backbone RUN_ID] [--tasks TASK[,TASK...]] [--out results.jsonl] [--mask-check]
TASK is any per-event task in ALL_TASKS. next_3/next_5, remaining_count and future_activity_set derive their
targets from the cached next_activity and padding_mask, exactly as their heads do on the standard path.
Defaults reproduce the r25 timing benchmark exactly. --backbone/--tasks/--out serve the feature-budget ladder
(protocols/feature_ladder.md, C1); the backbone's role_feature_mask is read from its manifest. --mask-check runs
C2.1 instead: cached states with no mask key vs an explicit all-ones mask (plus a no-mask rerun as control).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pm_foundation.data.dataset import SupervisedTraceDataset, collate_supervised
from pm_foundation.data.preprocessing import FeatureSpec, SplitStrategy, build_traces, fit_feature_spec, split_log
from pm_foundation.data.readers import get_reader
from pm_foundation.data.roles import apply_aggregator, fit_role_graph
from pm_foundation.data.schema import EventLog
from pm_foundation.evaluation.label_efficiency import _TASKS, _build_head, _KeepBestState, _strip_to_control_flow
from pm_foundation.experiments import RunManifest, RunRegistry
from pm_foundation.models import TraceBackbone
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.tasks.multitask_module import MultiTaskLitModule

# Raw logs live in DATA_DIR if set, else <repo>/data/raw, under the names scripts/check_data.py
# checks for (REPRODUCE.md, "Data"): the download filenames are renamed, so BPI13 is BPI13.xes.
RAW = os.environ.get("DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
LOGS = {
    "helpdesk": ("helpdesk.csv", None),
    "bpi13_incidents": ("BPI13.xes", None),
    "mimic_transfer": ("mimic_transfers.csv", 5000),
    "BPI20ID": ("BPI20ID.xes", None),
    "BPI17": ("BPI17.xes", None),
}
BACKBONE = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
TASKS = ["next_activity", "remaining_time"]          # default task set (r25 timing benchmark)
ALL_TASKS = ["next_activity", "next_3_activities", "next_5_activities", "next_time", "remaining_time",
             "remaining_count", "future_activity_set"]  # all per-event tasks the cache can serve
MAXLEN, BATCH, HEAD_HIDDEN, LR, MAX_EPOCHS, PATIENCE = 64, 128, 128, 1e-3, 100, 10


# --------------------------------------------------------------------------------------------
# cached-feature plumbing
# --------------------------------------------------------------------------------------------
class _CachedBackbone(nn.Module):
    """Stands in for the frozen TraceBackbone: returns the already-computed states for a batch."""

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> EncoderOutput:
        return EncoderOutput(trace_embedding=batch["trace_embedding"], event_states=batch["event_states"])


class _CacheDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.t = tensors
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.t.items()}


def _pad_to(x: torch.Tensor, length: int, value: float) -> torch.Tensor:
    """Right-pad dim 1 to `length`."""
    if x.shape[1] >= length:
        return x[:, :length]
    pad = list(x.shape)
    pad[1] = length - x.shape[1]
    return torch.cat([x, torch.full(pad, value, dtype=x.dtype, device=x.device)], dim=1)


@torch.no_grad()
def encode_split(backbone: TraceBackbone, loader: DataLoader, device: str) -> dict[str, torch.Tensor]:
    """Run the frozen backbone once over a split and keep h_i (+ targets) as dense tensors."""
    keys = {"event_states": 0.0, "trace_embedding": None, "padding_mask": True,
            "next_activity": float(_NA_IGNORE), "remaining_time": 0.0, "next_time": float("nan")}
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = backbone.forward_batch(batch)
        acc["event_states"].append(_pad_to(out.event_states, MAXLEN, 0.0))
        acc["trace_embedding"].append(out.trace_embedding)
        acc["padding_mask"].append(_pad_to(batch["padding_mask"], MAXLEN, True))
        acc["next_activity"].append(_pad_to(batch["next_activity"], MAXLEN, _NA_IGNORE))
        acc["remaining_time"].append(_pad_to(batch["remaining_time"], MAXLEN, 0.0))
        acc["next_time"].append(_pad_to(batch["next_time"], MAXLEN, float("nan")))  # last event: NaN, masked
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


def make_trainer(callbacks: list[L.Callback]) -> L.Trainer:
    return L.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=1, precision="32-true",
                     gradient_clip_val=1.0, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
                     num_sanity_val_steps=0)


def fit_and_test(module: MultiTaskLitModule, task: str, tr, va, te) -> tuple[float, int]:
    tdef = _TASKS[task]
    monitor = f"val/{task}/{tdef.metric}"
    cbs = [EarlyStopping(monitor=monitor, mode="max" if tdef.higher_is_better else "min",
                         patience=PATIENCE, strict=False),
           _KeepBestState(monitor, "max" if tdef.higher_is_better else "min")]
    t = make_trainer(cbs)
    t.fit(module, tr, va)
    res = t.test(module, te, verbose=False)[0]
    return float(res[f"test/{task}/{tdef.metric}"]), int(t.current_epoch)


def main() -> None:
    global _NA_IGNORE
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--seed", "--eval-seed", dest="seed", type=int, default=0)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", help="append the result JSON line to this file")
    ap.add_argument("--mask-check", action="store_true")
    ap.add_argument("--skip-standard", action="store_true")
    a = ap.parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    assert tasks and set(tasks) <= set(ALL_TASKS), f"--tasks must be a subset of {ALL_TASKS}"
    fname, max_traces = LOGS[a.log]
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run python scripts/check_data.py (or set DATA_DIR)")
    device = "cuda"

    from pm_foundation.data.dataset import NEXT_ACTIVITY_IGNORE_INDEX as _IGN
    _NA_IGNORE = _IGN

    # ---------------- data preparation (OUTSIDE the timer, for both variants) ----------------
    t_prep0 = time.perf_counter()
    reader_kw = {"max_traces": max_traces} if max_traces else {}
    log = get_reader(path.rsplit(".", 1)[-1], **reader_kw).read(path)
    log = _strip_to_control_flow(log)
    built = build_traces(log, min_trace_len=2, max_trace_len=MAXLEN)
    splits = split_log(built, SplitStrategy.TEMPORAL, (0.70, 0.15, 0.15), seed=0)
    registry = RunRegistry("outputs")
    run_dir = registry.run_dir("backbone", a.backbone)
    spec = FeatureSpec.load(run_dir / "feature_spec.json")
    model_cfg = dict(RunManifest.load(run_dir / "manifest.json").config["model"])
    model_cfg["causal"] = True
    eval_label_spec = fit_feature_spec(splits.train, max_seq_len=MAXLEN)
    eval_vocab = eval_label_spec.activity_vocab
    role_graph = fit_role_graph(list(splits.train.traces), eval_vocab)
    role_graph = apply_aggregator(role_graph, str(model_cfg.get("aggregator", "mean")))
    sd = torch.load(run_dir / "backbone.pt", map_location="cpu")
    role_sd = torch.load(run_dir / "role_encoder.pt", map_location="cpu") if (run_dir / "role_encoder.pt").exists() else None
    vocab_list = spec.activity_vocab.to_list()

    def raw_loader(traces, shuffle):
        ds = SupervisedTraceDataset(EventLog(traces=list(traces), activity_vocab=vocab_list), spec,
                                    target_activity_vocab=eval_vocab, role_vocab=eval_vocab)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=collate_supervised, num_workers=0)

    n_tr, n_va, n_te = len(splits.train.traces), len(splits.val.traces), len(splits.test.traces)
    t_prep = time.perf_counter() - t_prep0
    print(f"[bench] log={a.log} traces train/val/test = {n_tr}/{n_va}/{n_te} | prep {t_prep:.1f}s "
          f"(excluded from both timers)", flush=True)

    def fresh_backbone(cfg: dict | None = None) -> TraceBackbone:
        bb = TraceBackbone.from_config(cfg or model_cfg, spec)
        bb.load_pretrained(sd, role_sd)
        bb.role_encoder.set_graph(role_graph)
        return bb

    def emit(tag: str, payload: dict) -> None:
        line = json.dumps(payload)
        print(f"[{tag}] " + line, flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(line + "\n")

    if a.mask_check:  # C2.1: plumbing of role_feature_mask must not change a single cached value
        from pm_foundation.data.roles import N_ROLE_FEATURES
        ones_cfg = {**model_cfg, "role_feature_mask": list(range(N_ROLE_FEATURES))}
        checks = {}
        for label, cfg in (("control_no_mask_rerun", model_cfg), ("all_ones_mask", ones_cfg)):
            ref, oth = fresh_backbone().to(device).eval(), fresh_backbone(cfg).to(device).eval()
            with torch.no_grad():
                d = {"role_table": float((ref.role_encoder() - oth.role_encoder()).abs().max())}
            for part in ("train", "val", "test"):
                cr = encode_split(ref, raw_loader(getattr(splits, part).traces, False), device)
                co = encode_split(oth, raw_loader(getattr(splits, part).traces, False), device)
                d[f"{part}_event_states"] = float((cr["event_states"] - co["event_states"]).abs().max())
            checks[label] = d
            del ref, oth
        emit("mask-check-json", {"log": a.log, "backbone": a.backbone, "max_abs_diff": checks,
                                 "passed": all(v == 0.0 for v in checks["all_ones_mask"].values())})
        return

    # ---------------- variant A: CACHED ----------------
    torch.manual_seed(a.seed)
    tA0 = time.perf_counter()
    bb = fresh_backbone().to(device).eval()
    cache = {p: encode_split(bb, raw_loader(getattr(splits, p).traces, False), device)
             for p in ("train", "val", "test")}
    torch.cuda.synchronize()
    t_encode = time.perf_counter() - tA0
    del bb
    cached_res = {}
    for task in tasks:
        head = _build_head(task, _TASKS[task], int(model_cfg["d_model"]), spec, None,
                           head_hidden=HEAD_HIDDEN, next_activity_n=eval_label_spec.n_activities, pooling="trace")
        module = MultiTaskLitModule(_CachedBackbone(), {task: head}, freeze_backbone=True,
                                    optimizer_cfg={"lr": LR, "backbone_lr": LR},
                                    d_model=int(model_cfg["d_model"]))
        dls = [DataLoader(_CacheDataset(cache[p]), batch_size=BATCH, shuffle=(p == "train"), num_workers=0)
               for p in ("train", "val", "test")]
        v, ep = fit_and_test(module, task, *dls)
        cached_res[task] = (v, ep)
        print(f"[bench] cached  {task}: {v:.4f} ({ep} epochs)", flush=True)
    torch.cuda.synchronize()
    t_cached = time.perf_counter() - tA0

    # ---------------- variant B: STANDARD (backbone re-run every epoch) ----------------
    t_standard, std_res = float("nan"), {}
    if not a.skip_standard:
        torch.manual_seed(a.seed)
        tB0 = time.perf_counter()
        for task in tasks:
            head = _build_head(task, _TASKS[task], int(model_cfg["d_model"]), spec, None,
                               head_hidden=HEAD_HIDDEN, next_activity_n=eval_label_spec.n_activities, pooling="trace")
            module = MultiTaskLitModule(fresh_backbone(), {task: head}, freeze_backbone=True,
                                        optimizer_cfg={"lr": LR, "backbone_lr": LR},
                                        d_model=int(model_cfg["d_model"]))
            dls = [raw_loader(getattr(splits, p).traces, p == "train") for p in ("train", "val", "test")]
            v, ep = fit_and_test(module, task, *dls)
            std_res[task] = (v, ep)
            print(f"[bench] standard {task}: {v:.4f} ({ep} epochs)", flush=True)
        torch.cuda.synchronize()
        t_standard = time.perf_counter() - tB0

    out = {"log": a.log, "backbone": a.backbone, "tasks": tasks, "seed": a.seed, "n_train": n_tr, "n_val": n_va, "n_test": n_te,
           "prep_s": t_prep, "encode_s": t_encode, "cached_s": t_cached, "standard_s": t_standard,
           "cached_min": t_cached / 60, "standard_min": t_standard / 60,
           "speedup": (t_standard / t_cached) if t_standard == t_standard else None,
           "cached": {k: {"value": v, "epochs": e} for k, (v, e) in cached_res.items()},
           "standard": {k: {"value": v, "epochs": e} for k, (v, e) in std_res.items()},
           "gpu": torch.cuda.get_device_name(0)}
    emit("bench-json", out)


if __name__ == "__main__":
    main()
