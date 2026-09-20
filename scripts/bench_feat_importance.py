r"""Permutation importance of the 15 role-fingerprint features, measured on DOWNSTREAM task metrics.

For each feature j: shuffle column j of the fingerprint across the log's real activities (destroying
the association between that feature and the activity while keeping its marginal distribution),
re-run the frozen role encoder so e(a) changes, re-encode the log with the frozen backbone, train
the probe heads, and record how much the test metric degrades. Importance_j is that degradation.

Affordable only because the backbone is frozen: each variant costs one encoding pass plus two
head trainings from the cached states (bench_cached_pfm.py), not a full re-run per epoch.

Controls, so the signal is the feature and not training noise:
  * the head init and batch order are re-seeded identically before every variant,
  * the baseline (unpermuted) is run with the same seeds,
  * PERM_SEEDS independent shuffles per feature, reported as a mean.

CAVEAT recorded in the output: the 15 features are correlated, so permuting one at a time
UNDER-states the importance of any feature its neighbours can stand in for. Read this together with
the redundancy measure in feats_importance_plot.py.

Usage: python bench_feat_importance.py <log> [--perm-seeds 3]
"""
from __future__ import annotations

import argparse
import json
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from pm_foundation.data.dataset import SupervisedTraceDataset, collate_supervised
from pm_foundation.data.preprocessing import FeatureSpec, SplitStrategy, build_traces, fit_feature_spec, split_log
from pm_foundation.data.readers import get_reader
from pm_foundation.data.roles import apply_aggregator, fit_role_graph
from pm_foundation.data.schema import EventLog
from pm_foundation.evaluation.label_efficiency import _TASKS, _build_head, _KeepBestState, _strip_to_control_flow
from pm_foundation.experiments import RunManifest, RunRegistry
from pm_foundation.models import TraceBackbone
from pm_foundation.tasks.multitask_module import MultiTaskLitModule

import bench_cached_pfm as B   # reuse _CachedBackbone, _CacheDataset, encode_split, make_trainer

NAMES = ["pagerank", "betweenness", "self_loop_p", "in_gap_median", "in_gap_std", "out_gap_median",
         "out_gap_std", "p_start", "p_terminal", "support", "pred_entropy", "succ_entropy",
         "mean_pos", "std_pos", "rework_p"]
TASKS = ["next_activity", "remaining_time"]
BACKBONE = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
MAXLEN, BATCH, HEAD_HIDDEN, LR, MAX_EPOCHS, PATIENCE = 64, 128, 128, 1e-3, 100, 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--perm-seeds", type=int, default=3)
    a = ap.parse_args()
    path, max_traces = B.LOGS[a.log]
    device = "cuda"
    B._NA_IGNORE = __import__("pm_foundation.data.dataset", fromlist=["x"]).NEXT_ACTIVITY_IGNORE_INDEX

    reader_kw = {"max_traces": max_traces} if max_traces else {}
    log = _strip_to_control_flow(get_reader(path.rsplit(".", 1)[-1], **reader_kw).read(path))
    built = build_traces(log, min_trace_len=2, max_trace_len=MAXLEN)
    splits = split_log(built, SplitStrategy.TEMPORAL, (0.70, 0.15, 0.15), seed=0)
    run_dir = RunRegistry("outputs").run_dir("backbone", BACKBONE)
    spec = FeatureSpec.load(run_dir / "feature_spec.json")
    model_cfg = dict(RunManifest.load(run_dir / "manifest.json").config["model"])
    model_cfg["causal"] = True
    eval_label_spec = fit_feature_spec(splits.train, max_seq_len=MAXLEN)
    eval_vocab = eval_label_spec.activity_vocab
    graph0 = apply_aggregator(fit_role_graph(list(splits.train.traces), eval_vocab),
                              str(model_cfg.get("aggregator", "mean")))
    sd = torch.load(run_dir / "backbone.pt", map_location="cpu")
    role_pt = run_dir / "role_encoder.pt"
    role_sd = torch.load(role_pt, map_location="cpu") if role_pt.exists() else None
    vocab_list = spec.activity_vocab.to_list()
    real = torch.nonzero(graph0["real_mask"]).flatten()
    print(f"[imp] log={a.log} traces {len(splits.train.traces)}/{len(splits.val.traces)}/{len(splits.test.traces)} "
          f"| {len(real)} real activities | {a.perm_seeds} permutation seeds", flush=True)

    def raw_loader(traces, shuffle):
        ds = SupervisedTraceDataset(EventLog(traces=list(traces), activity_vocab=vocab_list), spec,
                                    target_activity_vocab=eval_vocab, role_vocab=eval_vocab)
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=collate_supervised, num_workers=0)

    def run(graph) -> dict[str, float]:
        """Encode once with this role graph, then train both heads from the cache. Fixed seeds."""
        torch.manual_seed(0)
        bb = TraceBackbone.from_config(model_cfg, spec)
        bb.load_pretrained(sd, role_sd)
        bb.role_encoder.set_graph(graph)
        bb = bb.to(device).eval()
        cache = {p: B.encode_split(bb, raw_loader(getattr(splits, p).traces, False), device)
                 for p in ("train", "val", "test")}
        del bb
        out = {}
        for task in TASKS:
            torch.manual_seed(0)                      # identical head init + batch order every variant
            tdef = _TASKS[task]
            head = _build_head(task, tdef, int(model_cfg["d_model"]), spec, None, head_hidden=HEAD_HIDDEN,
                               next_activity_n=eval_label_spec.n_activities, pooling="trace")
            module = MultiTaskLitModule(B._CachedBackbone(), {task: head}, freeze_backbone=True,
                                        optimizer_cfg={"lr": LR, "backbone_lr": LR},
                                        d_model=int(model_cfg["d_model"]))
            dls = [DataLoader(B._CacheDataset(cache[p]), batch_size=BATCH, shuffle=(p == "train"), num_workers=0)
                   for p in ("train", "val", "test")]
            mode = "max" if tdef.higher_is_better else "min"
            mon = f"val/{task}/{tdef.metric}"
            t = B.make_trainer([EarlyStopping(monitor=mon, mode=mode, patience=PATIENCE, strict=False),
                                _KeepBestState(mon, mode)])
            t.fit(module, dls[0], dls[1])
            out[task] = float(t.test(module, dls[2], verbose=False)[0][f"test/{task}/{tdef.metric}"])
        return out

    t0 = time.perf_counter()
    base = run(graph0)
    print(f"[imp] baseline {base} ({time.perf_counter()-t0:.0f}s)", flush=True)

    res = {n: {t: [] for t in TASKS} for n in NAMES}
    for j, name in enumerate(NAMES):
        for ps in range(a.perm_seeds):
            g = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in graph0.items()}
            gen = torch.Generator().manual_seed(1000 * j + ps)
            perm = real[torch.randperm(len(real), generator=gen)]
            g["feats"][real, j] = graph0["feats"][perm, j]
            r = run(g)
            for t in TASKS:
                res[name][t].append(r[t])
        msg = " ".join("%s=%.4f" % (t, sum(res[name][t]) / len(res[name][t])) for t in TASKS)
        print(f"[imp] {name:16s} {msg}", flush=True)

    out = {"log": a.log, "perm_seeds": a.perm_seeds, "n_real_activities": int(len(real)),
           "baseline": base, "permuted": {n: {t: res[n][t] for t in TASKS} for n in NAMES},
           "elapsed_s": time.perf_counter() - t0,
           "caveat": "features are correlated; one-at-a-time permutation under-states importance"}
    print("[imp-json] " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
