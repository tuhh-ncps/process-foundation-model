"""Evaluate Berti & van der Aalst's in-context FM (events-transf, FM-v2 retrieval mode) under OUR protocol.

Inputs are the partitions exported by scripts/export_splits.py (chronological 70/15/15 + per-(seed, budget)
sampled train cases). For each (seed, budget) the support pool is every prefix of the sampled cases; every
(case_id, prefix_end) pair exported by scripts/export_queries.py from OUR evaluation is a query; each query retrieves its top-k supports by cosine similarity of the
MoE-averaged prefix embedding and is predicted with (a) their prototypical heads, aggregated over experts
exactly as MoEModel._aggregate_outputs does, and (b) their similarity-weighted kNN ("foundation_knn").
Next activity -> accuracy pooled over test prefixes; remaining time -> MAE in days (their target is
sqrt(hours); we invert it). Their prefix window (last 10 events) is kept; prefix positions start at 1 event
(our convention) unless --min-prefix 2 (their get_task_data convention).

Usage:
  python scripts/fmv2_eval.py --repo /path/to/events-transf --checkpoint-dir /path/to/ckpt \
      --splits exports/helpdesk_splits.csv --log helpdesk --out fmv2_helpdesk.csv [--mask-names] [--k 5 10 20]
  --random-init  builds the model from config with random weights (smoke test / null reference).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


def load_repo(repo: str):
    sys.path.insert(0, repo)
    # data_generator imports pm4py + sentence_transformers at module top; we never need it.
    from components.moe_model import MoEModel  # noqa: E402
    from time_transf import inverse_transform_time, transform_time  # noqa: E402
    return MoEModel, transform_time, inverse_transform_time


def build_model(MoEModel, cfg: dict, char_to_id: dict, device):
    kw = dict(num_experts=cfg.get("moe_settings", {}).get("num_experts", 1), strategy=cfg["embedding_strategy"],
              num_feat_dim=cfg["num_numerical_features"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
              n_layers=cfg["n_layers"], dropout=cfg.get("dropout", 0.1))
    if cfg["embedding_strategy"] == "pretrained":
        kw["embedding_dim"] = cfg["pretrained_settings"]["embedding_dim"]
    else:
        kw.update(char_vocab_size=len(char_to_id), char_embedding_dim=cfg["learned_settings"]["char_embedding_dim"],
                  char_cnn_output_dim=cfg["learned_settings"]["char_cnn_output_dim"])
    m = MoEModel(**kw)
    if cfg["embedding_strategy"] == "learned":
        m.set_char_vocab(char_to_id)
    return m.to(device).eval()


def load_checkpoint(repo: str, ckpt_dir: str, device, random_init: bool):
    MoEModel, tt, itt = load_repo(repo)
    if random_init:
        from config import CONFIG  # noqa: E402
        cfg = dict(CONFIG)
        chars = {c: i + 2 for i, c in enumerate(sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.:/()")))}
        chars["<PAD>"] = 0; chars["<UNK>"] = 1
        return build_model(MoEModel, cfg, chars, device), cfg, tt, itt
    cfg = torch.load(os.path.join(ckpt_dir, "training_config.pth"), map_location="cpu", weights_only=False)
    art = torch.load(os.path.join(ckpt_dir, "training_artifacts.pth"), map_location="cpu", weights_only=False)
    char_to_id = art.get("char_to_id", {})
    m = build_model(MoEModel, cfg, char_to_id, device)
    # latest epoch checkpoint, as their load_model_weights does
    files = sorted((f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pth")),
                   key=lambda f: int(f[len("model_epoch_"):-4]))
    if not files:
        raise SystemExit("no model_epoch_*.pth in " + ckpt_dir)
    sd = torch.load(os.path.join(ckpt_dir, files[-1]), map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("loaded", files[-1], "| missing:", len(missing), "unexpected:", len(unexpected))
    if missing:
        print("  missing keys (first 5):", missing[:5])
    return m, cfg, tt, itt


# ---------------- our partitions -> their event dicts ----------------
def read_split(path: str, mask_names: bool):
    """<log>_splits.csv -> {split: {case_id: [event dict, ...]}} in FILE order (= our trace order; never re-sorted,
    so prefix_end indices from export_queries.py address the same events)."""
    df = pd.read_csv(path, dtype={"case_id": str, "activity": str})
    # ISO strings with mixed DST offsets and with/without fractional seconds (BPI17 mixes both) -> UTC instants
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    except (TypeError, ValueError):  # pandas < 2.0: no ISO8601 format keyword
        ts = pd.to_datetime(df["timestamp"].map(pd.Timestamp), utc=True)
    df["timestamp"] = ts.dt.tz_convert(None)
    if df["timestamp"].isna().any():
        raise SystemExit("unparseable timestamps in %s: %d rows" % (path, int(df["timestamp"].isna().sum())))
    if mask_names:  # opaque, bijective renaming: the model sees no lexical content
        names = sorted(df["activity"].astype(str).unique())
        ren = {a: "A%03d" % i for i, a in enumerate(names)}
        df["activity"] = df["activity"].astype(str).map(ren)
    traces = {"train": {}, "val": {}, "test": {}}
    for (split, cid), g in df.groupby(["split", "case_id"], sort=False):
        t0 = g.iloc[0]["timestamp"]; prev = t0; tr = []
        for _, e in g.iterrows():
            ts = e["timestamp"]
            tr.append({"case_id": cid, "activity_name": str(e["activity"]), "resource_name": "Unknown", "cost": 0.0,
                       "time_from_start": (ts - t0).total_seconds(), "time_from_previous": (ts - prev).total_seconds(),
                       "timestamp": ts.timestamp()})
            prev = ts
        traces[split][cid] = tr
    return traces


def make_tasks(by_id: dict, qdf, task: str, act2id: dict, transform_time, max_seq_len: int):
    """(prefix, label, case_id) for every exported query row. The prefix is events[hist_start:prefix_end+1]
    (the same history window our model gets), then their own last-``max_seq_len`` window. Labels are
    recomputed from the events and asserted against the exported label."""
    tasks = []
    for cid, hs, pe, lab in zip(qdf.case_id, qdf.hist_start, qdf.prefix_end, qdf.label):
        tr = by_id[cid]
        prefix = tr[hs:pe + 1]
        if len(prefix) > max_seq_len:
            prefix = prefix[-max_seq_len:]
        if task == "classification":
            nxt = tr[pe + 1]["activity_name"]
            tasks.append((prefix, act2id[nxt], cid))
        else:
            rem_s = max(tr[-1]["timestamp"] - tr[pe]["timestamp"], 0.0)
            if abs(rem_s - float(lab)) > 1.0:
                raise SystemExit("remaining-time label mismatch for case %s prefix_end %d: %.1f vs exported %s" % (cid, pe, rem_s, lab))
            tasks.append((prefix, float(transform_time(rem_s / 3600.0)), cid))
    return tasks


@torch.no_grad()
def embed(model, tasks, batch_size, device):
    """Model-level (MoE-averaged) and per-expert embeddings."""
    outs, per_exp = [], [[] for _ in model.experts]
    for i in range(0, len(tasks), batch_size):
        seqs = [t[0] for t in tasks[i:i + batch_size]]
        es = [ex._process_batch(seqs) for ex in model.experts]
        for j, e in enumerate(es):
            per_exp[j].append(e.detach())
        outs.append(torch.stack(es).mean(0).detach())
    return torch.cat(outs), [torch.cat(p) for p in per_exp]


def l2n(x):
    return torch.nn.functional.normalize(x, dim=-1)


@torch.no_grad()
def predict(model, task, k, q_feats, q_exp, s_feats, s_exp, s_labels, itt, batch=512):
    """Returns dict predictor -> predictions (tensor over queries)."""
    device = q_feats.device
    sn = l2n(s_feats)
    preds = {"proto_head": [], "foundation_knn": []}
    n_s = s_feats.shape[0]
    kk = min(k, n_s)
    for i in range(0, q_feats.shape[0], batch):
        qb = l2n(q_feats[i:i + batch])
        sims = qb @ sn.T                                     # (b, n_s)
        top_sim, top_idx = sims.topk(kk, dim=1)              # (b, k)
        lab = s_labels[top_idx]                              # (b, k)
        # ---- foundation_knn: similarity-weighted vote / average over retrieved supports ----
        if task == "classification":
            w = top_sim.clamp_min(0) + 1e-8
            out = []
            for r in range(lab.shape[0]):
                u, inv = torch.unique(lab[r], return_inverse=True)
                score = torch.zeros(u.numel(), device=device).scatter_add_(0, inv, w[r])
                out.append(u[score.argmax()])
            preds["foundation_knn"].append(torch.stack(out))
        else:
            w = top_sim.clamp_min(0) + 1e-8
            preds["foundation_knn"].append((w * lab).sum(1) / w.sum(1))
        # ---- proto_head: each expert's prototypical head on its own features, MoE aggregation ----
        outs = []
        for r in range(qb.shape[0]):
            idx = top_idx[r]
            exp_outs = []
            for e, ex in enumerate(model.experts):
                sf, qf = s_exp[e][idx], q_exp[e][i + r:i + r + 1]
                if task == "classification":
                    logits, classes, conf = ex.proto_head.forward_classification(sf, lab[r], qf)
                    if logits is None:
                        continue
                    exp_outs.append((logits, classes, conf))
                else:
                    p, conf = ex.proto_head.forward_regression(sf, lab[r], qf)
                    exp_outs.append((p, conf))
            if not exp_outs:
                outs.append(preds["foundation_knn"][-1][r]); continue
            if task == "classification":
                # experts may see the same retrieved labels, so class sets coincide; sum confidences per class
                classes = exp_outs[0][1]
                summed = torch.zeros(classes.numel(), device=device)
                for logits, cl, conf in exp_outs:
                    prob = torch.softmax(logits.view(-1), dim=-1) if logits.numel() == classes.numel() else torch.zeros_like(summed)
                    summed += prob
                outs.append(classes[summed.argmax()])
            else:
                ps = torch.stack([o[0].view(-1)[0] for o in exp_outs]); cs = torch.stack([o[1].view(-1)[0] for o in exp_outs])
                outs.append((ps * cs).sum() / cs.sum().clamp_min(1e-8))
        preds["proto_head"].append(torch.stack(outs))
    return {p: torch.cat(v) for p, v in preds.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--splits", required=True, help="<log>_splits.csv (events, our order)")
    ap.add_argument("--queries", required=True, help="<log>_queries.csv from scripts/export_queries.py")
    ap.add_argument("--log", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="next_activity,remaining_time")
    ap.add_argument("--query-splits", default="val,test", help="which exported splits are scored (val: model selection)")
    ap.add_argument("--val-max-queries", type=int, default=50000, help="cap on validation queries (fixed RNG); 0 = all")
    ap.add_argument("--budgets", default="all"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--mask-names", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=10); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, cfg, tt, itt = load_checkpoint(a.repo, a.checkpoint_dir, dev, a.random_init)
    print("model params: %s" % f"{sum(p.numel() for p in model.parameters()):,}")
    traces = read_split(a.splits, a.mask_names)
    qall = pd.read_csv(a.queries, dtype={"case_id": str, "label": str})
    if a.mask_names:  # the next-activity labels are names too: apply the same renaming
        names = sorted(pd.read_csv(a.splits, dtype={"activity": str})["activity"].astype(str).unique())
        ren = {x: "A%03d" % i for i, x in enumerate(names)}
        na = qall.task == "next_activity"; qall.loc[na, "label"] = qall.loc[na, "label"].map(ren)
    for sp in ("train", "val", "test"):
        missing = set(qall[qall.split == sp].case_id) - set(traces[sp])
        if missing:
            raise SystemExit("%d query cases of split %s not in %s" % (len(missing), sp, a.splits))
    budgets = json.load(open(a.splits + ".budgets.json")) if os.path.exists(a.splits + ".budgets.json") else {}
    # label space: activities of the training partition (+ unseen val/test activities as never-retrievable extra ids)
    acts = sorted({e["activity_name"] for tr in traces["train"].values() for e in tr})
    act2id = {a_: i for i, a_ in enumerate(acts)}
    for sp in ("val", "test"):
        for tr in traces[sp].values():
            for e in tr:
                act2id.setdefault(e["activity_name"], len(act2id))
    rows = []
    for tname in a.tasks.split(","):
        task = "classification" if tname == "next_activity" else "regression"
        q = {}
        for sp in a.query_splits.split(","):
            qdf = qall[(qall.task == tname) & (qall.split == sp)].reset_index(drop=True)
            n_total = len(qdf)
            if sp == "val" and a.val_max_queries and len(qdf) > a.val_max_queries:
                qdf = qdf.sample(n=a.val_max_queries, random_state=0).sort_index().reset_index(drop=True)
            q_tasks = make_tasks(traces[sp], qdf, task, act2id, tt, a.max_seq_len)
            t0 = time.time(); q_feats, q_exp = embed(model, q_tasks, a.batch_size, dev)
            q_lab = torch.tensor([t[1] for t in q_tasks], device=dev)
            q[sp] = (q_feats, q_exp, q_lab, len(q_tasks), n_total)
            print("[%s] %s: %d %s queries (of %d) embedded in %.0fs" % (a.log, tname, len(q_tasks), sp, n_total, time.time() - t0), flush=True)
        pool_df = qall[(qall.task == tname) & (qall.split == "train")].reset_index(drop=True)
        for seed in (int(s) for s in a.seeds.split(",")):
            for b in a.budgets.split(","):
                if b == "all":
                    ids = list(traces["train"].keys())
                else:
                    ids = [str(c) for c in budgets.get("%s/%s" % (seed, b), [])]
                    if not ids:
                        print("  no sampled cases for", seed, b); continue
                sup_df = pool_df[pool_df.case_id.isin(set(ids))]
                sup = make_tasks(traces["train"], sup_df, task, act2id, tt, a.max_seq_len)
                if len(sup) < 2:
                    continue
                t0 = time.time(); s_feats, s_exp = embed(model, sup, a.batch_size, dev)
                s_lab = torch.tensor([t[1] for t in sup], device=dev)
                print("  seed %d budget %-5s pool=%d prefixes from %d cases embedded in %.0fs" % (seed, b, len(sup), len(ids), time.time() - t0), flush=True)
                for k in a.k:
                    for sp, (q_feats, q_exp, q_lab, nq, n_total) in q.items():
                        t0 = time.time()
                        pr = predict(model, task, k, q_feats, q_exp, s_feats, s_exp, s_lab, itt)
                        for name, p in pr.items():
                            if task == "classification":
                                val = float((p.long() == q_lab.long()).float().mean())
                            else:
                                hours = itt(p.cpu().numpy()); true_h = itt(q_lab.cpu().numpy())
                                val = float(np.mean(np.abs(hours - true_h)) / 24.0)
                            rows.append([a.log, tname, name, k, seed, b, sp, len(ids), len(sup), nq, val])
                        print("    k=%-3d %-4s %s  (%.0fs)" % (k, sp, "  ".join("%s=%.3f" % (r[2][:5], r[10]) for r in rows[-2:]), time.time() - t0), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log", "task", "predictor", "k", "seed", "n_labels", "split", "n_cases", "n_support_prefixes", "n_queries", "value"]); w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
