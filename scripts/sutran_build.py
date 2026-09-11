"""Build SuTraN+ tensor datasets from OUR partitions and OUR canonical query set (exact-protocol adapter).

Reads <log>_splits.csv (events, our order) and <log>_queries.csv (scripts/export_queries.py) and writes, into
<out>/ (= SuTraN_Plus/<LOG>/), exactly the files TRAIN_EVAL_EQUAL_WEIGHTING.train_eval loads: the 8-tensor tuples
train/val/test_tensordataset.pt, og_caseint_{split}.pt and the six pickles, in the layout of Preprocessing/
tensor_creation.py for the non-data-aware setting (prefix cats = [activity], numerics = [ts_start, ts_prev]).

One instance per remaining_time query row (case_id, hist_start, prefix_end): prefix = events[hist_start..prefix_end]
(the most recent <= W events, as our probes see them), decoder suffix = events[prefix_end..n-1], labels = activities
prefix_end+1..n-1 then END, tt_next of each suffix event (0 at the last), remaining time of each suffix event.
Time features are seconds relative to the window start (ts_start) and to the previous event in the window (ts_prev,
0 for the first). Standardization follows the repo: prefix scaler on all train prefix rows, suffix scaler on all train
decoder-suffix rows, tt_next on all train label rows, rtime on the first label row of each instance; population std.
Activity codes: train activities in order of first appearance -> 0..n-1; val/test-only activities -> one OOV code n.
A side table rows_{split}.csv maps tensor row -> (case_id, hist_start, prefix_end, n_events) for our metrics.
Usage: python scripts/sutran_build.py --splits exports/helpdesk_splits.csv --queries exports/helpdesk_queries.csv --out external/SuTraN_Plus/HELPDESK --log HELPDESK
"""
from __future__ import annotations

import argparse, csv, json, os, pickle, statistics
import numpy as np
import pandas as pd
import torch

ACT = "concept:name"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--log", required=True, help="folder/pickle prefix, e.g. HELPDESK")
    ap.add_argument("--window", type=int, default=64)
    a = ap.parse_args()
    W = a.window
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.splits, dtype={"case_id": str, "activity": str})
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    except (TypeError, ValueError):
        ts = pd.to_datetime(df["timestamp"].map(pd.Timestamp), utc=True)
    if ts.isna().any():
        raise SystemExit("unparseable timestamps: %d" % int(ts.isna().sum()))
    df["t"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(np.float64)  # seconds, resolution-safe (pandas 3 parses to [us])  # seconds
    cases: dict[str, dict[str, tuple]] = {"train": {}, "val": {}, "test": {}}
    for (sp, cid), g in df.groupby(["split", "case_id"], sort=False):  # file order = our trace order
        cases[sp][cid] = (g["activity"].tolist(), g["t"].to_numpy(dtype=np.float64))

    # activity codes: train first-appearance order, OOV for anything else
    code: dict[str, int] = {}
    for cid, (acts, _) in cases["train"].items():
        for x in acts:
            if x not in code: code[x] = len(code)
    n_train_levels = len(code)
    oov_needed = any(x not in code for sp in ("val", "test") for acts, _ in cases[sp].values() for x in acts)
    card = n_train_levels + (1 if oov_needed else 0)
    OOV = n_train_levels
    END = card + 1  # label id of END_TOKEN (tensor ids are code+1, 0 = padding)

    q = pd.read_csv(a.queries, dtype={"case_id": str, "label": str})
    q = q[q.task == "remaining_time"]  # every kept position of every case = the SuTraN instance set
    inst = {sp: [] for sp in cases}
    for r in q.itertuples(index=False):
        inst[r.split].append((r.case_id, int(r.hist_start), int(r.prefix_end)))
    for sp in cases:  # sanity: hist_start = max(0, n-W), prefix_end covers hist_start..n-1
        for cid, hs, pe in inst[sp]:
            n = len(cases[sp][cid][0]); assert hs == max(0, n - W) and hs <= pe < n, (sp, cid, hs, pe, n)
        per_case = {}
        for cid, hs, pe in inst[sp]: per_case.setdefault(cid, set()).add(pe)
        for cid, pes in per_case.items():
            n = len(cases[sp][cid][0]); assert pes == set(range(max(0, n - W), n)), (sp, cid)
        assert set(per_case) == set(cases[sp]), "query cases differ from split cases (%s)" % sp

    def raw_arrays(sp):
        """Per split: raw (unstandardized) tensors + accumulators for the scalers + side table."""
        rows = inst[sp]; N = len(rows)
        act_pref = np.zeros((N, W), np.int64); num_pref = np.zeros((N, W, 2), np.float32); pad = np.ones((N, W), bool)
        act_suf = np.zeros((N, W), np.int64); num_suf = np.full((N, W, 2), -1.0, np.float32)
        ttne = np.full((N, W, 1), -100.0, np.float32); rrt = np.full((N, W, 1), -100.0, np.float32); lab = np.zeros((N, W), np.int64)
        og = np.zeros(N, np.int64); side = []; case_int = {}
        for r, (cid, hs, pe) in enumerate(rows):
            acts, t = cases[sp][cid]; n = len(acts)
            codes = np.array([code.get(x, OOV) for x in acts], np.int64) + 1
            tw = t[hs:]  # window
            ts_start = tw - tw[0]; ts_prev = np.concatenate([[0.0], np.diff(tw)])
            k = pe - hs + 1  # prefix length within the window
            act_pref[r, :k] = codes[hs:pe + 1]; num_pref[r, :k, 0] = ts_start[:k]; num_pref[r, :k, 1] = ts_prev[:k]; pad[r, :k] = False
            L = n - pe  # decoder suffix / label length (events pe..n-1 ; labels pe+1..n-1 + END)
            act_suf[r, :L] = codes[pe:]; num_suf[r, :L, 0] = ts_start[pe - hs:]; num_suf[r, :L, 1] = ts_prev[pe - hs:]
            tt_next = np.concatenate([np.diff(t[pe:]), [0.0]])  # tt_next of events pe..n-1 (0 at the last)
            ttne[r, :L, 0] = tt_next; rrt[r, :L, 0] = t[-1] - t[pe:]
            lab[r, :L - 1] = codes[pe + 1:]; lab[r, L - 1] = END
            og[r] = case_int.setdefault(cid, len(case_int)); side.append((r, cid, hs, pe, n))
        return dict(act_pref=act_pref, num_pref=num_pref, pad=pad, act_suf=act_suf, num_suf=num_suf, ttne=ttne, rrt=rrt, lab=lab, og=og, side=side)

    data = {sp: raw_arrays(sp) for sp in ("train", "val", "test")}
    tr = data["train"]
    def stats(x):  # population mean/std as sklearn StandardScaler
        m = float(np.mean(x)); s = float(np.sqrt(np.mean((x - m) ** 2))); return m, (s if s > 0 else 1.0)
    pm = ~tr["pad"]
    m_pref = [stats(tr["num_pref"][:, :, j][pm]) for j in range(2)]
    sm = tr["num_suf"][:, :, 0] != -1.0
    m_suf = [stats(tr["num_suf"][:, :, j][sm]) for j in range(2)]
    lm = tr["ttne"][:, :, 0] != -100.0
    m_ttne = stats(tr["ttne"][:, :, 0][lm]); m_rrt = stats(tr["rrt"][:, 0, 0])
    means = {"prefix_df": [m_pref[0][0], m_pref[1][0]], "suffix_df": [m_suf[0][0], m_suf[1][0]], "timeLabel_df": [m_ttne[0], m_rrt[0]]}
    stds = {"prefix_df": [m_pref[0][1], m_pref[1][1]], "suffix_df": [m_suf[0][1], m_suf[1][1]], "timeLabel_df": [m_ttne[1], m_rrt[1]]}

    for sp, d in data.items():
        pmask = ~d["pad"]
        for j in range(2): d["num_pref"][:, :, j][pmask] = (d["num_pref"][:, :, j][pmask] - means["prefix_df"][j]) / stds["prefix_df"][j]
        smask = d["num_suf"][:, :, 0] != -1.0
        for j in range(2): d["num_suf"][:, :, j][smask] = (d["num_suf"][:, :, j][smask] - means["suffix_df"][j]) / stds["suffix_df"][j]
        lmask = d["ttne"][:, :, 0] != -100.0
        d["ttne"][:, :, 0][lmask] = (d["ttne"][:, :, 0][lmask] - means["timeLabel_df"][0]) / stds["timeLabel_df"][0]
        d["rrt"][:, :, 0][lmask] = (d["rrt"][:, :, 0][lmask] - means["timeLabel_df"][1]) / stds["timeLabel_df"][1]
        tup = (torch.from_numpy(d["act_pref"]), torch.from_numpy(d["num_pref"]), torch.from_numpy(d["pad"]), torch.from_numpy(d["act_suf"]),
               torch.from_numpy(d["num_suf"]), torch.from_numpy(d["ttne"]), torch.from_numpy(d["rrt"]), torch.from_numpy(d["lab"]))
        torch.save(tup, os.path.join(a.out, "%s_tensordataset.pt" % sp)); torch.save(torch.from_numpy(d["og"]), os.path.join(a.out, "og_caseint_%s.pt" % sp))
        with open(os.path.join(a.out, "rows_%s.csv" % sp), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["row", "case_id", "hist_start", "prefix_end", "n_events"]); w.writerows(d["side"])
    P = lambda name, obj: pickle.dump(obj, open(os.path.join(a.out, "%s_%s.pkl" % (a.log, name)), "wb"))
    P("cardin_dict", {ACT: card}); P("cardin_list_prefix", [card]); P("cardin_list_suffix", [card])
    P("num_cols_dict", {"prefix_df": ["ts_start", "ts_prev"], "suffix_df": ["ts_start", "ts_prev"], "timeLabel_df": ["tt_next", "rtime"]})
    P("cat_cols_dict", {"prefix_df": [ACT], "suffix_df": [ACT], "actLabel_df": [ACT]})
    P("train_means_dict", means); P("train_std_dict", stds); P("categ_mapping", {ACT: code})
    median_caselen = int(statistics.median(min(len(acts), W) for acts, _ in cases["train"].values()))
    summary = {"log": a.log, "window": W, "cardinality": card, "oov": oov_needed, "num_activities_model": card + 2, "END_label_id": END,
               "instances": {sp: len(inst[sp]) for sp in inst}, "cases": {sp: len(cases[sp]) for sp in cases}, "median_caselen": median_caselen,
               "means": means, "stds": stds}
    json.dump(summary, open(os.path.join(a.out, "build_summary.json"), "w"), indent=1); print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
