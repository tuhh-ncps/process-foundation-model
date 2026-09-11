"""Our seven per-event metrics from SuTraN+'s saved per-prefix test predictions (exact-protocol adapter, part 2).

Inputs: the TEST_SET_RESULTS folder of one run (suffix_acts_decoded.pt (N,W) greedy ids incl. END, suffix_ttne_preds.pt
(N,W) standardized, rrt_pred.pt (N,) standardized, suf_len.pt), the data folder built by scripts/sutran_build.py
(rows_test.csv, <LOG>_categ_mapping.pkl, <LOG>_train_means_dict.pkl / _train_std_dict.pkl, build_summary.json) and
<log>_splits.csv (events, for the targets).  Row r of every prediction tensor is the instance rows_test.csv[r].
Definitions mirror our heads: next activity = first decoded token; next-K = first K tokens, per-step micro accuracy over
(row, step) pairs that have a target; future set = decoded activities before the first END vs the true suffix set,
micro-F1 pooled over (row, activity); remaining count = number of decoded activities before END (W-1 if END never
comes, SuTraN's own convention); next time = first time-to-next-event, de-standardized and clamped at 0, MAE in days
over rows with a successor; remaining time = the remaining-runtime head, same de-standardization, MAE in days over all
rows. Rows follow our query set: remaining_time/count/future set on every kept position, the others on positions
with a successor.  Usage: python scripts/sutran_metrics.py --results <...>/TEST_SET_RESULTS --data <SuTraN_Plus>/<LOG> --log-prefix <LOG> --splits exports/<log>_splits.csv --log <paper name> --seed 1 --out results.csv
"""
from __future__ import annotations

import argparse, csv, json, os, pickle
import numpy as np
import pandas as pd
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--data", required=True); ap.add_argument("--log-prefix", required=True)
    ap.add_argument("--splits", required=True); ap.add_argument("--log", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.data, "build_summary.json"))); W = summ["window"]; END = summ["END_label_id"]; OOV = summ["cardinality"] - 1 if summ["oov"] else None
    code = pickle.load(open(os.path.join(a.data, "%s_categ_mapping.pkl" % a.log_prefix), "rb"))["concept:name"]
    means = pickle.load(open(os.path.join(a.data, "%s_train_means_dict.pkl" % a.log_prefix), "rb")); stds = pickle.load(open(os.path.join(a.data, "%s_train_std_dict.pkl" % a.log_prefix), "rb"))
    m_ttne, s_ttne = means["timeLabel_df"][0], stds["timeLabel_df"][0]; m_rrt, s_rrt = means["timeLabel_df"][1], stds["timeLabel_df"][1]
    rows = pd.read_csv(os.path.join(a.data, "rows_test.csv"), dtype={"case_id": str})
    dec = torch.load(os.path.join(a.results, "suffix_acts_decoded.pt"), map_location="cpu").numpy()
    ttne = torch.load(os.path.join(a.results, "suffix_ttne_preds.pt"), map_location="cpu").numpy()
    rrt = torch.load(os.path.join(a.results, "rrt_pred.pt"), map_location="cpu").numpy()
    suf_len = torch.load(os.path.join(a.results, "suf_len.pt"), map_location="cpu").numpy()
    N = len(rows); assert dec.shape == (N, W) and ttne.shape == (N, W) and rrt.shape == (N,), (dec.shape, ttne.shape, rrt.shape, N)

    df = pd.read_csv(a.splits, dtype={"case_id": str, "activity": str}); df = df[df.split == "test"]
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    except (TypeError, ValueError):
        ts = pd.to_datetime(df["timestamp"].map(pd.Timestamp), utc=True)
    df["t"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(np.float64)  # seconds, resolution-safe (pandas 3 parses to [us])
    cases = {cid: (np.array([code.get(x, OOV if OOV is not None else -1) for x in g["activity"]], np.int64) + 1, g["t"].to_numpy(np.float64)) for cid, g in df.groupby("case_id", sort=False)}

    acc1 = []; accK = {3: [], 5: []}; tp = fp = fn = 0; cnt_err = []; nt_err = []; rt_err = []
    for r in rows.itertuples(index=False):
        codes, t = cases[r.case_id]; n = len(codes); pe = r.prefix_end
        assert n == r.n_events and suf_len[r.row] == n - pe, ("row alignment", r, n, suf_len[r.row])
        d = dec[r.row]
        ends = np.nonzero(d == END)[0]; pred_end = int(ends[0]) if len(ends) else W - 1  # SuTraN: artificial END at the last step
        future = codes[pe + 1:]  # true future activities (tensor ids)
        if pe <= n - 2:
            acc1.append(float(d[0] == future[0]))
            nt_err.append(abs(max(ttne[r.row, 0] * s_ttne + m_ttne, 0.0) - (t[pe + 1] - t[pe])) / 86400.0)
            for K in (3, 5):
                for j in range(min(K, len(future))):
                    accK[K].append(float(j < pred_end and d[j] == future[j]))
        pred_set = set(int(x) for x in d[:pred_end] if x != END and x != 0); true_set = set(int(x) for x in future)
        tp += len(pred_set & true_set); fp += len(pred_set - true_set); fn += len(true_set - pred_set)
        cnt_err.append(abs(pred_end - (n - 1 - pe)))
        rt_err.append(abs(max(rrt[r.row] * s_rrt + m_rrt, 0.0) - (t[-1] - t[pe])) / 86400.0)
    out = [("next_activity", len(acc1), float(np.mean(acc1))), ("next_3_activities", len(accK[3]), float(np.mean(accK[3]))),
           ("next_5_activities", len(accK[5]), float(np.mean(accK[5]))), ("future_activity_set", N, 2 * tp / max(2 * tp + fp + fn, 1)),
           ("next_time", len(nt_err), float(np.mean(nt_err))), ("remaining_time", N, float(np.mean(rt_err))), ("remaining_count", N, float(np.mean(cnt_err)))]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["log", "seed", "task", "n_queries", "value"])
        for task, n_q, v in out: w.writerow([a.log, a.seed, task, n_q, "%.6f" % v])
    print("%s seed %d: " % (a.log, a.seed) + "  ".join("%s=%.3f(n=%d)" % (t, v, n_q) for t, n_q, v in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
