#!/usr/bin/env python3
"""Amendment A2 aggregation: the six further tasks over the feature-budget ladder (r33).

    python scripts/feature_ladder_tasks_analysis.py --eval-dir outputs/feature_ladder/eval_tasks

Exploratory by A2: no primary claim, no per-task k_near threshold. Per task it reports the D2 five-log mean,
the D3 seed-wise SD (SD of the three five-log means across evaluation seeds) and the D5 paired per-log
differences to k = 15 with a t(4) 95% interval -- the same estimators scripts/feature_ladder_analysis.py
applies to next activity, which is imported here so the two reports cannot drift apart.

Units (A2: "MAE tasks are reported in raw units per log"). next_time and remaining_time are MAE on the
log1p-seconds target, remaining_count is MAE in events; none is a probability, so a five-log mean of the MAE
tasks averages quantities whose scale differs per log. Those pooled columns are still written (A2 asks for the
D2 mean per task) but carry scale_mixing = true, and the per-log columns are the ones to read.

The A1 pipeline caveat applies throughout: these come from the cached evaluator, so budgets are compared with
each other and not with the main result tables.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics as st
from pathlib import Path

from feature_ladder_analysis import KS, LOGS, ROOT, SEEDS, T4, read_jsonl

TASKS = ["next_3_activities", "next_5_activities", "future_activity_set",
         "next_time", "remaining_time", "remaining_count"]
METRIC = {"next_3_activities": "acc", "next_5_activities": "acc", "future_activity_set": "micro_f1",
          "next_time": "mae", "remaining_time": "mae", "remaining_count": "mae"}
UNIT = {"next_3_activities": "accuracy", "next_5_activities": "accuracy", "future_activity_set": "micro-F1",
        "next_time": "MAE, log1p seconds", "remaining_time": "MAE, log1p seconds",
        "remaining_count": "MAE, events"}
HIGHER_IS_BETTER = {t: METRIC[t] != "mae" for t in TASKS}


def load(eval_dir: str) -> dict[tuple[str, int, str, int], float]:
    acc = {}
    for path in sorted(glob.glob(f"{eval_dir}/fb*.jsonl")):
        k = int(Path(path).stem[2:])
        for r in read_jsonl(path):
            for task in TASKS:
                acc[(task, k, r["log"], int(r["seed"]))] = r["cached"][task]["value"]
    return acc


def report(acc: dict[tuple[str, int, str, int], float], out_dir: Path) -> list[dict]:
    missing = [(t, k, log, s) for t in TASKS for k in KS for log in LOGS for s in SEEDS
               if (t, k, log, s) not in acc]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(TASKS) * len(KS) * len(LOGS) * len(SEEDS)} runs missing, "
                         f"e.g. {missing[:5]}")

    rows = []
    for task in TASKS:
        abar = {(k, log): st.mean(acc[(task, k, log, s)] for s in SEEDS) for k in KS for log in LOGS}   # D1
        for k in KS:
            seed_means = [st.mean(acc[(task, k, log, s)] for log in LOGS) for s in SEEDS]              # D2/D3
            delta = [abar[(k, log)] - abar[(15, log)] for log in LOGS]                                 # D5
            d_mean = st.mean(delta)
            half = T4 * st.stdev(delta) / math.sqrt(len(LOGS)) if k != 15 else 0.0
            rows.append({
                "task": task, "metric": METRIC[task], "unit": UNIT[task],
                "higher_is_better": HIGHER_IS_BETTER[task], "k": k,
                "mu": st.mean(seed_means), "seedwise_sd": st.stdev(seed_means),
                "between_log_sd": st.stdev(abar[(k, log)] for log in LOGS),
                "scale_mixing": not HIGHER_IS_BETTER[task],   # MAE: mu/SD pool per-log units, read per-log
                "delta_mean": d_mean, "delta_ci_low": d_mean - half, "delta_ci_high": d_mean + half,
                "delta_excludes_zero": k != 15 and (d_mean - half) * (d_mean + half) > 0,
                **{f"delta_{log}": d for log, d in zip(LOGS, delta)},
                **{f"value_{log}": abar[(k, log)] for log in LOGS},
                "n_logs": len(LOGS), "n_eval_seeds": len(SEEDS),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_ladder_tasks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "k", "log", "eval_seed", "value"])
        for (task, k, log, s), v in sorted(acc.items()):
            w.writerow([task, k, log, s, v])
    with open(out_dir / "feature_ladder_tasks_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "feature_ladder_tasks_summary.json").write_text(json.dumps({
        "amendment": "A2 (exploratory); protocols/feature_ladder.md",
        "caveat_A1": "cached evaluator; compare budgets within a task, not with the main result tables",
        "tasks": {t: {"metric": METRIC[t], "unit": UNIT[t], "higher_is_better": HIGHER_IS_BETTER[t]}
                  for t in TASKS},
        "k_near": None, "k_near_note": "A2 defines no per-task k_near threshold",
        "n_runs": len(acc),
    }, indent=2))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=str(ROOT / "results" / "feature_ladder_eval_tasks"))
    a = ap.parse_args()
    rows = report(load(a.eval_dir), ROOT / "results")
    for task in TASKS:
        rs = {r["k"]: r for r in rows if r["task"] == task}
        arrow = "higher better" if HIGHER_IS_BETTER[task] else "lower better"
        print(f"\n{task}  ({UNIT[task]}, {arrow})")
        print("  k :  " + " ".join(f"{k:>7d}" for k in KS))
        print("  mu:  " + " ".join(f"{rs[k]['mu']:7.4f}" for k in KS))
        print("  sd:  " + " ".join(f"{rs[k]['seedwise_sd']:7.4f}" for k in KS))
        sig = [k for k in KS if rs[k]["delta_excludes_zero"]]
        print(f"  budgets whose paired 95% interval to k=15 excludes zero: {sig if sig else 'none'}")


if __name__ == "__main__":
    main()
