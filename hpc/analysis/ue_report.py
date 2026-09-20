#!/usr/bin/env python3
"""Compare next-activity probes at the default vs the enlarged optimisation budget.

Old = probe.max_epochs 15 / patience 3 (what the current curves used).
New = probe.max_epochs 100 / patience 10 (the ue-longprobe jobs).
Everything else - backbone, log, split, seeds - is identical, so any change is
optimisation budget alone.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.expanduser("~/hpc_training_frozen")
LOGS = ["BPI20ID", "BPI17", "bpi13_incidents"]
BUDGETS = ["10", "30", "100"]
ARMS = [("gin15allw1", "PFM"), ("random", "Baseline"), ("scratch", "Scratch")]


def main() -> None:
    os.chdir(ROOT)
    # (log, epochs, arm, budget) -> [values over seeds]
    vals: dict[tuple, list[float]] = defaultdict(list)
    runs: dict[tuple, set] = defaultdict(set)

    for d in sorted(glob.glob("outputs/label_efficiency/*/")):
        mf, cf_ = os.path.join(d, "manifest.json"), os.path.join(d, "curves.csv")
        if not (os.path.exists(mf) and os.path.exists(cf_)):
            continue
        try:
            man = json.load(open(mf))
        except Exception:
            continue
        if not man.get("completed_at"):
            continue
        cfg = man.get("config", {})
        log = cfg.get("eval_dataset")
        if log not in LOGS:
            continue
        ep = int((cfg.get("probe") or {}).get("max_epochs", 15))
        if ep not in (15, 100):
            continue
        for r in csv.DictReader(open(cf_)):
            if r["task"] != "next_activity" or r["n_labels"] not in BUDGETS:
                continue
            vals[(log, ep, r["backbone_alias"], r["n_labels"])].append(float(r["value"]))
            runs[(log, ep)].add(os.path.basename(d.rstrip("/")))

    def mean(k):
        v = vals.get(k)
        return sum(v) / len(v) if v else None

    print("next-activity accuracy (higher is better), mean over seeds")
    print("old = max_epochs 15 / patience 3   |   new = max_epochs 100 / patience 10\n")
    for log in LOGS:
        if not runs.get((log, 100)):
            print("%-16s  NEW RUN NOT PRESENT YET\n" % log)
            continue
        print("== %s" % log)
        print("  %-9s %-24s %-24s %-24s" % ("", "10 labels", "30 labels", "100 labels"))
        print("  %-9s %s" % ("", "  old     new     delta " * 3))
        for alias, label in ARMS:
            cells = ""
            for b in BUDGETS:
                o, n = mean((log, 15, alias, b)), mean((log, 100, alias, b))
                if o is None or n is None:
                    cells += "%-24s" % "    --      --      -- "
                else:
                    cells += "%7.3f %7.3f %+7.3f   " % (o, n, n - o)
            print("  %-9s %s" % (label, cells))
        print()


if __name__ == "__main__":
    main()
