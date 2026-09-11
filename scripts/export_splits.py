"""Export the evaluation partitions of a log exactly as label_efficiency.py builds them.

Writes one CSV per log with columns  case_id, activity, timestamp, split  (split in train/val/test), using the
same reader options (max_traces cap), control-flow stripping, min_trace_len filter and the single
chronological 70/15/15 split (seed 0) as the downstream evaluation, so an external model (SuTraN+,
ProcessTransformer, ...) can be trained and tested on identical cases.

Usage:
  python scripts/export_splits.py --log helpdesk --path data/raw/helpdesk.csv --out exports/helpdesk_splits.csv
  python scripts/export_splits.py --log mimic_transfer --path data/raw/mimic_transfers.csv --max-traces 5000 --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pm_foundation.data.preprocessing import SplitStrategy, build_traces, split_log
from pm_foundation.data.readers import get_reader
from pm_foundation.evaluation.label_efficiency import _strip_to_control_flow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="log name used in the paper (for the summary only)")
    ap.add_argument("--path", required=True)
    ap.add_argument("--format", default=None, help="csv|xes (default: from the file suffix)")
    ap.add_argument("--max-traces", type=int, default=None, help="reader cap, as in the eval config (MIMIC: 5000)")
    ap.add_argument("--max-trace-len", type=int, default=None, help="exclude cases longer than this (evaluate.max_trace_len)")
    ap.add_argument("--min-trace-len", type=int, default=2)
    ap.add_argument("--split", default="0.70,0.15,0.15")
    ap.add_argument("--no-strip", action="store_true", help="keep attributes (eval default strips to control flow)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="10,30,100,300,1000,5000", help="label budgets; case ids of each (seed, budget) sample are written to <out>.budgets.json")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    fmt = a.format or Path(a.path).suffix.lstrip(".")
    reader = get_reader(fmt, **({"max_traces": a.max_traces} if a.max_traces else {}))
    log = reader.read(a.path)
    if not a.no_strip:
        log = _strip_to_control_flow(log)
    built = build_traces(log, min_trace_len=a.min_trace_len, max_trace_len=a.max_trace_len)
    ratios = tuple(float(x) for x in a.split.split(","))
    splits = split_log(built, SplitStrategy.TEMPORAL, ratios, seed=0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ev = {"train": 0, "val": 0, "test": 0}
    acts: set[str] = set()
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "activity", "timestamp", "split"])
        for name in ("train", "val", "test"):
            for tr in getattr(splits, name).traces:
                for ev in tr.events:
                    w.writerow([tr.case_id, ev.activity, ev.timestamp.isoformat(), name])
                    n_ev[name] += 1
                    acts.add(ev.activity)
    summary = {
        "log": a.log, "path": a.path, "max_traces": a.max_traces, "min_trace_len": a.min_trace_len, "max_trace_len": a.max_trace_len,
        "split": ratios, "strategy": "temporal_by_case_start", "cases": {k: len(getattr(splits, k).traces) for k in ("train", "val", "test")},
        "events": n_ev, "activities": len(acts), "out": str(out),
    }
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
    # the exact cases each (seed, budget) probe trains on: same sampler as label_efficiency._subsample
    from pm_foundation.evaluation.label_efficiency import _subsample
    train = list(splits.train.traces)
    budgets = {}
    for seed in (int(x) for x in a.seeds.split(",")):
        for n in (int(x) for x in a.budgets.split(",")):
            budgets["%d/%d" % (seed, n)] = [t.case_id for t in _subsample(train, n, seed)]
    Path(str(out) + ".budgets.json").write_text(json.dumps(budgets))
    summary["budgets"] = {k: len(v) for k, v in budgets.items()}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
