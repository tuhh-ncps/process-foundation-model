"""Export the EXACT per-event query set that label_efficiency.py scores, so an external model (FM-v2)
is evaluated on identical (case_id, prefix_end) pairs with identical labels.

Mirrors the evaluation step by step: reader (max_traces cap) -> strip to control flow ->
build_traces(min_trace_len) -> chronological 70/15/15 split (seed 0) -> every trace truncated to its
most recent ``max_seq_len`` events (FeatureSpec.max_seq_len = model.max_seq_len = 64 in every run) ->
per-event targets exactly as ``event_targets``:
  next_activity : every kept position except the last one; label = the next event's activity
  remaining_time: every kept position (the last one has remaining time 0); label = seconds to the last event
Each exported row is verified against ``SupervisedTraceDataset``/``event_targets`` (position count and label).

Columns: task, split, case_id, hist_start, prefix_end, label
  hist_start = index (in the case's event order in <log>_splits.csv) of the first event a model may see
  prefix_end = index of the last prefix event (inclusive); the prediction is made right after it
The train split is exported too: it is the labelled pool an external model may use at full budget (same
truncation as the cases our head trains on).
Usage:
  python scripts/export_queries.py --log helpdesk --path data/raw/helpdesk.csv --splits-csv exports/helpdesk_splits.csv --out exports/helpdesk_queries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

from pm_foundation.data.dataset import NEXT_ACTIVITY_IGNORE_INDEX, SupervisedTraceDataset
from pm_foundation.data.preprocessing import SplitStrategy, build_traces, fit_feature_spec, split_log
from pm_foundation.data.readers import get_reader
from pm_foundation.evaluation.label_efficiency import _strip_to_control_flow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True); ap.add_argument("--path", required=True)
    ap.add_argument("--format", default=None); ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--max-trace-len", type=int, default=None, help="exclude cases longer than this (evaluate.max_trace_len)")
    ap.add_argument("--min-trace-len", type=int, default=2, help="label_efficiency default (config min_trace_len)")
    ap.add_argument("--split", default="0.70,0.15,0.15"); ap.add_argument("--no-strip", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=64, help="model.max_seq_len of the evaluation runs (64)")
    ap.add_argument("--splits-csv", default=None, help="<log>_splits.csv from export_splits.py: verify case/event order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fmt = a.format or Path(a.path).suffix.lstrip(".")
    reader = get_reader(fmt, **({"max_traces": a.max_traces} if a.max_traces else {}))
    log = reader.read(a.path)
    if not a.no_strip:
        log = _strip_to_control_flow(log)
    built = build_traces(log, min_trace_len=a.min_trace_len, max_trace_len=a.max_trace_len)
    ratios = tuple(float(x) for x in a.split.split(","))
    splits = split_log(built, SplitStrategy.TEMPORAL, ratios, seed=0)
    spec = fit_feature_spec(splits.train, max_seq_len=a.max_seq_len)
    vocab = spec.activity_vocab

    if a.splits_csv:  # the event order the external model will index into must be OUR order
        seqs: dict[tuple[str, str], list[str]] = OrderedDict()
        with open(a.splits_csv) as f:
            for r in csv.DictReader(f):
                seqs.setdefault((r["split"], str(r["case_id"])), []).append(r["activity"])
        for name in ("train", "val", "test"):
            part = getattr(splits, name)
            ids = [k[1] for k in seqs if k[0] == name]
            assert ids == [str(t.case_id) for t in part.traces], "case order differs from %s (%s)" % (a.splits_csv, name)
            for t in part.traces:
                assert seqs[(name, str(t.case_id))] == [e.activity for e in t.events], "event order differs for case %s" % t.case_id
        print("verified case and event order against", a.splits_csv)

    rows = []
    stats: dict[str, dict[str, int]] = {}
    for name in ("train", "val", "test"):
        part = getattr(splits, name)
        ds = SupervisedTraceDataset(part, spec, target_activity_vocab=vocab)
        n_na = n_rt = n_trunc = 0
        for j, tr in enumerate(part.traces):
            ev = tr.events
            n = len(ev)
            start = max(0, n - a.max_seq_len) if a.max_seq_len else 0
            n_trunc += start > 0
            item = ds[j]
            na, rt = item["next_activity"], item["remaining_time"]
            assert len(na) == n - start and len(rt) == n - start, (tr.case_id, n, start, len(na))
            for i in range(start, n):
                rem = max((ev[-1].timestamp - ev[i].timestamp).total_seconds(), 0.0)
                assert abs(math.log1p(rem) - float(rt[i - start])) < 1e-3, (tr.case_id, i)
                rows.append(("remaining_time", name, tr.case_id, start, i, "%.3f" % rem)); n_rt += 1
                if i < n - 1:
                    assert int(na[i - start]) == vocab.encode(ev[i + 1].activity), (tr.case_id, i)
                    rows.append(("next_activity", name, tr.case_id, start, i, ev[i + 1].activity)); n_na += 1
                else:
                    assert int(na[i - start]) == NEXT_ACTIVITY_IGNORE_INDEX
        stats[name] = {"cases": len(part.traces), "truncated_cases": n_trunc, "next_activity": n_na, "remaining_time": n_rt}

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "split", "case_id", "hist_start", "prefix_end", "label"]); w.writerows(rows)
    summary = {"log": a.log, "path": a.path, "max_traces": a.max_traces, "min_trace_len": a.min_trace_len, "max_trace_len": a.max_trace_len, "split": ratios,
               "max_seq_len": a.max_seq_len, "verified_against": "SupervisedTraceDataset/event_targets", "queries": stats, "out": str(out)}
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
