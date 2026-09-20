"""Raw-log statistics for the dataset statistics table.

Recomputes the columns already in docs/datasets_table.tex (#Cases, #Events, #Act., median/max
trace length, #Variants) as a CORRECTNESS CHECK, and adds the two new temporal columns:
  * med gap  = median over all consecutive event pairs within a case of (t_{i+1} - t_i), in days
  * med dur  = median over cases of (t_last - t_first), in days
Events are sorted by timestamp within each case first, so gaps are never negative.
Statistics are measured on the RAW logs (activity = concept:name), before prefix extraction or
trace-length filtering - the same convention as the existing table.
Streaming XES parse (iterparse, trace subtree cleared after each case) so the 1.9 GB BPI18 log
is read with flat memory. Usage: python scripts/log_stats.py [name ...]   (run from the repository root)
"""
import csv, sys, os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "results", "log_stats.csv")
LOGS = [  # display name, file, split ("pre" | "eval" | "both")
    ("BPI12", "BPI12.xes", "both"), ("BPI17", "BPI17.xes", "eval"), ("BPI19", "BPI19.xes", "pre"),
    ("BPI18", "BPI18.xes", "pre"), ("Road Traffic", "RoadTraffic.xes", "pre"),
    ("BPI20ID", "BPI20ID.xes", "eval"), ("BPI11", "BPI11.xes", "pre"),
    ("Hospital Billing", "HospitalBilling.xes", "pre"), ("MIMIC", "mimic_transfers.csv", "eval"),
    # MIMIC-5k is the log the paper evaluates: the first 5,000 cases, as evaluate.eval_log.max_traces=5000.
    ("MIMIC-5k", "mimic_transfers.csv", "eval", 5000),
    ("BPI13", "BPI13.xes", "eval"), ("Helpdesk", "helpdesk.csv", "eval"),
]
DAY = 86400.0


def _ts(raw):
    v = raw.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v).timestamp()


def cases_xes(path):
    """Yield (activities, epoch_seconds) per trace, streaming."""
    ctx = ET.iterparse(path, events=("end",))
    for _, el in ctx:
        if el.tag.rsplit("}", 1)[-1] != "trace":
            continue
        acts, ts = [], []
        for ev in el:
            if ev.tag.rsplit("}", 1)[-1] != "event":
                continue
            a = t = None
            for at in ev:
                k = at.get("key")
                if k == "concept:name":
                    a = at.get("value")
                elif k == "time:timestamp":
                    t = at.get("value")
            if a is not None and t is not None:
                acts.append(a); ts.append(_ts(t))
        if acts:
            yield acts, ts
        el.clear()


def cases_csv(path):
    cur, acts, ts = None, [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            cid = r["case_id"]
            if cid != cur:
                if acts:
                    yield acts, ts
                cur, acts, ts = cid, [], []
            acts.append(r["activity"])
            ts.append(datetime.fromisoformat(r["timestamp"]).timestamp())
    if acts:
        yield acts, ts


def stats(name, fn, split, cap=None):
    """`cap` keeps only the first `cap` cases, mirroring evaluate.eval_log.max_traces (MIMIC-5k: 5000)."""
    path = os.path.join(RAW, fn)
    it = cases_csv(path) if fn.endswith(".csv") else cases_xes(path)
    n_cases = n_events = n_unsorted = 0
    acts_seen, variants, variants_fo = set(), set(), set()
    lens, durs, gaps = [], [], []
    for acts, ts in it:
        if cap is not None and n_cases >= cap:
            break
        raw_t = np.asarray(ts, dtype=np.float64)
        order = np.argsort(raw_t, kind="stable")
        t = raw_t[order]
        n_cases += 1
        n_events += len(acts)
        lens.append(len(acts))
        acts_seen.update(acts)
        # variants under BOTH conventions: timestamp order (ours) and file order (the published
        # table). They differ only for logs whose file order is not chronological.
        variants.add(hash(tuple(acts[i] for i in order)))
        variants_fo.add(hash(tuple(acts)))
        if not np.all(np.diff(raw_t) >= 0):
            n_unsorted += 1
        durs.append((t[-1] - t[0]) / DAY)
        if len(t) > 1:
            gaps.append(np.diff(t) / DAY)
    lens = np.asarray(lens)
    g = np.concatenate(gaps) if gaps else np.zeros(0)
    assert (g >= 0).all()
    return dict(name=name, split=split, cases=n_cases, events=n_events, acts=len(acts_seen),
                med_len=int(np.median(lens)), max_len=int(lens.max()), variants=len(variants),
                variants_fileorder=len(variants_fo), frac_unsorted=n_unsorted / max(n_cases, 1),
                med_gap=float(np.median(g)), mean_gap=float(g.mean()), frac_zero_gap=float((g == 0).mean()),
                med_dur=float(np.median(durs)), mean_dur=float(np.mean(durs)))


want = sys.argv[1:]
out = []
for entry in LOGS:
    name, fn, split, cap = (*entry, None)[:4]
    if want and name not in want:
        continue
    s = stats(name, fn, split, cap)
    out.append(s)
    print("{name:17s} {split:5s} cases={cases:>7d} events={events:>9d} act={acts:>4d} "
          "medlen={med_len:>3d} maxlen={max_len:>5d} var={variants:>6d} varFO={variants_fileorder:>6d} "
          "unsorted={frac_unsorted:>6.2%} med_gap={med_gap:>9.4f}d zero_gap={frac_zero_gap:>6.1%} "
          "med_dur={med_dur:>9.3f}d mean_dur={mean_dur:>9.2f}d".format(**s),
          flush=True)
# Merge with any existing rows: a filtered run (e.g. `log_stats.py MIMIC-5k`) must not drop the other logs.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
merged = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        merged[r["name"]] = r
for s in out:
    merged[s["name"]] = {k: str(v) for k, v in s.items()}
order = [e[0] for e in LOGS]
rows = [merged[n] for n in order if n in merged] + [r for n, r in merged.items() if n not in order]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote", OUT)
