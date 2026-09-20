"""Collect the r26 linear-head runs (probe.head_hidden=0, three regression tasks) into one CSV.

Mirrors collect_v2.py but keeps 3-task grids and requires head_hidden=0, so it can never pick up the
main (head_hidden=128) runs. Columns: log,arm,task,n_labels,n_train_samples,seed,value,run.
"""
import csv, glob, json, sys

rows = {}
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    try:
        m = json.load(open(d + "manifest.json"))
    except Exception:
        continue
    c = m.get("config", {})
    if not m.get("completed_at") or c.get("max_trace_len") != 64 or c.get("role_corpus") != "budget":
        continue
    if int(((c.get("probe") or {}).get("head_hidden", 128)) or 0) != 0:
        continue
    ts = c.get("tasks") or []
    if sorted(ts) != ["next_time", "remaining_count", "remaining_time"]:
        continue
    bbs = c.get("backbones") or {}
    if not bbs or not all(("-v2-" in str(v)) or (v == "random_role") for v in bbs.values()):
        continue
    lg = c.get("eval_dataset")
    for r in csv.DictReader(open(d + "curves.csv")):
        key = (lg, r["backbone_alias"], r["task"], r["n_labels"], r["seed"])
        rows[key] = (m["completed_at"], [lg, r["backbone_alias"], r["task"], r["n_labels"],
                                         r.get("n_train_samples", ""), r["seed"], r["value"], m["completed_at"]])
if not rows:   # refuse to print a header-only CSV: redirecting it would truncate the committed results
    sys.exit("no matching linear-head runs under outputs/label_efficiency/; nothing collected")
w = csv.writer(sys.stdout)
w.writerow(["log", "arm", "task", "n_labels", "n_train_samples", "seed", "value", "run"])
for k in sorted(rows):
    w.writerow(rows[k][1])
