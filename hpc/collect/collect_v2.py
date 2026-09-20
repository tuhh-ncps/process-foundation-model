"""Collect v2 (leak-fixed, ID-free, budget-corpus, max_trace_len=64) label-efficiency runs into one CSV on stdout.
Columns: log,arm,task,n_labels,n_train_samples,seed,value,run.  Arms: pfm, random_role, pfm_ft (main grid) and the
ablation aliases (mlp15, raw15, gin11, gin0, latent0, norole). Only completed runs whose config has max_trace_len=64,
role_corpus=budget and a v2 backbone (or the random_role sentinel) are kept; the latest completed run wins per
(log, arm, seed)."""
import json, glob, csv, sys
rows = {}
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    try: m = json.load(open(d + "manifest.json"))
    except Exception: continue
    c = m.get("config", {})
    if not m.get("completed_at") or c.get("max_trace_len") != 64 or c.get("role_corpus") != "budget": continue
    bbs = c.get("backbones") or {}
    ok = all(("-v2-" in str(v)) or (v == "random_role") for v in bbs.values())
    if not ok or not bbs: continue
    # skip the timing runs: fewer than the 7 tasks, or main-grid arms evaluated at a single budget
    if len(c.get("tasks") or []) < 7: continue
    if len(c.get("label_sizes") or []) <= 1 and any(a in ("pfm", "pfm_ft", "random_role") for a in bbs): continue
    lg = c.get("eval_dataset"); seed = (c.get("seeds") or [None])[0]
    for r in csv.DictReader(open(d + "curves.csv")):
        key = (lg, r["backbone_alias"], r["task"], r["n_labels"], r["seed"])
        rows[key] = (m["completed_at"], [lg, r["backbone_alias"], r["task"], r["n_labels"], r.get("n_train_samples", ""), r["seed"], r["value"], m["completed_at"]])
if not rows:   # refuse to print a header-only CSV: redirecting it would truncate the committed results
    sys.exit("no matching v2 label-efficiency runs under outputs/label_efficiency/; nothing collected")
w = csv.writer(sys.stdout); w.writerow(["log", "arm", "task", "n_labels", "n_train_samples", "seed", "value", "run"])
for key in sorted(rows): w.writerow(rows[key][1])
