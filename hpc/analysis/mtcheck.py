import json,glob,os,csv
from collections import defaultdict
# max_traces per log, per epoch-setting, across all scratch-containing runs
seen=defaultdict(set)
ntrain=defaultdict(set)  # actual #train cases at the 'all' budget (n_train_samples)
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{}); lg=c.get("eval_dataset")
    if not lg: continue
    if "scratch" not in set(c.get("backbones") or {}): continue
    ep=int((c.get("probe") or {}).get("max_epochs",15))
    mt=(c.get("eval_log") or {}).get("max_traces","NONE")
    seen[(lg,ep)].add(str(mt))
    for r in csv.DictReader(open(cv)):
        if r["n_labels"]=="all" and r.get("n_train_samples"):
            ntrain[(lg,ep)].add(r["n_train_samples"])
for (lg,ep) in sorted(seen):
    nt=sorted(int(x) for x in ntrain[(lg,ep)] if x)
    print("  %-18s ep=%-4d max_traces=%-8s  n_train@all=%s"%(lg,ep,",".join(sorted(seen[(lg,ep)])), (str(max(nt)) if nt else "?")))
