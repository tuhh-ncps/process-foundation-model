import json,glob,os,csv
from collections import defaultdict
LOGS=["bpi13_incidents","BPI17","BPI20ID","helpdesk","mimic_transfer"]
TASKS=["next_activity","next_3_activities","next_5_activities","future_activity_set","next_time","remaining_time","remaining_count"]
acc=defaultdict(list)  # (log,variant,task) -> values
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{}); lg=c.get("eval_dataset")
    if lg not in LOGS or int((c.get("probe") or {}).get("max_epochs",15))!=100: continue
    if lg=="bpi13_incidents" and (c.get("eval_log") or {}).get("max_traces"): continue
    bbs=set(c.get("backbones") or {})
    for v in ("gin15allw1","gin15jepa1"):
        if v not in bbs: continue
        full=defaultdict(set)
        rows=list(csv.DictReader(open(cv)))
        for r in rows:
            if r["backbone_alias"]==v: full[(r["task"],r["seed"])].add(r["n_labels"])
        for r in rows:
            if r["backbone_alias"]==v and r["n_labels"]=="all" and len(full[(r["task"],r["seed"])])>=8:
                acc[(lg,v,r["task"])].append(float(r["value"]))
print("full-budget, 100ep: gin15allw1 (PFM, Table4) vs gin15jepa1 (Table6 default) -- logs where BOTH exist")
for lg in LOGS:
    if not acc.get((lg,"gin15jepa1","next_activity")): continue
    print("== "+lg)
    for t in TASKS:
        a=acc.get((lg,"gin15allw1",t)); j=acc.get((lg,"gin15jepa1",t))
        if a and j:
            am=sum(a)/len(a); jm=sum(j)/len(j)
            print("   %-20s allw1=%.3f  jepa1=%.3f  diff=%+.3f"%(t,am,jm,jm-am))
