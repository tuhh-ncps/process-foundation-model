import json,glob,os,csv
from collections import defaultdict
# BPI13 full-budget means: capped(5000) vs uncapped, both 100-epoch, seeds 0,1,2, full-curve runs only
acc=defaultdict(lambda: defaultdict(list))   # (cap,task,arm) -> values
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{})
    if c.get("eval_dataset")!="bpi13_incidents": continue
    if int((c.get("probe") or {}).get("max_epochs",15))!=100: continue
    if "scratch" not in set(c.get("backbones") or {}): continue
    cap=(c.get("eval_log") or {}).get("max_traces")
    cap="capped5000" if cap else "UNCAPPED"
    rows=list(csv.DictReader(open(cv)))
    grp=defaultdict(list)
    for r in rows: grp[(r["task"],r["seed"])].append(r)
    for (t,s),rs in grp.items():
        if len({r["n_labels"] for r in rs})<8 or s not in ("0","1","2"): continue
        for r in rs:
            if r["n_labels"]=="all" and r["backbone_alias"] in ("gin15allw1","random","scratch"):
                acc[(cap,t)][r["backbone_alias"]].append(float(r["value"]))
old={"next_activity":(0.80,0.53,0.67),"next_3_activities":(0.73,0.49,0.60),"next_5_activities":(0.74,0.48,0.56),
     "future_activity_set":(0.93,0.78,0.84),"next_time":(1.90,2.48,1.92),"remaining_time":(3.72,3.79,3.42),"remaining_count":(1.07,2.66,2.12)}
tasks=["next_activity","next_3_activities","next_5_activities","future_activity_set","next_time","remaining_time","remaining_count"]
def m(cap,t,a):
    v=acc.get((cap,t),{}).get(a,[]); return (sum(v)/len(v), len(v)) if v else (float("nan"),0)
print("%-20s | %-22s | %-22s | %-18s"%("task (PFM/Base/E2E)","paper (old table)","capped 5000 (ep100)","UNCAPPED (ep100)"))
for t in tasks:
    o=old[t]
    cp=[m("capped5000",t,a) for a in ("gin15allw1","random","scratch")]
    un=[m("UNCAPPED",t,a) for a in ("gin15allw1","random","scratch")]
    print("%-20s | %5.2f/%5.2f/%5.2f       | %5.2f/%5.2f/%5.2f (n=%d) | %5.2f/%5.2f/%5.2f (n=%d)"%(
        t,o[0],o[1],o[2],cp[0][0],cp[1][0],cp[2][0],cp[0][1],un[0][0],un[1][0],un[2][0],un[0][1]))
