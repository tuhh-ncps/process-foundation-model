import json,glob,os,csv
from collections import defaultdict
LOGS=["bpi13_incidents","BPI17","BPI20ID","helpdesk","mimic_transfer"]
acc=defaultdict(list)   # (log,variant) -> next_activity@all values over seeds
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{}); lg=c.get("eval_dataset")
    if lg not in LOGS or int((c.get("probe") or {}).get("max_epochs",15))!=100: continue
    if lg=="bpi13_incidents" and (c.get("eval_log") or {}).get("max_traces"): continue  # uncapped only
    bbs=set(c.get("backbones") or {})
    for v in ("gin15jepa1","norole"):
        if v not in bbs: continue
        grp=defaultdict(set)
        for r in csv.DictReader(open(cv)):
            if r["backbone_alias"]==v and r["task"]=="next_activity": grp[r["seed"]].add(r["n_labels"])
        for r in csv.DictReader(open(cv)):
            if r["backbone_alias"]==v and r["task"]=="next_activity" and r["n_labels"]=="all" and len(grp[r["seed"]])>=8:
                acc[(lg,v)].append((r["seed"],float(r["value"])))
print("next_activity @ full budget, 100ep uncapped-BPI13:  default(gin15jepa1) vs no-role(norole)")
print("%-16s %-22s %-22s"%("log","default  (seeds)","no-role  (seeds)"))
dm,nm=[],[]
for lg in LOGS:
    d=dict(acc.get((lg,"gin15jepa1"),[])); n=dict(acc.get((lg,"norole"),[]))
    dv=sum(d.values())/len(d) if d else None; nv=sum(n.values())/len(n) if n else None
    if dv is not None and nv is not None: dm.append(dv); nm.append(nv)
    print("%-16s %-22s %-22s"%(lg, ("%.3f  %s"%(dv,sorted(d))) if d else "-- pending --", ("%.3f  %s"%(nv,sorted(n))) if n else "-- pending --"))
if dm:
    print("\nPRELIMINARY mean over %d logs with BOTH variants: default=%.1f%%  no-role=%.1f%%  gain=%+.1f pp"%(len(dm),100*sum(dm)/len(dm),100*sum(nm)/len(nm),100*(sum(dm)/len(dm)-sum(nm)/len(nm))))
