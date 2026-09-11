import json,glob,os,csv,sys
from collections import defaultdict
LOGS={"bpi13_incidents","BPI17","BPI20ID","helpdesk","mimic_transfer"}
VARS={"gin15jepa1","mlp15jepa1","raw15jepa1","gin11jepa1","gin0jepa1","gin15jepa0","norole"}
NB=8
rows={}
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf)); done=m.get("completed_at")
    if not done: continue
    c=m.get("config",{}); lg=c.get("eval_dataset")
    if lg not in LOGS: continue
    if int((c.get("probe") or {}).get("max_epochs",15))!=100: continue
    if lg=="bpi13_incidents" and (c.get("eval_log") or {}).get("max_traces"): continue
    if not (set(c.get("backbones") or {}) & VARS): continue
    grp=defaultdict(list)
    for r in csv.DictReader(open(cv)):
        if r["backbone_alias"] in VARS: grp[(lg,r["backbone_alias"],r["task"],r["seed"])].append(r)
    for k,rs in grp.items():
        if len({r["n_labels"] for r in rs})<NB: continue
        if k not in rows or done>rows[k][0]: rows[k]=(done,rs)
w=csv.writer(sys.stdout); w.writerow(["log","backbone_alias","task","metric","n_labels","n_train_samples","seed","value","run"])
for (lg,al,t,s),(done,rs) in sorted(rows.items()):
    for r in rs: w.writerow([lg,al,t,r.get("metric",""),r["n_labels"],r.get("n_train_samples",""),s,r["value"],done])
cov=defaultdict(set)
for (lg,al,t,s) in rows:
    if t=="next_activity": cov[(al,lg)].add(s)
print("coverage (seeds with full next_activity curve):",file=sys.stderr)
for al in sorted(VARS):
    print("  %-11s "%al+"  ".join("%s:%d"%(lg[:6],len(cov.get((al,lg),()))) for lg in sorted(LOGS)),file=sys.stderr)
