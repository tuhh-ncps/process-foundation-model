import json,glob,os,csv
from collections import defaultdict
want={"gin15allw1","random","scratch"}
tab={}; hdr=None
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{})
    if c.get("eval_dataset")!="bpi13_incidents": continue
    if int((c.get("probe") or {}).get("max_epochs",15))!=100: continue
    if (c.get("eval_log") or {}).get("max_traces"): continue   # UNCAPPED only
    if not want & set(c.get("backbones") or {}): continue
    rows=list(csv.DictReader(open(cv)))
    if rows and hdr is None: hdr=list(rows[0])
    g=defaultdict(list)
    for r in rows: g[(r["task"],r["seed"])].append(r)
    for (t,s),rs in g.items():
        if len({r["n_labels"] for r in rs})<8: continue
        for r in rs: tab.setdefault((r["task"],r["seed"],r["backbone_alias"],r["n_labels"]),r)
with open("/tmp/bpi13_incidents_ep100.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=hdr); w.writeheader()
    for r in tab.values(): w.writerow(r)
print("uncapped BPI13 ep100:",len(tab),"rows; tasks",len({k[0] for k in tab}),"seeds",sorted({k[1] for k in tab}),"arms",sorted({k[2] for k in tab}))
