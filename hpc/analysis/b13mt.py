import json,glob,os,csv
rows=[]
for d in sorted(glob.glob("outputs/label_efficiency/*/")):
    mf,cv=d+"manifest.json",d+"curves.csv"
    if not (os.path.exists(mf) and os.path.exists(cv)): continue
    m=json.load(open(mf))
    if not m.get("completed_at"): continue
    c=m.get("config",{})
    if c.get("eval_dataset")!="bpi13_incidents": continue
    if "scratch" not in set(c.get("backbones") or {}): continue
    ep=int((c.get("probe") or {}).get("max_epochs",15))
    mt=(c.get("eval_log") or {}).get("max_traces","NONE")
    nt=None
    for r in csv.DictReader(open(cv)):
        if r["task"]=="next_time" and r["n_labels"]=="all" and r["backbone_alias"]=="gin15allw1":
            nt=float(r["value"])
    rows.append((ep,str(mt),nt,os.path.basename(d.rstrip("/"))[:42]))
for ep,mt,nt,rid in sorted(rows,key=lambda r:(r[0],r[1])):
    print("  ep=%-4d max_traces=%-6s PFM_next_time_all=%s  %s"%(ep,mt,("%.3f"%nt if nt is not None else "--"),rid))
# also: does the split job (single-task next_time) exist? and what BPI13 timestamp unit
