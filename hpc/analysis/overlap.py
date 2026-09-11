import json, os, csv
from pm_foundation.training.ar_pretrain import _read_log
# (2) pretraining vocab = the backbone's feature_spec activity vocab
BB="outputs/backbones/backbone-20260825-125337-multi-none-frozen-gin15allw1-b1041f"
spec=json.load(open(BB+"/feature_spec.json"))
def find_vocab(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if "activit" in k.lower() and isinstance(v,(list,dict)) and len(v)>50: return set(v if isinstance(v,list) else v.keys())
            r=find_vocab(v)
            if r: return r
    return None
pre=find_vocab(spec) or set()
print("pretraining activity vocab size:", len(pre))
logs={"bpi13_incidents":("/workspace/data/raw/BPI_Challenge_2013_incidents.xes",None),
      "BPI17":("/workspace/data/raw/BPI17.xes",None),
      "BPI20ID":("/workspace/data/raw/BPI20ID.xes",None),
      "helpdesk":("/workspace/data/raw/helpdesk.csv",None),
      "mimic_transfer":("/workspace/data/raw/mimic_transfers.csv",5000)}
for lg,(p,cap) in logs.items():
    s={"path":p}; 
    if cap: s["max_traces"]=cap
    try:
        log=_read_log(s, strip=True)
        acts={e.activity for t in log.traces for e in t.events}
        ev=sum(len(t.events) for t in log.traces)
        seen_ev=sum(1 for t in log.traces for e in t.events if e.activity in pre)
        print("  %-16s #act=%3d  in-pretrain-vocab=%3d (%.0f%% of types, %.0f%% of events)"%(lg,len(acts),len(acts&pre),100*len(acts&pre)/max(1,len(acts)),100*seen_ev/max(1,ev)))
    except Exception as e:
        print("  %-16s ERR %s"%(lg,str(e)[:80]))
