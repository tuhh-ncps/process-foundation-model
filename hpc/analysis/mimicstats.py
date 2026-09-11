import statistics as st
from pm_foundation.training.ar_pretrain import _read_log
from pm_foundation.data.preprocessing import build_traces
spec={"path":"/workspace/data/raw/mimic_transfers.csv","format":"csv","max_traces":5000}
log=_read_log(spec, strip=True)                       # same call the eval makes
traces=build_traces(log, min_trace_len=2)              # same filter the eval applies
def stats(obj, label):
    trs=getattr(obj,'traces',obj)
    lens=[len(t.events) for t in trs]
    acts={e.activity for t in trs for e in t.events}
    variants={tuple(e.activity for e in t.events) for t in trs}
    print("%-28s #Cases=%s  #Events=%s  #Act.=%d  Med.=%d  Max=%d  #Var.=%s"%(
        label, format(len(trs),","), format(sum(lens),","), len(acts), int(st.median(lens)), max(lens), format(len(variants),",")))
stats(traces, "MIMIC 5000-cap (eval subset)")
# also the raw cap before the min_trace_len filter, for transparency
try:
    raw=build_traces(log, min_trace_len=1)
    stats(raw, "  (before min_trace_len=2)")
except Exception as e:
    print("  raw-cap stats unavailable:", e)
