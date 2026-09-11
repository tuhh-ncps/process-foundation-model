import glob, os
# count traces in each raw log to size uncapped runs
from pm_foundation.data.readers.xes_reader import XesReader
import warnings; warnings.filterwarnings("ignore")
paths = {
  "BPI13": "data/raw/BPI_Challenge_2013_incidents.xes",
  "MIMIC": "data/raw/mimic_transfers.csv",
  "BPI18": "data/raw/BPI_Challenge_2018.xes",
}
import subprocess
for nm,p in paths.items():
    full = os.path.join("/workspace", p) if not p.startswith("/") else p
    if not os.path.exists(full):
        # try alternates
        cand = glob.glob("data/raw/*"+os.path.basename(p).split('.')[0][:6]+"*")
        print("  %-6s path? %s  alts=%s"%(nm, os.path.exists(full), cand[:3])); continue
    if p.endswith(".csv"):
        import csv
        cases=set()
        with open(full) as f:
            r=csv.DictReader(f)
            col=None
            for row in r:
                if col is None:
                    for c in row:
                        if "case" in c.lower() or "trace" in c.lower(): col=c; break
                    if col is None: col=list(row)[0]
                cases.add(row.get(col))
        print("  %-6s ~%d cases (csv, col=%s)"%(nm,len(cases),col))
    else:
        # count <trace> occurrences cheaply
        n=int(subprocess.run(["grep","-c","<trace>",full],capture_output=True,text=True).stdout or 0)
        print("  %-6s ~%d traces (xes)"%(nm,n))
