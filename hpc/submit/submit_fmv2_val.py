"""FM-v2 under our protocol, validation-selected read-out, full budget, one seed.
  python submit_fmv2_val.py export   -> 5 CPU jobs: scripts/export_queries.py (exact query set of our evaluation)
  python submit_fmv2_val.py eval     -> 10 GPU jobs (log x task), each afterok its export job
"""
import json, os, subprocess, sys
LOGS = {"helpdesk": "02:00:00", "bpi13_incidents": "02:00:00", "mimic_transfer": "02:00:00", "BPI20ID": "03:00:00", "BPI17": "10:00:00"}
IDS = "outputs/fmv2/export_jobs.json"
os.makedirs("outputs/fmv2", exist_ok=True)

def sbatch(name, tl, cmd, gpu, dep=None, nice=None):
    args = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--export=ALL"]
    if gpu: args.append("--gres=gpu:1")
    if dep: args.append("--dependency=afterok:%s" % dep)
    if nice: args.append("--nice=%d" % nice)
    args.append("slurm/ncps/run_cmd.sbatch")
    r = subprocess.run(args, env=dict(os.environ, USE_GPU="1" if gpu else "0", CMD=cmd), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid

phase = sys.argv[1]
if phase == "export":
    ids = {}
    for log in LOGS:
        meta = json.load(open("exports/%s_splits.csv.json" % log))
        mt = " --max-traces %d" % meta["max_traces"] if meta.get("max_traces") else ""
        cmd = ("cd /workspace && python scripts/export_queries.py --log %s --path %s%s --splits-csv exports/%s_splits.csv "
               "--out exports/%s_queries.csv" % (log, meta["path"], mt, log, log))
        ids[log] = sbatch("r10-export-" + log, "01:00:00", cmd, gpu=False)
    json.dump(ids, open(IDS, "w"))
elif phase == "eval":
    ids = json.load(open(IDS))
    only = sys.argv[2:]  # optional log filter; exports already done -> no dependency
    for log, tl in LOGS.items():
        if only and log not in only: continue
        for task in ("next_activity", "remaining_time"):
            out = "outputs/fmv2/%s_val_%s.csv" % (log, task)
            cmd = ("cd /workspace && python scripts/fmv2_eval.py --repo external/events-transf --checkpoint-dir external/fmv2_ckpt/checkpoints "
                   "--splits exports/%s_splits.csv --queries exports/%s_queries.csv --log %s --tasks %s --k 1 5 10 20 50 100 200 "
                   "--val-max-queries 50000 --budgets all --seeds 0 --device cuda --out %s" % (log, log, log, task, out))
            sbatch("r10-fmv2v-%s-%s" % (log, task), tl, cmd, gpu=True, dep=None if only else ids[log])
