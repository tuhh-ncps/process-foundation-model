"""Setting 1 timing for the PFM-Scratch arm (r30): PFM's architecture randomly initialised and trained end to end
on the target log. Common task set (next_activity, remaining_time), full budget, one seed, each job pinned to a
full H200 and chained afterany so exactly one timing job runs at a time -- identical protocol to r17
(submit_timing3.py, PFM / PFM-FT) and r24 (submit_timing4.py, PFM-RFT).

Arm definition identical to the r29 label-efficiency grid (submit_scratch_grid.py): the `random_role` sentinel
(no pretrained weights) with its alias added to `evaluate.finetune`. The reported time is the job's elapsed
wall-clock (sacct), the same quantity recorded for r17/r24 in results/timing_pinned.csv.

Usage (from the repository root):  python hpc/submit/submit_timing5.py [<jobid to chain after> | none] [--dry]
"""
import os
import subprocess
import sys

GRES = "--gres=gpu:nvidia_h200_nvl:1"
LOGS = [("helpdesk", "/workspace/data/raw/helpdesk.csv", ""), ("mimic_transfer", "/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        ("bpi13_incidents", "/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""), ("BPI20ID", "/workspace/data/raw/BPI20ID.xes", ""), ("BPI17", "/workspace/data/raw/BPI17.xes", "")]
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
          "evaluate.tasks=[next_activity,remaining_time]"]
ARM = ["+evaluate.backbones.pfm_scratch=random_role", "+evaluate.finetune=[pfm_scratch]"]
# From random init training may run longer than PFM-FT (r17 BPI17: 6.5 min, 2 tasks). Generous limits.
TIME = {"BPI17": "04:00:00", "_": "01:00:00"}

DRY = "--dry" in sys.argv
pos = [a for a in sys.argv[1:] if a != "--dry"]
prev = pos[0] if pos and pos[0] != "none" else None
assert os.path.exists("slurm/ncps/run.sbatch"), "run from the repository root"


def sbatch(name, tl, args, dep):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", GRES, "--export=ALL"]
    if dep:
        cmd.append("--dependency=afterany:%s" % dep)
    cmd.append("slurm/ncps/run.sbatch")
    if DRY:
        print("DRY  %-28s %s  dep=%s" % (name, tl, dep))
        return "DRY-" + name
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
    return jid


for log, path, extra in LOGS:
    base = COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["evaluate.label_sizes=[null]", "evaluate.seeds=[0]"]
    prev = sbatch("r30-time2-scratch-" + log, TIME.get(log, TIME["_"]), " ".join(base + ARM), prev)
print("last job:", prev)
