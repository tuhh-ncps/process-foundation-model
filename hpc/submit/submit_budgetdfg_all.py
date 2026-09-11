"""Budget-corpus descriptors (role_corpus=budget) for the PFM arm on the four logs not yet run; mirrors submit_randrole.py:
reuse each log's main-table hydra overrides (paths, caps), keep only the PFM backbone, 100-epoch probe, 8 budgets, 7 tasks."""
import glob, os, subprocess, sys
LOGS = {"helpdesk": "04:00:00", "bpi13_incidents": "05:00:00", "mimic_transfer": "04:00:00", "BPI17": "14:00:00"}
PFM = "+evaluate.backbones.gin15allw1=backbone-20260825-125337-multi-none-frozen-gin15allw1-b1041f"
PATCH = ["evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
         "evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]",
         "evaluate.tasks=[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]",
         "evaluate.role_corpus=budget"]
DROP = ("evaluate.seeds", "evaluate.probe.max_epochs", "evaluate.probe.early_stop_patience", "evaluate.label_sizes",
        "evaluate.tasks", "+evaluate.backbones.", "model=", "evaluate.role_corpus")
def find(log):
    for d in sorted(glob.glob("outputs/hydra/*/"), reverse=True):
        f = d + ".hydra/overrides.yaml"
        if not os.path.exists(f): continue
        ov = [l[2:].strip() for l in open(f) if l.startswith("- ")]
        if any(o == "evaluate.eval_dataset=" + log for o in ov) and any(o.startswith("evaluate.probe.max_epochs=100") for o in ov) and not any("role_corpus=budget" in o or "random_role" in o for o in ov):
            return ov
    return None
only = sys.argv[1:]
for log, tl in LOGS.items():
    if only and log not in only: continue
    ov = find(log)
    if ov is None: print("NO OVERRIDES:", log); continue
    base = [o for o in ov if not o.startswith(DROP)]
    if log == "bpi13_incidents": base = [o for o in base if "max_traces" not in o]
    base += ["model=role_gin15", "~evaluate.backbones.random", PFM] + PATCH
    for sd in (0, 1, 2):
        args = " ".join(base + ["evaluate.seeds=[%d]" % sd]); name = "r12-budgetdfg-%s-s%d" % (log, sd)
        cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"]
        r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
        print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
        if sd == 0: print("     ARGS: " + args)
