import glob, os, subprocess
LOGS = ["berti_receipt", "berti_sepsis", "berti_helpdesk"]
PATCH = ["evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
         "evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]",
         "evaluate.tasks=[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"]
DROP = ("evaluate.seeds", "evaluate.probe.max_epochs", "evaluate.probe.early_stop_patience",
        "evaluate.label_sizes", "evaluate.tasks")
def find(log):
    for d in sorted(glob.glob("outputs/hydra/*/"), reverse=True):
        f = d + ".hydra/overrides.yaml"
        if not os.path.exists(f): continue
        ov = [l[2:].strip() for l in open(f) if l.startswith("- ")]
        if any(o == "evaluate.eval_dataset=" + log for o in ov) and any("scratch" in o for o in ov):
            return ov
    return None
for log in LOGS:
    ov = find(log)
    if ov is None:
        print("NO 3-ARM OVERRIDES FOUND:", log); continue
    base = [o for o in ov if not o.startswith(DROP)] + PATCH
    for sd in (0, 1, 2):
        args = " ".join(base + ["evaluate.seeds=[%d]" % sd])
        name = "r6-fmv2-%s-s%d" % (log.replace("berti_", ""), sd)
        cmd = ["sbatch", "--job-name=" + name, "--time=04:00:00", "--cpus-per-task=4",
               "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"]
        r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
        print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
        if sd == 0: print("     ARGS: " + args)
