"""Fine-tuned PFM arm (alias pfm_ft): pretrained PFM backbone + role encoder + fresh head, ALL trained end to end on the target
log with the Scratch-E2E recipe (same optimizer/lr/schedule/epochs/patience), main-table protocol: 7 tasks, 8 budgets, 3 seeds.
Reuses each log's main-table hydra overrides (paths, caps). Continuation jobs (afterany) re-run identical args; the runner's
crash recovery resumes finished probes, so a 24 h time-out costs nothing.  Usage: python submit_pfmft.py [LOG ...]"""
import glob, os, subprocess, sys
LOGS = {"helpdesk": ("10:00:00", 1), "bpi13_incidents": ("12:00:00", 1), "mimic_transfer": ("10:00:00", 1), "BPI20ID": ("12:00:00", 1), "BPI17": ("23:59:00", 2)}
PFM = "+evaluate.backbones.pfm_ft=backbone-20260825-125337-multi-none-frozen-gin15allw1-b1041f"
PATCH = ["evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
         "evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]",
         "evaluate.tasks=[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]",
         "+evaluate.finetune=[pfm_ft]"]
DROP = ("evaluate.seeds", "evaluate.probe.max_epochs", "evaluate.probe.early_stop_patience", "evaluate.label_sizes",
        "evaluate.tasks", "+evaluate.backbones.", "model=", "evaluate.role_corpus", "evaluate.finetune", "+evaluate.finetune")
def find(log):
    for d in sorted(glob.glob("outputs/hydra/*/"), reverse=True):
        f = d + ".hydra/overrides.yaml"
        if not os.path.exists(f): continue
        ov = [l[2:].strip() for l in open(f) if l.startswith("- ")]
        if any(o == "evaluate.eval_dataset=" + log for o in ov) and any(o.startswith("evaluate.probe.max_epochs=100") for o in ov) and not any(("role_corpus=budget" in o) or ("random_role" in o) or ("pfm_ft" in o) for o in ov):
            return ov
    return None
def sbatch(name, tl, args, dep=None):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL"]
    if dep: cmd.append("--dependency=afternotok:%s" % dep)  # continue ONLY if the previous job failed/timed out
    cmd.append("slurm/ncps/run.sbatch")
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
only = [a for a in sys.argv[1:] if a != "chain"]
CHAIN_ONLY = "chain" in sys.argv[1:]   # attach afternotok continuations to already-submitted primaries (looked up by name)
for log, (tl, chain) in LOGS.items():
    if only and log not in only: continue
    ov = find(log)
    if ov is None: print("NO OVERRIDES:", log); continue
    base = [o for o in ov if not o.startswith(DROP)]
    if log == "bpi13_incidents": base = [o for o in base if "max_traces" not in o]
    base += ["model=role_gin15", "~evaluate.backbones.random", PFM] + PATCH
    for sd in (0, 1, 2):
        args = " ".join(base + ["evaluate.seeds=[%d]" % sd])
        name = "r13-pfmft-%s-s%d" % (log, sd)
        if CHAIN_ONLY:
            jid = subprocess.run(["squeue", "-h", "-n", name, "-o", "%i"], capture_output=True, text=True).stdout.split()
            jid = jid[0] if jid else None
            if not jid: print("no running/pending primary for", name); continue
        else:
            jid = sbatch(name, tl, args)
        for c in range(chain):
            jid = sbatch("r13-pfmft-%s-s%d-c%d" % (log, sd, c + 1), tl, args, dep=jid)
        if sd == 0: print("     ARGS: " + args)
