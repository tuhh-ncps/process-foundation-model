"""Setting 1 timing: the common task set (next_activity, remaining_time) only, PFM frozen + PFM-FT, full budget, one seed,
pinned to a full H200, appended to the running chain (afterany the last chain job) so one timing job runs at a time."""
import glob, json, os, subprocess, sys
GRES = "--gres=gpu:nvidia_h200_nvl:1"
LOGS = [("helpdesk", "/workspace/data/raw/helpdesk.csv", ""), ("mimic_transfer", "/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        ("bpi13_incidents", "/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""), ("BPI20ID", "/workspace/data/raw/BPI20ID.xes", ""), ("BPI17", "/workspace/data/raw/BPI17.xes", "")]
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
          "evaluate.tasks=[next_activity,remaining_time]"]
hits = []
for f in glob.glob("outputs/backbones/*/manifest.json"):
    m = json.load(open(f)); d = os.path.dirname(f)
    if os.path.basename(d).split("-v2-")[-1].rsplit("-", 1)[0] == "gin15" and m.get("completed_at") and os.path.exists(d + "/backbone.pt"):
        hits.append((m["completed_at"], os.path.basename(d)))
pfm = sorted(hits)[-1][1]
prev = sys.argv[1]  # last job of the existing chain
def sbatch(name, tl, args, dep):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", GRES, "--export=ALL", "--dependency=afterany:%s" % dep, "slurm/ncps/run.sbatch"]
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
for log, path, extra in LOGS:
    tl = "02:00:00" if log == "BPI17" else "01:00:00"
    base = COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["evaluate.label_sizes=[null]", "evaluate.seeds=[0]"]
    prev = sbatch("r17-time2-pfm-" + log, tl, " ".join(base + ["+evaluate.backbones.pfm=" + pfm]), prev)
    prev = sbatch("r17-time2-ft-" + log, tl, " ".join(base + ["+evaluate.backbones.pfm_ft=" + pfm, "+evaluate.finetune=[pfm_ft]"]), prev)
print("last job:", prev)
