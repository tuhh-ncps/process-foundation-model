"""Timing runs: full-budget-only PFM-FT and frozen PFM, one seed, 7 tasks, one job per log (no continuations)."""
import glob, json, os, subprocess
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""), "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"), "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""), "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
hits = []
for f in glob.glob("outputs/backbones/*/manifest.json"):
    m = json.load(open(f)); d = os.path.dirname(f)
    if os.path.basename(d).split("-v2-")[-1].rsplit("-", 1)[0] == "gin15" and m.get("completed_at") and os.path.exists(d + "/backbone.pt"):
        hits.append((m["completed_at"], os.path.basename(d)))
pfm = sorted(hits)[-1][1]; print("backbone:", pfm)
def sbatch(name, tl, args):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"]
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
for log, (path, extra) in LOGS.items():
    tl = "06:00:00" if log == "BPI17" else "02:00:00"
    base = COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["evaluate.label_sizes=[null]", "evaluate.seeds=[0]"]
    sbatch("r15-time-ft-" + log, tl, " ".join(base + ["+evaluate.backbones.pfm_ft=" + pfm, "+evaluate.finetune=[pfm_ft]"]))
    sbatch("r15-time-pfm-" + log, tl, " ".join(base + ["+evaluate.backbones.pfm=" + pfm]))
