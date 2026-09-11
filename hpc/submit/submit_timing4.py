"""Setting 1 timing for the PFM-RFT arm (role encoder fine-tuned, transformer frozen): common task set
(next_activity, remaining_time), full budget, one seed, each job pinned to a full H200 and chained
afterany so exactly one timing job runs at a time — identical protocol to r17 (submit_timing3.py)."""
import glob, json, os, subprocess, sys
GRES = "--gres=gpu:nvidia_h200_nvl:1"
LOGS = [("helpdesk", "/workspace/data/raw/helpdesk.csv", ""), ("mimic_transfer", "/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        ("bpi13_incidents", "/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""), ("BPI20ID", "/workspace/data/raw/BPI20ID.xes", ""), ("BPI17", "/workspace/data/raw/BPI17.xes", "")]
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
          "evaluate.tasks=[next_activity,remaining_time]"]
PFM = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
assert os.path.exists("outputs/backbones/%s/backbone.pt" % PFM)
prev = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
def sbatch(name, tl, args, dep):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", GRES, "--export=ALL"]
    if dep: cmd.append("--dependency=afterany:%s" % dep)
    cmd.append("slurm/ncps/run.sbatch")
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
for log, path, extra in LOGS:
    tl = "02:00:00" if log == "BPI17" else "01:00:00"
    base = COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["evaluate.label_sizes=[null]", "evaluate.seeds=[0]"]
    prev = sbatch("r24-time2-rft-" + log, tl, " ".join(base + ["+evaluate.backbones.pfm_rft=" + PFM, "+evaluate.finetune_role=[pfm_rft]"]), prev)
print("last job:", prev)
