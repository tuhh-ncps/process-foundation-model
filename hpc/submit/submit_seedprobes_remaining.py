"""Complete the v2 pretraining-seed replication: full-budget frozen probes (7 tasks, 3 eval seeds) for the
-v2-gin15-s{1,2} and -v2-latent0-s{1,2} backbones on all five logs, submitting only (alias, log, seed) cells
that have no completed run yet."""
import glob, json, os, subprocess
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""), "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"), "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""), "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
bbs = {}
for f in glob.glob("outputs/backbones/*/manifest.json"):
    m = json.load(open(f)); d = os.path.dirname(f); n = os.path.basename(d)
    for v in ("gin15", "latent0"):
        for s in (1, 2):
            if ("-v2-%s-s%d-" % (v, s)) in n and m.get("completed_at") and os.path.exists(d + "/backbone.pt"):
                c = m["config"]; assert c.get("seed") == s and float(c["ar"]["jepa_weight"]) == (1.0 if v == "gin15" else 0.0), n
                bbs["%s_s%d" % (v, s)] = n
print("backbones:", bbs); assert len(bbs) == 4
done = set()
for f in glob.glob("outputs/label_efficiency/*/manifest.json"):
    m = json.load(open(f)); c = m.get("config", {})
    if not m.get("completed_at") or c.get("max_trace_len") != 64 or c.get("role_corpus") != "budget": continue
    if len(c.get("tasks") or []) < 7: continue
    for alias, run in (c.get("backbones") or {}).items():
        if alias in bbs and run == bbs[alias]:
            for sd in c.get("seeds") or []: done.add((alias, c.get("eval_dataset"), int(sd)))
print("already completed cells:", len(done))
n_sub = 0
for alias, bb in bbs.items():
    for log, (path, extra) in LOGS.items():
        for sd in (0, 1, 2):
            if (alias, log, sd) in done: continue
            args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["+evaluate.backbones.%s=%s" % (alias, bb), "evaluate.label_sizes=[null]", "evaluate.seeds=[%d]" % sd])
            tl = "04:00:00" if log == "BPI17" else "02:00:00"
            r = subprocess.run(["sbatch", "--job-name=r22-sp-%s-%s-s%d" % (alias, log, sd), "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                               env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
            n_sub += r.returncode == 0
            if r.returncode != 0: print("FAIL", alias, log, sd, r.stderr.strip())
print("submitted:", n_sub)
