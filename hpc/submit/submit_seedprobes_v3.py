"""Full-budget frozen probes (7 tasks, 3 eval seeds, 5 logs, budget corpus) for the seed-replication backbones
(-v2-gin15-s{1,2}, -v2-latent0-s{1,2}). Alias = <variant>_s<pretrain seed>. Usage: python submit_seedprobes.py"""
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
            if ("-v3-%s-s%d-" % (v, s)) in n and m.get("completed_at") and os.path.exists(d + "/backbone.pt"):
                c = m["config"]; assert c.get("seed") == s and c["model"].get("id_dropout") == 1.0, n
                assert float(c["ar"]["jepa_weight"]) == (1.0 if v == "gin15" else 0.0), n
                bbs["%s_s%d" % (v, s)] = n
print("backbones:", bbs)
assert len(bbs) == 4, "expected 4 completed seed backbones"
for alias, bb in bbs.items():
    for log, (path, extra) in LOGS.items():
        for sd in (0, 1, 2):
            args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["+evaluate.backbones.%s=%s" % (alias, bb), "evaluate.label_sizes=[null]", "evaluate.seeds=[%d]" % sd])
            tl = "04:00:00" if log == "BPI17" else "02:00:00"
            r = subprocess.run(["sbatch", "--job-name=r20-sp-%s-%s-s%d" % (alias, log, sd), "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                               env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + "%s %s s%d  " % (alias, log, sd) + (r.stdout or r.stderr).strip())
