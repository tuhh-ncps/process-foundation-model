"""v2 evaluation grids on the leak-fixed, ID-free backbones.
  python submit_v2.py main [LOG ...]      -> PFM (frozen), Random-role, PFM-ft; budget corpus; 8 budgets; 3 seeds; one job per arm/log/seed
  python submit_v2.py ablation [LOG ...]  -> 6 variant backbones, frozen, FULL budget only, budget corpus, 3 seeds
Backbone ids are resolved from outputs/backbones/*/manifest.json by tag (-v2-<variant>), newest completed with backbone.pt."""
import glob, json, os, subprocess, sys
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""), "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"), "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""), "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
TIME = {"main": {"BPI17": {"pfm": "10:00:00", "random_role": "10:00:00", "pfm_ft": "23:59:00"}, "_": "06:00:00"}, "ablation": {"BPI17": "04:00:00", "_": "02:00:00"}}
def backbone(variant):
    hits = []
    for f in glob.glob("outputs/backbones/*/manifest.json"):
        m = json.load(open(f)); c = m.get("config", {})
        tagged = str(c.get("tag", "")) == "-v2-" + variant or str(c.get("name", "")).endswith("-v2-" + variant) or os.path.basename(os.path.dirname(f)).split("-v2-")[-1].rsplit("-", 1)[0] == variant
        if tagged and m.get("completed_at") and os.path.exists(os.path.dirname(f) + "/backbone.pt"):
            hits.append((m["completed_at"], os.path.basename(os.path.dirname(f))))
    return sorted(hits)[-1][1] if hits else None
def sbatch(name, tl, args, dep=None):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL"]
    if dep: cmd.append("--dependency=afternotok:%s" % dep)
    cmd.append("slurm/ncps/run.sbatch")
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
phase = sys.argv[1]; only = sys.argv[2:]
pfm = backbone("gin15")
if phase == "main":
    if not pfm: sys.exit("no completed v2 gin15 backbone yet")
    ARMS = {"pfm": ["+evaluate.backbones.pfm=" + pfm], "random_role": ["+evaluate.backbones.random_role=random_role"],
            "pfm_ft": ["+evaluate.backbones.pfm_ft=" + pfm, "+evaluate.finetune=[pfm_ft]"]}
    for log, (path, extra) in LOGS.items():
        if only and log not in only: continue
        for arm, ov in ARMS.items():
            for sd in (0, 1, 2):
                args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ov + ["evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
                tl = TIME["main"]["BPI17"][arm] if log == "BPI17" else TIME["main"]["_"]
                jid = sbatch("r14-main-%s-%s-s%d" % (log, arm, sd), tl, args)
                for c in range(2 if log == "BPI17" else 1):
                    jid = sbatch("r14-main-%s-%s-s%d-c%d" % (log, arm, sd, c + 1), tl, args, dep=jid)
                if sd == 0 and log == "helpdesk": print("     ARGS: " + args)
elif phase == "ablation":
    for v in ("mlp15", "raw15", "gin11", "gin0", "latent0", "norole"):
        bb = backbone(v)
        if not bb: print("no completed v2 backbone for", v); continue
        for log, (path, extra) in LOGS.items():
            if only and log not in only: continue
            for sd in (0, 1, 2):
                args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["+evaluate.backbones.%s=%s" % (v, bb), "evaluate.label_sizes=[null]", "evaluate.seeds=[%d]" % sd])
                jid = sbatch("r14-abl-%s-%s-s%d" % (v, log, sd), TIME["ablation"]["BPI17"] if log == "BPI17" else TIME["ablation"]["_"], args)
                jid = sbatch("r14-abl-%s-%s-s%d-c1" % (v, log, sd), TIME["ablation"]["BPI17"] if log == "BPI17" else TIME["ablation"]["_"], args, dep=jid)
