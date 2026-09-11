"""PFM-RFT arm (alias pfm_rft): pretrained v2 gin15 backbone with the TRANSFORMER FROZEN, the ROLE ENCODER fine-tuned
together with the fresh head (partial fine-tuning; evaluate.finetune_role=[pfm_rft]). Same protocol as the v2 main grid
(PFM / PFM-FT): 7 tasks, 8 budgets, 3 seeds, budget corpus, max_trace_len 64, probes <=100 epochs / patience 10,
head lr 1e-3 and role-encoder lr 1e-3 (= backbone_lr default, as PFM-FT). One job per log/seed with afternotok
continuations (crash recovery resumes finished probes).  Usage: python submit_rft.py [--dry] [LOG ...]"""
import glob, json, os, subprocess, sys
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""), "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"), "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""), "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
TIME = {"BPI17": ("06:00:00", 1), "_": ("02:00:00", 1)}
PFM = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"   # the reported v2 backbone (seed 0)
DRY = "--dry" in sys.argv; only = [a for a in sys.argv[1:] if a != "--dry"]
m = json.load(open("outputs/backbones/%s/manifest.json" % PFM)); assert m.get("completed_at") and os.path.exists("outputs/backbones/%s/backbone.pt" % PFM)
assert os.path.exists("outputs/backbones/%s/role_encoder.pt" % PFM) or True
print("backbone:", PFM, "| role_arch", m["config"]["model"].get("role_arch"), "| freeze_role", m["config"].get("freeze_role"))
def sbatch(name, tl, args, dep=None):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL"]
    if dep: cmd.append("--dependency=afternotok:%s" % dep)
    cmd.append("slurm/ncps/run.sbatch")
    if DRY: print("DRY  " + name + "  " + " ".join(cmd[1:4])); return "0"
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
for log, (path, extra) in LOGS.items():
    if only and log not in only: continue
    tl, chain = TIME.get(log, TIME["_"])
    for sd in (0, 1, 2):
        args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else [])
                        + ["+evaluate.backbones.pfm_rft=" + PFM, "+evaluate.finetune_role=[pfm_rft]",
                           "evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
        jid = sbatch("r23-rft-%s-s%d" % (log, sd), tl, args)
        for c in range(chain):
            jid = sbatch("r23-rft-%s-s%d-c%d" % (log, sd, c + 1), tl, args, dep=jid)
        if sd == 0 and log == "helpdesk": print("     ARGS: " + args)
