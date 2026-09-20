"""Label-efficiency grid on the SEED-2 GIN-15 backbone (r28).

The main grid (submit_v2.py main) probed PFM and PFM-FT on the seed-0 backbone at all eight label
budgets. The seed-2 replica (backbone-20260907-103600-...-gin15-s2-bfb92b) was only probed at full
budget, by the pretraining-seed replication. This grid gives it full curves so the frozen_agg
figures can be drawn from it.

Arms get NEW aliases, pfm_s2 and pfm_ft_s2, so the collector keeps them apart from the seed-0 pfm /
pfm_ft rows and from the full-budget gin15_s2 rows instead of overwriting either. Frozen Random does
not depend on the backbone, so it is not re-run.

Protocol identical to the v2 main grid: budget corpus, max_trace_len 64, 8 budgets, 3 evaluation
seeds, 7 tasks, probes <=100 epochs with patience 10.

Usage (from the repository root):  python hpc/submit/submit_seed2_grid.py [--dry] [LOG ...]
"""
import json
import os
import subprocess
import sys

BACKBONE = "backbone-20260907-103600-multi-none-v2-gin15-s2-bfb92b"
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""),
        "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""),
        "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar",
          "~evaluate.backbones.random", "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget",
          "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
ARMS = {"pfm_s2": ["+evaluate.backbones.pfm_s2=" + BACKBONE],
        "pfm_ft_s2": ["+evaluate.backbones.pfm_ft_s2=" + BACKBONE, "+evaluate.finetune=[pfm_ft_s2]"]}
# v2 main grid maxima on BPI17: frozen 3h06, fine-tuned 1h16. Generous limits, no continuation jobs.
TIME = {"BPI17": "12:00:00", "_": "04:00:00"}

DRY = "--dry" in sys.argv
only = [a for a in sys.argv[1:] if a != "--dry"]

man = "outputs/backbones/%s/manifest.json" % BACKBONE
m = json.load(open(man))
assert m.get("completed_at") and os.path.exists("outputs/backbones/%s/backbone.pt" % BACKBONE), BACKBONE
assert int(m["config"].get("seed")) == 2, "expected the seed-2 backbone, got seed %r" % m["config"].get("seed")
print("backbone:", BACKBONE, "| seed", m["config"]["seed"], "| role", m["config"].get("role_init_from"))

n = 0
for log, (path, extra) in LOGS.items():
    if only and log not in only:
        continue
    for arm, ov in ARMS.items():
        for sd in (0, 1, 2):
            args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path]
                            + ([extra] if extra else []) + ov
                            + ["evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
            name = "r28-s2-%s-%s-s%d" % (log, arm, sd)
            tl = TIME.get(log, TIME["_"])
            n += 1
            if DRY:
                print("DRY  %-38s %s" % (name, tl))
                continue
            r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4",
                                "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                               env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
            if n == 1:
                print("     ARGS: " + args)
print(("would submit" if DRY else "submitted"), n, "jobs")
