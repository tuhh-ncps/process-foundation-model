"""Linear-head variant of the three time/count probes (r26).

The v2 protocol gives the regression probes a hidden layer (probe.head_hidden=128, i.e.
Linear(256->128)->GELU->Dropout->Linear(128->1), 33,025 params). This grid re-runs ONLY the three
regression tasks with probe.head_hidden=0, a single Linear(256->1) (257 params) — a true linear
probe, matching the four activity tasks. head_hidden is part of the grid hash, so these runs get
their own checkpoints and cannot collide with the existing ones.

Arms: pfm (frozen), random_role (the MAE normaliser — must use the SAME head to stay comparable),
pfm_ft. Everything else is the v2 protocol: budget corpus, max_trace_len 64, 8 budgets, 3 seeds,
probes <=100 epochs with patience 10 on the validation partition.

Usage: python submit_linhead.py [--dry] [LOG ...]
"""
import os, subprocess, sys

LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""),
        "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""),
        "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_time,remaining_time,remaining_count]"
PFM = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar",
          "~evaluate.backbones.random", "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget",
          "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
          "evaluate.probe.head_hidden=0",            # <-- the whole point: linear regression heads
          "evaluate.tasks=" + TASKS]
ARMS = {"pfm": ["+evaluate.backbones.pfm=" + PFM],
        "random_role": ["+evaluate.backbones.random_role=random_role"],
        "pfm_ft": ["+evaluate.backbones.pfm_ft=" + PFM, "+evaluate.finetune=[pfm_ft]"]}
TIME = {"BPI17": "12:00:00", "_": "04:00:00"}

DRY = "--dry" in sys.argv
only = [a for a in sys.argv[1:] if a != "--dry"]
n = 0
for log, (path, extra) in LOGS.items():
    if only and log not in only:
        continue
    for arm, ov in ARMS.items():
        for sd in (0, 1, 2):
            args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path]
                            + ([extra] if extra else []) + ov
                            + ["evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
            name = "r26-lin-%s-%s-s%d" % (log, arm, sd)
            tl = TIME.get(log, TIME["_"])
            if DRY:
                print("DRY  %-34s %s" % (name, tl)); n += 1; continue
            r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4",
                                "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                               env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
            n += 1
            if n == 1:
                print("     ARGS: " + args)
print("submitted" if not DRY else "would submit", n, "jobs")
