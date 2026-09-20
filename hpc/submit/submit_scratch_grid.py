"""PFM-Scratch label-efficiency grid (r29): PFM's architecture trained from scratch on each target log.

Definition. The SAME model as PFM -- GIN-15 role encoder, 6-layer causal RoPE backbone, vocabulary-blind
input -- but randomly initialised and trained END TO END on the target log's labelled cases. It is
PFM-FT without pretraining, so PFM-FT minus PFM-Scratch isolates what pretraining contributes under full
fine-tuning. It is built from the `random_role` baseline (the untrained PFM architecture that Frozen
Random probes) with that alias added to `evaluate.finetune`.

Why not the built-in `scratch` baseline: that is a PLAIN transformer (role_dim=0) that learns embeddings
for the target log's activity IDs. It is a different, vocabulary-dependent model, not PFM from scratch,
and its existing runs predate the v2 protocol.

Protocol identical to the v2 main grid: budget corpus, max_trace_len 64, 8 budgets, 3 seeds, 7 tasks,
AdamW lr 1e-3 for all parameters, <=100 epochs, patience 10, best-validation weights restored.
New alias `pfm_scratch`, so collect_v2.py keeps these rows apart from every existing arm.

Usage (from the repository root):  python hpc/submit/submit_scratch_grid.py [--dry] [LOG ...]
"""
import os
import subprocess
import sys

LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""),
        "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", ""),
        "BPI17": ("/workspace/data/raw/BPI17.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar",
          "~evaluate.backbones.random", "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget",
          "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
ARM = ["+evaluate.backbones.pfm_scratch=random_role", "+evaluate.finetune=[pfm_scratch]"]
# From random init training may run longer than PFM-FT (v2 max on BPI17: 1h16). Generous limits.
TIME = {"BPI17": "12:00:00", "_": "04:00:00"}

DRY = "--dry" in sys.argv
only = [a for a in sys.argv[1:] if a != "--dry"]
assert os.path.exists("configs/model/role_gin15.yaml"), "run from the repository root"

n = 0
for log, (path, extra) in LOGS.items():
    if only and log not in only:
        continue
    for sd in (0, 1, 2):
        args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path]
                        + ([extra] if extra else []) + ARM
                        + ["evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
        name = "r29-scratch-%s-s%d" % (log, sd)
        tl = TIME.get(log, TIME["_"])
        n += 1
        if DRY:
            print("DRY  %-30s %s" % (name, tl))
            continue
        r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4",
                            "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                           env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
        print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
        if n == 1:
            print("     ARGS: " + args)
print(("would submit" if DRY else "submitted"), n, "jobs")
