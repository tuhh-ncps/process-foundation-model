"""v3 check on the four small held-out logs: PFM (frozen) and PFM-FT, 8 budgets, 3 seeds, budget corpus,
on the single v3 gin15 backbone. Verifies the backbone manifest + logged losses first. Usage: python submit_v3_small.py"""
import csv, glob, json, os, subprocess, sys
LOGS = {"helpdesk": ("/workspace/data/raw/helpdesk.csv", ""), "bpi13_incidents": ("/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""),
        "mimic_transfer": ("/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"), "BPI20ID": ("/workspace/data/raw/BPI20ID.xes", "")}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
hits = []
for f in glob.glob("outputs/backbones/*-v3-gin15-*/manifest.json"):
    d = os.path.dirname(f); n = os.path.basename(d); m = json.load(open(f))
    if "-v3-gin15-s" in n or not m.get("completed_at") or not os.path.exists(d + "/backbone.pt"): continue
    c = m["config"]; a = c["ar"]
    assert float(a["outcome_weight"]) == 0 and float(a["role_contrast_weight"]) == 0 and float(a["jepa_weight"]) == 1.0, (n, a)
    assert float(c["model"]["id_dropout"]) == 1.0 and c["freeze_role"] is True and c["seed"] == 0, n
    cols = next(csv.reader(open(d + "/learning_curve.csv")))
    assert "train/ar_outcome_loss" not in cols and "train/ar_role_loss" not in cols, cols
    hits.append((m["completed_at"], n, cols))
assert hits, "no completed v3 gin15 backbone"
bb = sorted(hits)[-1][1]; print("backbone:", bb, "| logged losses:", sorted(hits)[-1][2])
for log, (path, extra) in LOGS.items():
    for arm, ov in {"pfm": ["+evaluate.backbones.pfm=" + bb], "pfm_ft": ["+evaluate.backbones.pfm_ft=" + bb, "+evaluate.finetune=[pfm_ft]"]}.items():
        for sd in (0, 1, 2):
            args = " ".join(COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ov + ["evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.seeds=[%d]" % sd])
            r = subprocess.run(["sbatch", "--job-name=r21-v3-%s-%s-s%d" % (log, arm, sd), "--time=03:00:00", "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                               env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + "%s %s s%d  " % (log, arm, sd) + (r.stdout or r.stderr).strip())
