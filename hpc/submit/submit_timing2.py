"""Controlled timing runs pinned to a FULL H200 (gres gpu:nvidia_h200_nvl:1), chained one after another (afterany)
so only one timing job runs at a time. One seed. PFM frozen + PFM-FT (7 tasks, full budget), FM-v2 (both tasks,
test queries only, validation-selected k), SuTraN seed 4 on BPI13 + MIMIC (full-GPU runs exist for the other logs)."""
import glob, json, os, subprocess
GRES = "--gres=gpu:nvidia_h200_nvl:1"
LOGS = [("helpdesk", "HELPDESK", "/workspace/data/raw/helpdesk.csv", ""), ("mimic_transfer", "MIMIC", "/workspace/data/raw/mimic_transfers.csv", "+evaluate.eval_log.max_traces=5000"),
        ("bpi13_incidents", "BPI13", "/workspace/data/raw/BPI_Challenge_2013_incidents.xes", ""), ("BPI20ID", "BPI20ID", "/workspace/data/raw/BPI20ID.xes", ""), ("BPI17", "BPI17", "/workspace/data/raw/BPI17.xes", "")]
KS = {"helpdesk": "200 100 5", "mimic_transfer": "200 50 100", "bpi13_incidents": "100 200", "BPI20ID": "1 5 100", "BPI17": "100 20 200"}
TASKS = "[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]"
COMMON = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
          "+evaluate.max_trace_len=64", "evaluate.role_corpus=budget", "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10", "evaluate.tasks=" + TASKS]
hits = []
for f in glob.glob("outputs/backbones/*/manifest.json"):
    m = json.load(open(f)); d = os.path.dirname(f)
    if os.path.basename(d).split("-v2-")[-1].rsplit("-", 1)[0] == "gin15" and m.get("completed_at") and os.path.exists(d + "/backbone.pt"):
        hits.append((m["completed_at"], os.path.basename(d)))
pfm = sorted(hits)[-1][1]; print("backbone:", pfm)
os.makedirs("outputs/fmv2_timing", exist_ok=True); os.makedirs("outputs/sutran", exist_ok=True)
prev = None
def sbatch(name, tl, script, env, dep):
    cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", GRES, "--export=ALL"]
    if dep: cmd.append("--dependency=afterany:%s" % dep)
    cmd.append(script)
    r = subprocess.run(cmd, env=dict(os.environ, **env), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid
for log, LOG, path, extra in LOGS:
    big = log == "BPI17"
    base = COMMON + ["evaluate.eval_dataset=" + log, "evaluate.eval_log.path=" + path] + ([extra] if extra else []) + ["evaluate.label_sizes=[null]", "evaluate.seeds=[0]"]
    prev = sbatch("r16-time-pfm-" + log, "04:00:00" if big else "01:00:00", "slurm/ncps/run.sbatch", {"USE_GPU": "1", "ARGS": " ".join(base + ["+evaluate.backbones.pfm=" + pfm])}, prev)
    prev = sbatch("r16-time-ft-" + log, "04:00:00" if big else "01:00:00", "slurm/ncps/run.sbatch", {"USE_GPU": "1", "ARGS": " ".join(base + ["+evaluate.backbones.pfm_ft=" + pfm, "+evaluate.finetune=[pfm_ft]"])}, prev)
    cmd = ("cd /workspace && python scripts/fmv2_eval.py --repo external/events-transf --checkpoint-dir external/fmv2_ckpt/checkpoints "
           "--splits exports/%s_splits.csv --queries exports/%s_queries.csv --log %s --tasks next_activity,remaining_time --query-splits test "
           "--k %s --budgets all --seeds 0 --device cuda --out outputs/fmv2_timing/%s.csv" % (log, log, log, KS[log], log))
    prev = sbatch("r16-time-fmv2-" + log, "04:00:00" if big else "01:00:00", "slurm/ncps/run_cmd.sbatch", {"USE_GPU": "1", "CMD": cmd}, prev)
    if log in ("mimic_transfer", "bpi13_incidents"):
        seed = 4
        res = "external/SuTraN_Plus/%s/SUTRAN_DA_results_seed_%d/CaLenDiR_training/Default_Equal_Weighting/TEST_SET_RESULTS" % (LOG, seed)
        cmd = ("cd /workspace/external/SuTraN_Plus && export PYTHONPATH=/workspace/external/pylib:$PYTHONPATH && export SUTRAN_RESUME=1 && "
               "M=$(python -c \"import json;print(json.load(open(%r))[\\\"median_caselen\\\"])\") && "
               "python -m TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_EQUAL_WEIGHTING --log_name %s --median_caselen $M --outcome_bool False "
               "--out_mask False --clen_dis_ref True --out_type None --num_outclasses None --seed %d && cd /workspace && "
               "python scripts/sutran_metrics.py --results %s --data external/SuTraN_Plus/%s --log-prefix %s --splits exports/%s_splits.csv "
               "--log %s --seed %d --out outputs/sutran/%s_s%d.csv" % ("%s/build_summary.json" % LOG, LOG, seed, res, LOG, LOG, log, log, seed, log, seed))
        prev = sbatch("r16-time-sutran-" + log, "03:00:00", "slurm/ncps/run_cmd.sbatch", {"USE_GPU": "1", "CMD": cmd}, prev)
print("last job:", prev)
