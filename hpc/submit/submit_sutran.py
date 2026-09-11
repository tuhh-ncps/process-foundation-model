"""SuTraN+ (equal weighting, non-data-aware) on our five held-out logs under our exact protocol.
  python submit_sutran.py build            -> 5 CPU jobs: scripts/sutran_build.py (tensors from our splits + canonical queries)
  python submit_sutran.py train [LOG ...]  -> 3 GPU jobs per log (seeds 1,2,3), each afterok its build job, then our metrics
"""
import json, os, subprocess, sys
LOGS = {"helpdesk": ("HELPDESK", "06:00:00"), "bpi13_incidents": ("BPI13", "08:00:00"), "mimic_transfer": ("MIMIC", "06:00:00"),
        "BPI20ID": ("BPI20ID", "08:00:00"), "BPI17": ("BPI17", "23:59:00")}
SEEDS = (1, 2, 3)
IDS = "outputs/sutran/build_jobs.json"
os.makedirs("outputs/sutran", exist_ok=True)

FULL = "--gres=gpu:nvidia_h200_nvl:1"  # full H200 on cn01 (the login node only has MIG slices)
def sbatch(name, tl, cmd, gpu, dep=None, full=False, after_any=None):
    args = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--export=ALL"]
    if gpu: args.append(FULL if full else "--gres=gpu:1")
    if dep: args.append("--dependency=afterok:%s" % dep)
    if after_any: args.append("--dependency=afterany:%s" % after_any)
    args.append("slurm/ncps/run_cmd.sbatch")
    r = subprocess.run(args, env=dict(os.environ, USE_GPU="1" if gpu else "0", CMD=cmd), capture_output=True, text=True)
    jid = r.stdout.strip().split()[-1] if r.returncode == 0 else None
    print(("OK   " if jid else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip()); return jid

phase = sys.argv[1]; only = sys.argv[2:]
if phase == "build":
    ids = {}
    for log, (LOG, _) in LOGS.items():
        if only and log not in only: continue
        cmd = ("cd /workspace && python scripts/sutran_build.py --splits exports/%s_splits.csv --queries exports/%s_queries.csv "
               "--out external/SuTraN_Plus/%s --log %s --window 64" % (log, log, LOG, LOG))
        ids[log] = sbatch("r11-sutran-build-" + log, "02:00:00", cmd, gpu=False)
    old = json.load(open(IDS)) if os.path.exists(IDS) else {}; old.update(ids); json.dump(old, open(IDS, "w"))
elif phase == "train":
    ids = json.load(open(IDS)) if os.path.exists(IDS) else {}
    for log, (LOG, tl) in LOGS.items():
        if only and log not in only: continue
        for seed in SEEDS:
            res = "external/SuTraN_Plus/%s/SUTRAN_DA_results_seed_%d/CaLenDiR_training/Default_Equal_Weighting/TEST_SET_RESULTS" % (LOG, seed)
            cmd = ("cd /workspace/external/SuTraN_Plus && export PYTHONPATH=/workspace/external/pylib:$PYTHONPATH && export SUTRAN_RESUME=1 && "
                   "if [ -f /workspace/%s/suffix_acts_decoded.pt ]; then echo ALREADY-DONE; exit 0; fi && " % res +
                   "M=$(python -c \"import json;print(json.load(open(%r))[\\\"median_caselen\\\"])\") && "
                   "python -m TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_EQUAL_WEIGHTING --log_name %s --median_caselen $M --outcome_bool False "
                   "--out_mask False --clen_dis_ref True --out_type None --num_outclasses None --seed %d && cd /workspace && "
                   "python scripts/sutran_metrics.py --results %s --data external/SuTraN_Plus/%s --log-prefix %s --splits exports/%s_splits.csv "
                   "--log %s --seed %d --out outputs/sutran/%s_s%d.csv" % ("%s/build_summary.json" % LOG, LOG, seed, res, LOG, LOG, log, log, seed, log, seed))
            full = os.environ.get("FULLGPU") == "1"; chain = int(os.environ.get("CHAIN", "0"))
            jid = sbatch("r11-sutran-%s-s%d" % (log, seed), tl, cmd, gpu=True, dep=ids.get(log), full=full)
            for c in range(chain):  # continuation jobs: resume from the last checkpoint if the previous one timed out
                jid = sbatch("r11-sutran-%s-s%d-c%d" % (log, seed, c + 1), tl, cmd, gpu=True, full=full, after_any=jid)
