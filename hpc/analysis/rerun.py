#!/usr/bin/env python3
"""Re-run the label-efficiency grid with a fair probe optimisation budget.

The diagnostic showed the frozen probe was underfit at max_epochs=15/patience=3: PFM
gained up to +0.46 next-activity accuracy from more optimiser steps alone, while Scratch
(already converged) gained ~0.01-0.13. Every curve below ~300 labels is therefore
measuring convergence speed, not transferable structure.

Tier 1 = the five unseen logs with all three arms (Fig. 5 + the 35-setting claims).
Tier 2 = every other (variant, log) pair that already has results.

One job per (config, seed) so nothing approaches the 1-day partition limit; the seeds are
what the 3-seed sweep just established, so tier 1 lands 3 seeds per curve.
Dry-run by default; --submit to sbatch.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import os
import subprocess
import sys

ROOT = os.path.expanduser("~/hpc_training_frozen")
spec = importlib.util.spec_from_file_location("seed3", os.path.join(ROOT, "seed3.py"))
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)

TIER1_LOGS = ["BPI17", "BPI20ID", "bpi13_incidents", "helpdesk", "mimic_transfer"]
# Table 4's two pretraining rows (BPI12*, BPI18*), same three arms as tier 1
TIER4_LOGS = ["BPI12", "bpi18"]
# Table 6: the ablation variants, averaged over its four held-out logs
TIER3_LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]
TIER3_VARIANTS = {"gin15jepa1", "mlp15jepa1", "raw15jepa1", "gin11jepa1",
                  "gin0jepa1", "gin15jepa0", "norole"}
SEEDS = [0, 1, 2]
TASKS = ("[next_activity,next_3_activities,next_5_activities,next_time,"
         "remaining_time,remaining_count,future_activity_set]")
SIZES = "[0,10,30,100,300,1000,5000,null]"
TASK_LIST = ["next_activity", "next_3_activities", "next_5_activities", "next_time",
             "remaining_time", "remaining_count", "future_activity_set"]
PATCH = ["evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
         "evaluate.tasks=" + TASKS, "evaluate.label_sizes=" + SIZES]
DROP = ("evaluate.seeds=", "evaluate.probe.max_epochs=", "evaluate.probe.early_stop_patience=",
        "evaluate.tasks=", "evaluate.label_sizes=")

# 1-seed/15-epoch wall-clock measured from the seed-3 sweep, x5 for the epoch budget,
# then rounded up generously -- a job that dies on TIMEOUT wastes the whole grid.
MINUTES = {"BPI17": 77, "berti_billing": 60, "bpi18": 32, "BPI12": 25, "mimic_transfer": 12,
           "BPI20ID": 9, "bpi13_incidents": 6, "helpdesk": 3}


def limit(log: str) -> str:
    m = MINUTES.get(log, 30) * 5 * 2  # x5 epochs, x2 safety
    m = max(m, 120)
    return "%02d:%02d:00" % divmod(min(m, 20 * 60), 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=(1, 2, 3, 4), required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--logs", default=None, help="comma-separated log filter")
    ap.add_argument("--no-cap", action="store_true", help="strip eval_log.max_traces (uncapped)")
    ap.add_argument("--split-tasks", action="store_true",
                    help="one job per task -- 7x jobs, ~7x parallelism on a long log")
    args = ap.parse_args()
    os.chdir(ROOT)

    done = s3.completed()
    midx = s3.model_index()
    hyd = {}
    for f in sorted(glob.glob("outputs/hydra/*/.hydra/overrides.yaml")):
        ovr = s3.read_overrides(f)
        if "task=evaluate" not in ovr or "evaluate=label_efficiency" not in ovr:
            continue
        i = s3.parse(ovr)
        if i["dataset"] and i["backbones"]:
            hyd.setdefault((i["dataset"], frozenset(i["backbones"].items())), ovr)

    def sel(key, tier):
        log, bb = key
        arms = {a for a, _ in bb}
        if tier == 1:
            return log in TIER1_LOGS and arms == {"gin15allw1", "scratch"}
        if tier == 4:
            return log in TIER4_LOGS and arms == {"gin15allw1", "scratch"}
        if tier == 3:
            return log in TIER3_LOGS and len(arms) == 1 and arms <= TIER3_VARIANTS
        return not (sel(key, 1) or sel(key, 3) or sel(key, 4))  # tier 2 = the remainder

    keys = [k for k in done if sel(k, args.tier)]
    if args.logs:
        want = set(args.logs.split(","))
        keys = [k for k in keys if k[0] in want]
    keys.sort(key=lambda k: (-MINUTES.get(k[0], 30), str(k[0])))  # long pole first

    n = 0
    for key in keys:
        log, bb = key
        ovr = hyd.get(key) or s3.rebuild(done[key], midx)
        if ovr is None:
            print("SKIP (cannot rebuild overrides): %s %s" % (log, sorted(dict(bb))))
            continue
        base = [o for o in ovr if not o.startswith(DROP)] + PATCH
        if args.no_cap:
            base = [o for o in base if "eval_log.max_traces" not in o]
        alias = "-".join(sorted(dict(bb)))[:20]
        # one job per seed normally; per (seed, task) when splitting a long log, so the
        # 168 probes spread over every free slot instead of running serially in one job
        units = ([(sd, None) for sd in SEEDS] if not args.split_tasks
                 else [(sd, t) for sd in SEEDS for t in TASK_LIST])
        for sd, task in units:
            extra = ["evaluate.seeds=[%d]" % sd]
            if task:
                extra.append("evaluate.tasks=[%s]" % task)
                # one task of seven, x5 epochs, x3 safety
                tl = "%02d:00:00" % max(2, math.ceil(MINUTES.get(log, 30) * 5 / 7 * 3 / 60))
                name = "r%d-%s-%s-%s-s%d" % (args.tier, alias, log, task[:15], sd)
            else:
                tl, name = limit(log), "r%d-%s-%s-s%d" % (args.tier, alias, log, sd)
            # run.sbatch asks for 8 CPUs; the login node has 32, which caps it at 4 concurrent
            # jobs and strands the 5th GPU. num_workers=0, so 4 CPUs is ample.
            cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4",
                   "--gres=" + s3.SMALL_GRES, "--export=ALL", "slurm/ncps/run.sbatch"]
            env = dict(os.environ, USE_GPU="1", ARGS=" ".join(base + extra))
            if args.submit:
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                ok = r.returncode == 0
                n += ok
                print(("OK   " if ok else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
            else:
                print("%-58s %s" % (name, tl))
                n += 1
    print("\ntier %d: %d jobs%s" % (args.tier, n, " submitted" if args.submit else " (dry run)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
