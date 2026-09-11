#!/usr/bin/env python3
"""Does the low-budget PFM/Scratch gap come from underfitting, not representation quality?

At 10-100 labels a frozen probe gets only tens of gradient steps under the default
probe settings (max_epochs 15, early_stop_patience 3). Scratch updates 4.9M parameters
over those same steps, so it can move far further in function space. This re-runs the
next-activity probe on three logs at budgets 10/30/100 with a much larger optimisation
budget, all three arms, everything else identical.

  BPI20ID  - largest gap (PFM 0.25 vs Scratch 0.82 @10)
  BPI17    - large gap    (0.28 vs 0.70)
  BPI13    - control: PFM already leads at 10 labels (0.64 vs 0.58)

If PFM's numbers jump and Scratch's do not, the gap was optimisation budget.
Dry-run by default; --submit to sbatch.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.expanduser("~/hpc_training_frozen")
spec = importlib.util.spec_from_file_location("seed3", os.path.join(ROOT, "seed3.py"))
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)

LOGS = ["BPI20ID", "BPI17", "bpi13_incidents"]
# the knobs under test; everything else is inherited from the original run
PATCH = [
    "evaluate.tasks=[next_activity]",
    "evaluate.label_sizes=[10,30,100]",
    "evaluate.probe.max_epochs=100",
    "evaluate.probe.early_stop_patience=10",
    "evaluate.seeds=[0,1]",
]
DROP = ("evaluate.tasks=", "evaluate.label_sizes=", "evaluate.seeds=",
        "evaluate.probe.max_epochs=", "evaluate.probe.early_stop_patience=")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    done = s3.completed()
    midx = s3.model_index()
    hyd = {}
    for f in sorted(__import__("glob").glob("outputs/hydra/*/.hydra/overrides.yaml")):
        ovr = s3.read_overrides(f)
        if "task=evaluate" not in ovr or "evaluate=label_efficiency" not in ovr:
            continue
        i = s3.parse(ovr)
        if i["dataset"] and i["backbones"]:
            hyd.setdefault((i["dataset"], frozenset(i["backbones"].items())), ovr)

    n = 0
    for log in LOGS:
        # the run that measured all three arms together
        key = next((k for k, cfg in done.items()
                    if k[0] == log and {a for a, _ in k[1]} == {"gin15allw1", "scratch"}), None)
        if key is None:
            print("no 3-arm run for %s -- skipped" % log)
            continue
        ovr = hyd.get(key) or s3.rebuild(done[key], midx)
        if ovr is None:
            print("could not rebuild overrides for %s -- skipped" % log)
            continue
        ovr = [o for o in ovr if not o.startswith(DROP)] + PATCH
        name = "ue-longprobe-%s" % log
        cmd = ["sbatch", "--job-name=" + name, "--time=01:00:00",
               "--gres=" + s3.SMALL_GRES, "--export=ALL", "slurm/ncps/run.sbatch"]
        env = dict(os.environ, ARGS=" ".join(ovr), USE_GPU="1")
        if args.submit:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " +
                  (r.stdout or r.stderr).strip())
            n += r.returncode == 0
        else:
            print(name + "\n      ARGS=" + " ".join(ovr))
        n += 0 if args.submit else 1
    print("\n%d jobs%s" % (n, " submitted" if args.submit else " (dry run)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
