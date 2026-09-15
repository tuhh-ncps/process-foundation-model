"""Phase C of protocols/feature_ladder.md (r32): cached-feature next-activity evaluation of the GIN-k ladder.

  python hpc/submit/submit_feature_ladder_eval.py validate [--dry]   # C2 gate: mask plumbing, agreement, repeatability
  python hpc/submit/submit_feature_ladder_eval.py ladder   [--dry]   # C3: 16 backbones x 5 held-out logs x seeds {0,1,2}

Every job is pinned to a full H200 so all feature budgets share one GPU type. Jobs only WRITE results
(outputs/feature_ladder/c2/*.jsonl, outputs/feature_ladder/eval/fbKK.jsonl); the C2 verdict is computed by
scripts/feature_ladder_analysis.py, and the ladder stage refuses to submit until that verdict file says PASSED.
Run from the repository root.
"""
import glob
import json
import os
import subprocess
import sys

ART = "results/feature_ladder.json"
C2_VERDICT = "results/feature_ladder_c2.json"
GIN15 = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
LOGS = ["helpdesk", "mimic_transfer", "bpi13_incidents", "BPI20ID", "BPI17"]
SEEDS = [0, 1, 2]
GRES = "--gres=gpu:nvidia_h200_nvl:1"

stage = sys.argv[1] if len(sys.argv) > 1 else ""
DRY = "--dry" in sys.argv
assert stage in ("validate", "ladder"), __doc__
BENCH = next((p for p in ("hpc/bench/bench_cached_pfm.py", "bench_cached_pfm.py") if os.path.exists(p)), None)
assert BENCH, "bench_cached_pfm.py not found (run from the repository root)"

art = json.load(open(ART))
assert art["status"] == "OK" and art["invariants"]["passed"], f"{ART} is not a passing Phase-A artifact"
order = art["order_code_indices"]


def bench(log: str, backbone: str, out: str, extra: str) -> str:
    return f"python {BENCH} {log} --backbone {backbone} --out {out} {extra}"


def submit(name: str, cmds: list[str], time_limit: str) -> None:
    cmd = "cd /workspace && set -e && mkdir -p outputs/feature_ladder/c2 outputs/feature_ladder/eval && " + " && ".join(cmds)
    if DRY:
        print(f"DRY  {name}  ({len(cmds)} invocations, {time_limit})\n     first: {cmds[0]}\n     last:  {cmds[-1]}")
        return
    r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + time_limit, "--cpus-per-task=4", "--mem=64G",
                        GRES, "--export=ALL", "slurm/ncps/run_cmd.sbatch"],
                       env=dict(os.environ, USE_GPU="1", CMD=cmd), capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())


def ladder_backbones() -> dict[int, str]:
    found = {15: GIN15}
    for k in range(15):
        want = sorted(order[:k])
        hits = []
        for f in glob.glob(f"outputs/backbones/*-v2-fb{k:02d}-*/manifest.json"):
            m = json.load(open(f))
            d = os.path.dirname(f)
            if not (m.get("completed_at") and os.path.exists(os.path.join(d, "backbone.pt"))):
                continue
            got = m["config"]["model"].get("role_feature_mask")
            assert got is not None and sorted(int(i) for i in got) == want, f"{d}: mask {got} != frozen M_{k} {want}"
            hits.append(os.path.basename(d))
        assert len(hits) == 1, f"k={k}: expected exactly one completed backbone, found {hits}"
        found[k] = hits[0]
    return found


if stage == "validate":
    c2 = "outputs/feature_ladder/c2"
    cmds = [bench(L, GIN15, f"{c2}/mask_check.jsonl", "--mask-check") for L in LOGS]                       # C2.1
    cmds += [bench(L, GIN15, f"{c2}/agreement.jsonl", f"--seed {s} --tasks next_activity --skip-standard")  # C2.2
             for L in LOGS for s in SEEDS]
    cmds += [bench("helpdesk", GIN15, f"{c2}/repeat.jsonl", "--seed 0 --tasks next_activity --skip-standard")  # C2.3
             for _ in range(2)]
    submit("r32-c2-validate", cmds, "03:00:00")
else:
    verdict = json.load(open(C2_VERDICT)) if os.path.exists(C2_VERDICT) else {}
    assert verdict.get("status") == "PASSED", f"C2 gate not passed ({C2_VERDICT}: {verdict.get('status')})"
    for k, backbone in sorted(ladder_backbones().items()):
        cmds = [bench(L, backbone, f"outputs/feature_ladder/eval/fb{k:02d}.jsonl",
                      f"--seed {s} --tasks next_activity --skip-standard") for L in LOGS for s in SEEDS]
        submit(f"r32-eval-fb{k:02d}", cmds, "02:00:00")
