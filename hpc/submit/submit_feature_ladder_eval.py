"""Phase C of protocols/feature_ladder.md (r32): cached-feature next-activity evaluation of the GIN-k ladder.

  python hpc/submit/submit_feature_ladder_eval.py validate [--dry]   # C2 gate: mask plumbing, agreement, repeatability
  python hpc/submit/submit_feature_ladder_eval.py ladder   [--dry]   # C3: 16 backbones x 5 held-out logs x seeds {0,1,2}
  python hpc/submit/submit_feature_ladder_eval.py chain JOBID[:JOBID...] [--dry]
        # one job that starts after the given jobs succeed, computes the C2 verdict, and runs C3 only if it PASSED
  python hpc/submit/submit_feature_ladder_eval.py run      # (inside that job) gate check, then C3 sequentially
  python hpc/submit/submit_feature_ladder_eval.py tasks [--dry]
        # amendment A2: the six other tasks for all 16 backbones; a smoke job first, then 3 grouped jobs afterok

Every job is pinned to a full H200 so all feature budgets share one GPU type. Jobs only WRITE results
(outputs/feature_ladder/c2/*.jsonl, outputs/feature_ladder/eval/fbKK.jsonl); the C2 verdict is computed by
scripts/feature_ladder_analysis.py, and the ladder stage refuses to submit until that verdict file says PASSED (or PASSED_WITH_WAIVER,
when every failed check is waived by a protocol amendment in protocols/feature_ladder_waivers.json).
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
assert stage in ("validate", "ladder", "chain", "run", "tasks"), __doc__
TASKS6 = "next_3_activities,next_5_activities,future_activity_set,next_time,remaining_time,remaining_count"
BENCH = next((p for p in ("hpc/bench/bench_cached_pfm.py", "bench_cached_pfm.py") if os.path.exists(p)), None)
assert BENCH, "bench_cached_pfm.py not found (run from the repository root)"

art = json.load(open(ART))
assert art["status"] == "OK" and art["invariants"]["passed"], f"{ART} is not a passing Phase-A artifact"
order = art["order_code_indices"]


def bench(log: str, backbone: str, out: str, extra: str) -> str:
    return f"python {BENCH} {log} --backbone {backbone} --out {out} {extra}"


def submit(name: str, cmds: list[str], time_limit: str, dep: str | None = None) -> str | None:
    cmd = ("cd /workspace && set -e && mkdir -p outputs/feature_ladder/c2 outputs/feature_ladder/eval "
           "outputs/feature_ladder/eval_tasks && " + " && ".join(cmds))
    if DRY:
        print(f"DRY  {name}  ({len(cmds)} invocations, {time_limit}, dep={dep})\n     first: {cmds[0]}\n     last:  {cmds[-1]}")
        return "DRYJOB"
    extra = [f"--dependency=afterok:{dep}"] if dep else []
    r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + time_limit, "--cpus-per-task=4", "--mem=64G",
                        GRES] + extra + ["--export=ALL", "slurm/ncps/run_cmd.sbatch"],
                       env=dict(os.environ, USE_GPU="1", CMD=cmd), capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
    return r.stdout.strip().split()[-1] if r.returncode == 0 else None


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
    def ladder_cmds(k: int, backbone: str) -> list[str]:
        return [bench(L, backbone, f"outputs/feature_ladder/eval/fb{k:02d}.jsonl",
                      f"--seed {s} --tasks next_activity --skip-standard") for L in LOGS for s in SEEDS]

    if stage == "chain":
        deps = next((x for x in sys.argv[2:] if x != "--dry"), "")
        if not deps or not all(p.isdigit() for p in deps.split(":")):
            sys.exit("chain needs JOBID[:JOBID...]")
        cmd = ("cd /workspace && set -e && python scripts/feature_ladder_analysis.py c2 && "
               f"python {os.path.basename(__file__) if os.path.exists(os.path.basename(__file__)) else __file__} run")
        if DRY:
            print(f"DRY  r32-eval-chain  afterok:{deps}\n     CMD: {cmd}")
        else:
            r = subprocess.run(["sbatch", "--job-name=r32-eval-chain", "--time=10:00:00", "--cpus-per-task=4", "--mem=64G",
                                GRES, f"--dependency=afterok:{deps}", "--export=ALL", "slurm/ncps/run_cmd.sbatch"],
                               env=dict(os.environ, USE_GPU="1", CMD=cmd), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + "r32-eval-chain  " + (r.stdout or r.stderr).strip())
        sys.exit(0)

    verdict = json.load(open(C2_VERDICT)) if os.path.exists(C2_VERDICT) else {}
    if verdict.get("status") not in ("PASSED", "PASSED_WITH_WAIVER"):  # waivers: protocol amendments only
        sys.exit(f"C2 gate not passed ({C2_VERDICT}: {verdict.get('status')}); no ladder evaluation")
    backbones = sorted(ladder_backbones().items())
    if stage == "ladder":
        for k, backbone in backbones:
            submit(f"r32-eval-fb{k:02d}", ladder_cmds(k, backbone), "02:00:00")
    elif stage == "tasks":  # amendment A2 (exploratory): six more tasks, same backbones / logs / seeds / budget
        out = "outputs/feature_ladder/eval_tasks"
        smoke = submit("r33-tasks-smoke", [bench("helpdesk", GIN15, f"{out}/smoke.jsonl",
                                                 f"--seed 0 --tasks {TASKS6} --skip-standard")], "00:30:00")
        by_k = dict(backbones)
        for ks in ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12, 13, 14, 15)):
            cmds = [bench(L, by_k[k], f"{out}/fb{k:02d}.jsonl", f"--seed {s} --tasks {TASKS6} --skip-standard")
                    for k in ks for L in LOGS for s in SEEDS]
            submit(f"r33-tasks-fb{ks[0]:02d}-{ks[-1]:02d}", cmds, "08:00:00", dep=smoke)
    else:  # run: inside the chained job
        os.makedirs("outputs/feature_ladder/eval", exist_ok=True)
        for k, backbone in backbones:
            print(f"[ladder] k={k} {backbone}", flush=True)
            for c in ladder_cmds(k, backbone):
                subprocess.run(c, shell=True, check=True)
        print("[ladder] all 16 backbones evaluated", flush=True)
