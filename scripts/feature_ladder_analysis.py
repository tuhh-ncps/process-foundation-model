#!/usr/bin/env python3
"""C2 gate and Phase-D aggregation for protocols/feature_ladder.md. Nothing here changes a threshold or an order.

    python scripts/feature_ladder_analysis.py c2     --c2-dir outputs/feature_ladder/c2      # -> results/feature_ladder_c2.json
    python scripts/feature_ladder_analysis.py report --eval-dir outputs/feature_ladder/eval  # -> results/feature_ladder_*.csv
    python scripts/feature_ladder_analysis.py --selftest

c2      C2.1 cached event states with an all-ones mask equal those with no mask key (max |diff| = 0) on every log;
        C2.2 cached GIN-15 next-activity accuracy vs the existing `pfm` rows of results/v2_all.csv:
             |diff| <= 0.010 per (log, seed) and <= 0.005 for each seed's five-log mean and the overall mean;
        C2.3 the repeated Helpdesk/seed-0 run differs by <= 0.002.
report  D1-D6: per-(k, log) seed means; mu_k; seed-wise mean and SD of the three five-log means (primary
        uncertainty); pooled and between-log SD (table only); paired differences to k = 15 with a t(4) 95% interval;
        k_near (primary: ladder's own k = 15; sensitivity: three-seed GIN-15 mean 0.7303), sigma_15 = 0.0058.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]
SEEDS = [0, 1, 2]
KS = list(range(16))
GIN15 = "backbone-20260906-153102-multi-none-v2-gin15-17fc3c"
SIGMA15 = 0.0058                       # D3b: SD of the five-log mean across GIN-15 17fc3c / ddeadf / bfb92b
MU15_THREE_SEED = 0.7303               # D4 sensitivity
T4 = 2.776                             # t_{4, 0.975}
TOL_PER_RUN, TOL_MEAN, TOL_REPEAT = 0.010, 0.005, 0.002


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def v2_rows(arm: str, task: str = "next_activity") -> dict[tuple[str, int], float]:
    out = {}
    for r in csv.DictReader(open(ROOT / "results" / "v2_all.csv")):
        if r["arm"] == arm and r["task"] == task and r["n_labels"] == "all" and r["log"] in LOGS:
            out[(r["log"], int(r["seed"]))] = float(r["value"])
    return out


# ----------------------------------------------------------------------------- C2
def c2(c2_dir: str, out: Path) -> dict:
    checks, ok = {}, True

    mask = {r["log"]: r for r in read_jsonl(f"{c2_dir}/mask_check.jsonl") if r.get("backbone") == GIN15}
    c21 = {log: (mask[log]["max_abs_diff"] if log in mask else None) for log in LOGS}
    c21_pass = all(log in mask and all(v == 0.0 for v in mask[log]["max_abs_diff"]["all_ones_mask"].values())
                   for log in LOGS)
    checks["C2.1_mask_plumbing"] = {"passed": c21_pass, "max_abs_diff": c21}
    ok &= c21_pass

    ref = v2_rows("pfm")
    got = {(r["log"], int(r["seed"])): r["cached"]["next_activity"]["value"]
           for r in read_jsonl(f"{c2_dir}/agreement.jsonl") if r.get("backbone") == GIN15}
    missing = [(log, s) for log in LOGS for s in SEEDS if (log, s) not in got or (log, s) not in ref]
    diffs = {f"{log}/s{s}": got[(log, s)] - ref[(log, s)] for log in LOGS for s in SEEDS if (log, s) not in missing}
    seed_mean_diff = {s: st.mean(got[(log, s)] for log in LOGS) - st.mean(ref[(log, s)] for log in LOGS)
                      for s in SEEDS if all((log, s) not in missing for log in LOGS)}
    overall = (st.mean(got[k] for k in ref if k in got) - st.mean(ref[k] for k in ref if k in got)) if not missing else None
    c22_pass = (not missing and all(abs(d) <= TOL_PER_RUN for d in diffs.values())
                and all(abs(d) <= TOL_MEAN for d in seed_mean_diff.values()) and abs(overall) <= TOL_MEAN)
    checks["C2.2_agreement"] = {"passed": c22_pass, "missing": missing, "per_run_diff": diffs,
                                "seed_five_log_mean_diff": seed_mean_diff, "overall_mean_diff": overall,
                                "max_abs_per_run": max((abs(d) for d in diffs.values()), default=None),
                                "tolerances": {"per_run": TOL_PER_RUN, "means": TOL_MEAN}}
    ok &= c22_pass

    rep = [r["cached"]["next_activity"]["value"] for r in read_jsonl(f"{c2_dir}/repeat.jsonl")
           if r.get("backbone") == GIN15 and r["log"] == "helpdesk" and int(r["seed"]) == 0]
    c23_pass = len(rep) == 2 and abs(rep[0] - rep[1]) <= TOL_REPEAT
    checks["C2.3_repeatability"] = {"passed": c23_pass, "values": rep,
                                    "abs_diff": abs(rep[0] - rep[1]) if len(rep) == 2 else None, "tolerance": TOL_REPEAT}
    ok &= c23_pass

    verdict = {"protocol": "protocols/feature_ladder.md", "status": "PASSED" if ok else "FAILED", "checks": checks}
    out.write_text(json.dumps(verdict, indent=2))
    return verdict


# ----------------------------------------------------------------------------- D
def load_eval(eval_dir: str) -> dict[tuple[int, str, int], float]:
    acc = {}
    for path in sorted(glob.glob(f"{eval_dir}/fb*.jsonl")):
        k = int(Path(path).stem[2:])
        for r in read_jsonl(path):
            acc[(k, r["log"], int(r["seed"]))] = r["cached"]["next_activity"]["value"]
    return acc


def report(acc: dict[tuple[int, str, int], float], artifact: dict, out_dir: Path) -> dict:
    missing = [(k, log, s) for k in KS for log in LOGS for s in SEEDS if (k, log, s) not in acc]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(KS) * len(LOGS) * len(SEEDS)} runs missing, e.g. {missing[:5]}")
    abar = {(k, log): st.mean(acc[(k, log, s)] for s in SEEDS) for k in KS for log in LOGS}          # D1
    var = {(k, log): st.variance([acc[(k, log, s)] for s in SEEDS]) for k in KS for log in LOGS}
    mu = {k: st.mean(abar[(k, log)] for log in LOGS) for k in KS}                                      # D2
    jover = {0: 0.0, **{s["k"]: s["J_over_15"] for s in artifact["steps"]}}
    enters = artifact["p_start_p_terminal_enter_at"]["first_of_either"]

    rows = []
    for k in KS:
        seed_means = [st.mean(acc[(k, log, s)] for log in LOGS) for s in SEEDS]                        # D3
        delta = [abar[(k, log)] - abar[(15, log)] for log in LOGS]                                     # D5
        d_mean = st.mean(delta)
        half = T4 * st.stdev(delta) / math.sqrt(len(LOGS)) if k != 15 else 0.0
        rows.append({
            "k": k, "J_over_15": jover[k], "mu": mu[k],
            "seedwise_mean": st.mean(seed_means), "seedwise_sd": st.stdev(seed_means),
            "pooled_sd": math.sqrt(sum(var[(k, log)] for log in LOGS)) / len(LOGS),
            "between_log_sd": st.stdev(abar[(k, log)] for log in LOGS),
            "delta_mean": d_mean, "delta_ci_low": d_mean - half, "delta_ci_high": d_mean + half,
            **{f"delta_{log}": d for log, d in zip(LOGS, delta)},
            **{f"acc_{log}": abar[(k, log)] for log in LOGS},
            "n_logs": len(LOGS), "n_eval_seeds": len(SEEDS),
        })

    def k_near(mu15: float) -> int:                                                                    # D4
        thr = mu15 - SIGMA15
        return min(k for k in KS if all(mu[j] >= thr for j in KS if j >= k))

    fr = v2_rows("random_role")
    frozen_random = st.mean(st.mean(fr[(log, s)] for s in SEEDS) for log in LOGS)
    summary = {"k_near_primary": k_near(mu[15]), "k_near_sensitivity": k_near(MU15_THREE_SEED),
               "mu15": mu[15], "sigma15": SIGMA15, "sigma15_source": "GIN-15 17fc3c, ddeadf, bfb92b (five-log mean SD)",
               "threshold_primary": mu[15] - SIGMA15, "threshold_sensitivity": MU15_THREE_SEED - SIGMA15,
               "frozen_random_reference": frozen_random, "case_start_or_end_enters_at_k": enters,
               "order_letters": artifact["order_letters"]}
    for r in rows:
        r["k_near_primary"], r["k_near_sensitivity"] = summary["k_near_primary"], summary["k_near_sensitivity"]
        r["sigma15"] = SIGMA15

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_ladder_next_activity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "log", "eval_seed", "accuracy"])
        for (k, log, s), v in sorted(acc.items()):
            w.writerow([k, log, s, v])
    with open(out_dir / "feature_ladder_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "feature_ladder_summary.json").write_text(json.dumps(summary, indent=2))
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------- selftest
def selftest() -> None:
    import random
    rng = random.Random(0)
    base = {log: 0.6 + 0.05 * i for i, log in enumerate(LOGS)}
    acc = {(k, log, s): base[log] + 0.004 * k + rng.gauss(0, 0.003) for k in KS for log in LOGS for s in SEEDS}
    art = {"steps": [{"k": k, "J_over_15": k / 15} for k in range(1, 16)], "order_letters": "X" * 15,
           "p_start_p_terminal_enter_at": {"first_of_either": 9}}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        res = report(acc, art, Path(tmp))
    rows = {r["k"]: r for r in res["rows"]}
    k = 7
    seed_means = [st.mean(acc[(k, log, s)] for log in LOGS) for s in SEEDS]
    assert abs(rows[k]["seedwise_sd"] - st.stdev(seed_means)) < 1e-12
    assert abs(rows[k]["mu"] - st.mean(seed_means)) < 1e-12                        # same mean either way
    d = [st.mean(acc[(k, log, s)] for s in SEEDS) - st.mean(acc[(15, log, s)] for s in SEEDS) for log in LOGS]
    assert abs(rows[k]["delta_ci_high"] - (st.mean(d) + T4 * st.stdev(d) / math.sqrt(5))) < 1e-12
    assert rows[15]["delta_mean"] == 0.0 and rows[15]["delta_ci_low"] == 0.0
    mu = {k: rows[k]["mu"] for k in KS}
    kn = res["summary"]["k_near_primary"]
    assert all(mu[j] >= mu[15] - SIGMA15 for j in KS if j >= kn) and (kn == 0 or mu[kn - 1] < mu[15] - SIGMA15 or
                                                                      any(mu[j] < mu[15] - SIGMA15 for j in KS if j >= kn - 1))
    print(f"selftest OK | k_near {kn} | mu_15 {mu[15]:.4f} | seed-wise SD at k=7 {rows[7]['seedwise_sd']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["c2", "report"])
    ap.add_argument("--c2-dir", default="outputs/feature_ladder/c2")
    ap.add_argument("--eval-dir", default="outputs/feature_ladder/eval")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.cmd == "c2":
        v = c2(a.c2_dir, ROOT / "results" / "feature_ladder_c2.json")
        print("C2", v["status"], {name: c["passed"] for name, c in v["checks"].items()})
        print(json.dumps(v["checks"]["C2.2_agreement"], indent=1)[:1500])
    elif a.cmd == "report":
        art = json.load(open(ROOT / "results" / "feature_ladder.json"))
        res = report(load_eval(a.eval_dir), art, ROOT / "results")
        print(json.dumps(res["summary"], indent=2))
    else:
        ap.error("give c2, report or --selftest")


if __name__ == "__main__":
    main()
