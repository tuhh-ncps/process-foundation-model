#!/usr/bin/env python3
"""Queue the 3rd ablation seed.

For every COMPLETED label-efficiency eval, re-runs the same evaluation with only
`evaluate.seeds` changed, so each (variant, eval log) ends up with seeds 0,1,2.

Overrides come from the run's own hydra record where one survives; otherwise they are
rebuilt from the run manifest by diffing its resolved config against
configs/evaluate/label_efficiency.yaml. Also expands the bpi13_incidents ablation grid
onto mimic_transfer.

Dry-run by default; pass --submit to actually sbatch.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.expanduser("~/hpc_training_frozen")

# 1-seed wall-clock, from sacct on the 2-seed originals: halved, then padded.
TIME = {
    "BPI17": "03:00:00", "BPI12": "01:30:00", "bpi18": "01:30:00",
    "berti_billing": "01:30:00", "BPI20ID": "00:45:00",
    "bpi13_incidents": "00:45:00", "mimic_transfer": "00:45:00",
    "helpdesk": "00:30:00",
}
DEFAULT_TIME = "02:00:00"
BIG_GRES = "gpu:nvidia_h200_nvl_3g.71gb:1"    # one slice; hand it to the longest jobs
SMALL_GRES = "gpu:nvidia_h200_nvl_1g.18gb:1"  # four slices
MIMIC_PATH = "/workspace/data/raw/mimic_transfers.csv"

# config keys worth diffing against the defaults (scalars/lists hydra can take verbatim)
DIFF_KEYS = ["strip_to_control_flow", "label_sizes", "pooling", "split",
             "warm_start", "length_bucketing", "x_scale"]
MODEL_KEYS = ["role_arch", "role_dim", "role_layers", "role_feature_subset", "aggregator"]


def fmt(v) -> str:
    """Render a python value the way a hydra override expects it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return str(v)


def read_overrides(path: str) -> list[str]:
    out = []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("- "):
            v = line[2:].strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out.append(v)
    return out


def parse(ovr: list[str]) -> dict:
    d = {"dataset": None, "backbones": {}, "model": None}
    for o in ovr:
        if o.startswith("evaluate.eval_dataset="):
            d["dataset"] = o.split("=", 1)[1]
        elif o.startswith("+evaluate.backbones."):
            k, v = o[len("+evaluate.backbones."):].split("=", 1)
            d["backbones"][k] = v
        elif o.startswith("model="):
            d["model"] = o.split("=", 1)[1]
    return d


def model_index() -> dict:
    """(role fields) -> configs/model name, for rebuilding a lost `model=` choice."""
    idx = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "configs/model/*.yaml"))):
        d = {}
        for line in open(f):
            m = re.match(r"^(%s):\s*([^\s#]+)" % "|".join(MODEL_KEYS), line)
            if m:
                d[m.group(1)] = m.group(2)
        if "role_arch" not in d:
            continue
        idx.setdefault(tuple(d.get(k) for k in MODEL_KEYS), os.path.basename(f)[:-5])
    return idx


def rebuild(cfg: dict, midx: dict) -> list[str] | None:
    """Reconstruct the override list for a run from its resolved manifest config."""
    defaults = yaml.safe_load(open(os.path.join(ROOT, "configs/evaluate/label_efficiency.yaml")))
    mdl = cfg.get("model") or {}
    key = tuple(None if mdl.get(k) is None else str(mdl.get(k)) for k in MODEL_KEYS)
    name = midx.get(key)
    if not name:
        return None
    ovr = ["task=evaluate", "evaluate=label_efficiency", "model=" + name,
           "~evaluate.backbones.ar"]
    bbs = cfg.get("backbones") or {}
    for alias, rid in sorted(bbs.items()):
        if alias != "random":
            ovr.append("+evaluate.backbones.%s=%s" % (alias, rid))
    if "random" not in bbs:
        ovr.append("~evaluate.backbones.random")
    ovr.append("evaluate.eval_dataset=" + str(cfg.get("eval_dataset")))
    el = cfg.get("eval_log") or {}
    if el.get("path"):
        ovr.append("evaluate.eval_log.path=" + el["path"])
    if el.get("max_traces") is not None:
        ovr.append("+evaluate.eval_log.max_traces=%s" % el["max_traces"])
    ovr.append("evaluate.tasks=" + fmt(cfg.get("tasks")))
    for k in DIFF_KEYS:
        if k in cfg and cfg[k] != defaults.get(k):
            ovr.append("evaluate.%s=%s" % (k, fmt(cfg[k])))
    for k, v in (cfg.get("probe") or {}).items():
        if v != (defaults.get("probe") or {}).get(k):
            ovr.append("evaluate.probe.%s=%s" % (k, fmt(v)))
    for k, v in (cfg.get("outcome") or {}).items():
        if v != (defaults.get("outcome") or {}).get(k):
            ovr.append("evaluate.outcome.%s=%s" % (k, fmt(v)))
    return ovr


def completed() -> dict:
    """(dataset, frozenset(alias->id)) -> resolved config, for finished runs."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "outputs/label_efficiency/*/"))):
        m = os.path.join(d, "manifest.json")
        if not os.path.exists(m):
            continue
        try:
            c = json.load(open(m))
        except Exception:
            continue
        if not c.get("completed_at"):
            continue
        cf = c.get("config", {})
        bbs = {k: v for k, v in (cf.get("backbones") or {}).items() if k != "random"}
        if not bbs:
            continue
        out[(cf.get("eval_dataset"), frozenset(bbs.items()))] = cf
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the job name")
    ap.add_argument("--verbose", action="store_true", help="dry run: print full ARGS")
    args = ap.parse_args()
    os.chdir(ROOT)

    done = completed()
    midx = model_index()

    # hydra records, keyed the same way as the completed runs
    hyd = {}
    for f in sorted(glob.glob("outputs/hydra/*/.hydra/overrides.yaml")):
        ovr = read_overrides(f)
        if "task=evaluate" not in ovr or "evaluate=label_efficiency" not in ovr:
            continue
        i = parse(ovr)
        if i["dataset"] and i["backbones"]:
            hyd.setdefault((i["dataset"], frozenset(i["backbones"].items())), ovr)

    jobs, unresolved = [], []
    for key, cfg in sorted(done.items(), key=lambda kv: (str(kv[0][0]), str(sorted(dict(kv[0][1]))))):
        ds, bb = key
        src = "hydra"
        ovr = hyd.get(key)
        if ovr is None:
            ovr, src = rebuild(cfg, midx), "manifest"
            if ovr is None:
                unresolved.append(key)
                continue
        base = [o for o in ovr if not o.startswith("evaluate.seeds")]
        jobs.append({"ovr": base + ["evaluate.seeds=[2]"], "ds": ds,
                     "bb": dict(bb), "seeds": "[2]", "src": src, "why": "3rd seed"})

    # expand the bpi13_incidents ablation grid onto mimic_transfer
    mimic_done = {frozenset(b) for d, b in done if d == "mimic_transfer"}
    already = {frozenset(x["bb"].items()) for x in jobs if x["ds"] == "mimic_transfer"}
    for j in [x for x in jobs if x["ds"] == "bpi13_incidents"]:
        bb = frozenset(j["bb"].items())
        if bb in already:
            continue
        already.add(bb)
        seeds = "[2]" if bb in mimic_done else "[0,1,2]"
        ovr = []
        for o in j["ovr"]:
            if o.startswith("evaluate.eval_dataset="):
                o = "evaluate.eval_dataset=mimic_transfer"
            elif o.startswith("evaluate.eval_log.path="):
                o = "evaluate.eval_log.path=" + MIMIC_PATH
            elif o.startswith("evaluate.seeds="):
                o = "evaluate.seeds=" + seeds
            ovr.append(o)
        jobs.append({"ovr": ovr, "ds": "mimic_transfer", "bb": j["bb"],
                     "seeds": seeds, "src": j["src"], "why": "mimic ablation"})

    # longest jobs get the single big MIG slice
    order = sorted(range(len(jobs)), key=lambda i: -_secs(TIME.get(jobs[i]["ds"], DEFAULT_TIME)))
    big = set(order[:6])

    per_ds, nsub = collections.Counter(), 0
    for i, j in enumerate(jobs):
        alias = "-".join(sorted(j["bb"]))[:22]
        name = "s3-%s-%s" % (alias, j["ds"])
        if args.only and args.only not in name:
            continue
        per_ds[j["ds"]] += 1
        tl = TIME.get(j["ds"], DEFAULT_TIME)
        if j["seeds"] == "[0,1,2]":
            tl = "%02d:%02d:00" % divmod(min(3 * _secs(tl) // 60, 23 * 60), 60)
        gres = BIG_GRES if i in big else SMALL_GRES
        cmd = ["sbatch", "--job-name=" + name, "--time=" + tl, "--gres=" + gres,
               "--export=ALL", "slurm/ncps/run.sbatch"]
        env = dict(os.environ, ARGS=" ".join(j["ovr"]), USE_GPU="1")
        if args.submit:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            ok = r.returncode == 0
            nsub += ok
            print(("OK   " if ok else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
        else:
            print("%-44s %-9s seeds=%-8s %-8s %-9s %s"
                  % (name, tl, j["seeds"], gres.split("_")[-1], j["src"], j["why"]))
            if args.verbose:
                print("      ARGS=" + " ".join(j["ovr"]))

    print("\n%d jobs%s (%s)" % (sum(per_ds.values()), " SUBMITTED=%d" % nsub if args.submit else "",
                               ", ".join("%s=%d" % kv for kv in per_ds.most_common())))
    if unresolved:
        print("UNRESOLVED (no hydra record, model config unmatched):")
        for ds, bb in unresolved:
            print("   %-18s %s" % (ds, ",".join(sorted(dict(bb)))))
    return 0


def _secs(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


if __name__ == "__main__":
    sys.exit(main())
