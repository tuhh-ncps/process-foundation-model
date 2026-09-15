"""Phase B of protocols/feature_ladder.md (r31): pretrain the zero-masked GIN-k ladder, k = 0..14, seed 0.

GIN-15 (k = 15) is the released backbone and is not retrained. Masks come ONLY from the frozen artifact
results/feature_ladder.json (Phase A); M_k = the first k code indices of its order.

Two stages, because Phase 1b needs the run id of its Phase-1a encoder:

  python hpc/submit/submit_feature_ladder.py role     [--dry]   # 15 Phase-1a role encoders (CPU, ~1 min each)
  python hpc/submit/submit_feature_ladder.py backbone [--dry]   # 15 Phase-1b backbones, each on its encoder

Phase 1a args are those of the GIN-15 role encoder (r14: task=role_pretrain role=frozen role.model.role_arch=gin
role.optim.epochs=120) plus the mask. Phase 1b args are those of the released GIN-15 backbone
(submit_pretrain_v2.py: gin15 variant) plus the same mask; ar_pretrain_hpc refuses a role encoder whose
recorded mask differs. Run from the repository root.
"""
import glob
import json
import os
import subprocess
import sys

ART = "results/feature_ladder.json"
PROTOCOL_FREEZE = "2026-09-15T00:00:00"          # role runs older than the protocol cannot belong to the ladder
KS = list(range(0, 15))
DATA = "data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11]"
BB_COMMON = "freeze_role=true ar.time_weight=1.0 ar.remaining_time_weight=1.0 model.id_dropout=1.0 ar.jepa_weight=1.0 seed=0"

stage = sys.argv[1] if len(sys.argv) > 1 else ""
DRY = "--dry" in sys.argv
assert stage in ("role", "backbone"), __doc__
assert os.path.exists("slurm/ncps/run.sbatch"), "run from the repository root"

art = json.load(open(ART))
assert art["status"] == "OK" and art["invariants"]["passed"], f"{ART} is not a passing Phase-A artifact"
order = art["order_code_indices"]
assert sorted(order) == list(range(15)), order
print(f"frozen order {art['order_letters']} (script blob {art['script_git_blob']}, protocol blob {art['protocol_git_blob']})")


def mask(k: int) -> list[int]:
    return sorted(order[:k])


def hydra_list(m: list[int]) -> str:
    return "[" + ",".join(str(i) for i in m) + "]"


def sbatch(name: str, args: str, extra: list[str], use_gpu: bool) -> None:
    cmd = ["sbatch", "--job-name=" + name, "--export=ALL"] + extra + ["slurm/ncps/run.sbatch"]
    if DRY:
        print(f"DRY  {name:18s} {' '.join(extra)}\n     ARGS: {args}")
        return
    env = dict(os.environ, USE_GPU="1" if use_gpu else "0", ARGS=args)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())


def role_runs() -> dict[str, list[tuple[str, str]]]:
    """Completed ladder-era GIN role encoders keyed by their recorded mask."""
    hits: dict[str, list[tuple[str, str]]] = {}
    for f in glob.glob("outputs/role_encoders/*/manifest.json"):
        m = json.load(open(f))
        mo = m.get("config", {}).get("model", {})
        if not m.get("completed_at") or m.get("created_at", "") < PROTOCOL_FREEZE:
            continue
        if mo.get("role_arch") != "gin" or mo.get("role_feature_subset", "all") != "all" or mo.get("role_feature_mask") is None:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(f), "role_encoder.pt")):
            continue
        key = hydra_list(sorted(int(i) for i in mo["role_feature_mask"]))
        hits.setdefault(key, []).append((m["completed_at"], m["run_id"]))
    return hits


if stage == "role":
    for k in KS:
        args = (f"task=role_pretrain role=frozen role.model.role_arch=gin role.optim.epochs=120 seed=0 "
                f"+role.model.role_feature_mask={hydra_list(mask(k))} tag=-v2-fb{k:02d}")
        sbatch(f"r31-role-fb{k:02d}", args, ["--time=01:00:00", "--cpus-per-task=8", "--mem=64G"], use_gpu=False)
else:
    runs = role_runs()
    missing = []
    for k in KS:
        key = hydra_list(mask(k))
        found = sorted(runs.get(key, []))
        if not found:
            missing.append(k)
            continue
        if len(found) > 1:
            print(f"WARN k={k}: {len(found)} role encoders with mask {key}; using the newest {found[-1][1]}")
        role = found[-1][1]
        args = (f"model=role_gin15 {DATA} {BB_COMMON} +model.role_feature_mask={key} "
                f"tag=-v2-fb{k:02d} role_init_from={role}")
        sbatch(f"r31-bb-fb{k:02d}", args, ["--time=05:00:00", "--cpus-per-task=4", "--gres=gpu:nvidia_h200_nvl:1"], use_gpu=True)
    if missing:
        sys.exit(f"no completed ladder role encoder for k = {missing}; run the role stage first")
