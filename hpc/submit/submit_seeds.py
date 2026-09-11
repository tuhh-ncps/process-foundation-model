"""Pretraining-seed replication for the latent ablation: gin15 (jepa 1.0) and latent0 (jepa 0) x seeds {1,2},
same v2 recipe and the same v2 GIN-15 role encoder, pinned to a full H200. Usage: python submit_seeds.py"""
import glob, json, os, subprocess
DATA = "data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11]"
COMMON = "freeze_role=true ar.time_weight=1.0 ar.remaining_time_weight=1.0 model.id_dropout=1.0"
def role_run():
    hits = []
    for f in glob.glob("outputs/role_encoders/*/manifest.json"):
        m = json.load(open(f)); c = m.get("config", {}); mo = c.get("model", {})
        vals = " ".join(v.get("path", "") for v in c.get("val_logs", []))
        if "Sepsis" not in vals or (mo.get("role_arch"), mo.get("role_feature_subset", "all"), int(mo.get("role_dim", 64))) != ("gin", "all", 64): continue
        if m.get("completed_at") and os.path.exists(os.path.dirname(f) + "/role_encoder.pt"): hits.append((m["completed_at"], os.path.basename(os.path.dirname(f))))
    return sorted(hits)[-1][1]
role = role_run(); print("role encoder:", role)
for v, jepa in (("gin15", "1.0"), ("latent0", "0")):
    for seed in (1, 2):
        args = "model=role_gin15 %s %s ar.jepa_weight=%s tag=-v2-%s-s%d seed=%d role_init_from=%s" % (DATA, COMMON, jepa, v, seed, seed, role)
        r = subprocess.run(["sbatch", "--job-name=r19-bb-%s-s%d" % (v, seed), "--time=05:00:00", "--cpus-per-task=4", "--gres=gpu:nvidia_h200_nvl:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                           env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
        print(("OK   " if r.returncode == 0 else "FAIL ") + "%s-s%d  " % (v, seed) + (r.stdout or r.stderr).strip())
        if v == "gin15" and seed == 1: print("     ARGS: " + args)
