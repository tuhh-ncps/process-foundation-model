"""Phase 1b (v2): re-pretrain the seven backbones on the leak-fixed code with NO activity-ID information
(model.id_dropout=1.0) and act/time/rem/latent weights 1/1/1/1, each on its freshly trained v2 role encoder.
Usage: python submit_pretrain_v2.py [variant ...]   (variants: gin15 mlp15 raw15 gin11 gin0 latent0 norole)"""
import glob, json, os, subprocess, sys
DATA = "data=multi ++data.datasets=[bpi12_solo,bpi18,bpi19,road_traffic,hospital_billing,bpi11]"
COMMON = "freeze_role=true ar.time_weight=1.0 ar.remaining_time_weight=1.0 model.id_dropout=1.0"
#            model config    role arch tag   jepa   extra
VARIANTS = {"gin15":   ("role_gin15", "gin15", "1.0", ""),
            "mlp15":   ("role_mlp15", "mlp15", "1.0", ""),
            "raw15":   ("role_raw15", "raw15", "1.0", ""),
            "gin11":   ("role_gin11", "gin11", "1.0", ""),
            "gin0":    ("role_gin0",  "gin0",  "1.0", ""),
            "latent0": ("role_gin15", "gin15", "0",   ""),
            "norole":  ("role_gin15", None,    "1.0", "model.role_dim=0")}
SPEC = {"gin15": ("gin", "all", 64), "mlp15": ("mlp", "all", 64), "raw15": ("raw", "all", 15), "gin11": ("gin", "eleven", 64), "gin0": ("gin", "none", 64)}
def role_run(arch):
    """Newest COMPLETED role run with this architecture whose validation logs are the v2 ones (Sepsis/Receipt)."""
    want = SPEC[arch]; hits = []
    for f in glob.glob("outputs/role_encoders/*/manifest.json"):
        m = json.load(open(f)); c = m.get("config", {}); mo = c.get("model", {})
        vals = " ".join(v.get("path", "") for v in c.get("val_logs", []))
        if "Sepsis" not in vals: continue
        if (mo.get("role_arch"), mo.get("role_feature_subset", "all"), int(mo.get("role_dim", 64))) != want: continue
        if m.get("completed_at") and os.path.exists(os.path.dirname(f) + "/role_encoder.pt"): hits.append((m["completed_at"], os.path.basename(os.path.dirname(f))))
    return sorted(hits)[-1][1] if hits else None
only = sys.argv[1:]
for v, (cfg, arch, jepa, extra) in VARIANTS.items():
    if only and v not in only: continue
    role = role_run(arch) if arch else None
    if arch and not role: print("NO v2 ROLE RUN for", v, "(arch", arch + ")"); continue
    args = "model=%s %s %s ar.jepa_weight=%s tag=-v2-%s %s" % (cfg, DATA, COMMON, jepa, v, extra)
    if role: args += " role_init_from=%s" % role
    r = subprocess.run(["sbatch", "--job-name=r14-bb-" + v, "--time=04:00:00", "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"],
                       env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + v + "  " + (r.stdout or r.stderr).strip() + "  role=" + str(role))
    print("     ARGS: " + args)
