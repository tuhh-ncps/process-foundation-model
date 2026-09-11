import json, glob
for f in sorted(glob.glob("outputs/role_encoders/*/manifest.json")):
    m = json.load(open(f)); c = m.get("config", {}); mo = c.get("model", {}); vals = " ".join(v.get("path", "") for v in c.get("val_logs", []))
    if "Sepsis" in vals:
        mt = m.get("metrics") or {}
        print(f.split("/")[2], mo.get("role_arch"), mo.get("role_feature_subset"), mo.get("role_dim"), "best_epoch", mt.get("best_epoch"), "composite", mt.get("best_composite"), "done" if m.get("completed_at") else "RUNNING")
