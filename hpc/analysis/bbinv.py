import json,glob,os
def w(c,k,default=None):
    ar=c.get("ar") or {}; return ar.get(k,default)
print("%-52s %-6s %-8s %-6s %-6s %-6s %-6s %-6s"%("backbone","role","dim","jepa","time","rem","act","rolec"))
for d in sorted(glob.glob("outputs/backbones/*/")):
    mf=d+"manifest.json"
    if not os.path.exists(mf): continue
    m=json.load(open(mf)); c=m.get("config",{}); mdl=c.get("model") or {}
    name=os.path.basename(d.rstrip("/"))
    print("%-52s %-6s %-8s %-6s %-6s %-6s %-6s %-6s"%(name[:52], mdl.get("role_arch"), mdl.get("role_dim"),
        w(c,"jepa_weight"), w(c,"next_time_weight",w(c,"time_weight")), w(c,"remaining_time_weight"),
        w(c,"activity_weight",w(c,"next_activity_weight")), w(c,"role_contrast_weight")))
print()
print("=== gin15allw1 full ar block (the allw1 recipe) ===")
for d in glob.glob("outputs/backbones/*gin15allw1*/"):
    c=json.load(open(d+"manifest.json")).get("config",{})
    print(json.dumps({k:v for k,v in (c.get("ar") or {}).items() if "weight" in k or "jepa" in k},indent=1))
print("=== norole ar block (what it was trained with) ===")
for d in glob.glob("outputs/backbones/*norole*/"):
    c=json.load(open(d+"manifest.json")).get("config",{})
    print(json.dumps({k:v for k,v in (c.get("ar") or {}).items() if "weight" in k or "jepa" in k},indent=1))
