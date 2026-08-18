"""Extract data/very-raw/ datasets, each into its own parent dir.

- <name>.zip  -> mkdir <name>/ ; extract the archive into it
- <name>.xes  -> mkdir <name>/ ; MOVE the xes into it (already correct format)
- inner *.xes.gz (and any *.gz) that land in a parent dir -> decompress to *.xes
- nested *.zip inside a parent dir (collections, e.g. Process Discovery Contest) -> left as-is, reported
Original top-level .zip archives are KEPT (downloads are not deleted).
"""

from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path

VR = Path("data/very-raw")
actions: list[str] = []

# 1) top-level files -> per-dataset parent dir
for f in sorted(VR.iterdir()):
    if f.is_dir() or f.suffix.lower() == ".xlsx":
        continue
    if f.suffix.lower() == ".zip":
        dest = VR / f.name[:-4]  # strip .zip
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(f) as z:
            z.extractall(dest)
        actions.append(f"[zip]  {f.name}  ->  {dest.name}/")
    elif f.suffix.lower() == ".xes":
        dest = VR / f.stem
        dest.mkdir(exist_ok=True)
        shutil.move(str(f), str(dest / f.name))
        actions.append(f"[xes]  {f.name}  ->  {dest.name}/ (moved)")

# 2) decompress any .gz that landed inside a parent dir (e.g. *.xes.gz -> *.xes)
for gz in sorted(VR.glob("*/**/*.gz")):
    out = gz.with_suffix("")  # drop trailing .gz
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
    gz.unlink()
    actions.append(f"[gz]   {gz.relative_to(VR)}  ->  {out.name}")

# 3) nested zips (collections) -> report, do not auto-extract
nested = sorted(str(z.relative_to(VR)) for z in VR.glob("*/**/*.zip"))

print("=== ACTIONS ===")
for a in actions:
    print(a)
if nested:
    print("\n=== NESTED ZIPS (collections; left for manual handling) ===")
    for n in nested:
        print(" ", n)
dirs = sorted(d.name for d in VR.iterdir() if d.is_dir())
print(f"\n=== {len(dirs)} PARENT DIRS CREATED ===")
for d in dirs:
    xes = list((VR / d).glob("**/*.xes"))
    print(f"  {d}/  ({len(xes)} .xes)")
print("DONE")
