"""Second pass: extract the nested zips left inside each parent dir, then decompress *.gz.

- data.zip                 -> flattened INTO the parent dir (it IS the dataset's content)
- <Named> Logs.zip / etc.  -> extracted into a subdir <Named> Logs/ (collection member)
- any resulting *.xes.gz    -> decompressed to *.xes
Intermediate nested zips are removed after extraction (reproducible from the kept top-level zip).
"""

from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path

VR = Path("data/very-raw")
actions: list[str] = []

for parent in sorted(d for d in VR.iterdir() if d.is_dir()):
    for z in sorted(parent.glob("*.zip")):  # nested zips directly under a parent dir
        dest = parent if z.stem.lower() == "data" else (parent / z.stem)
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
        actions.append(f"[nested] {z.relative_to(VR)}  ->  {dest.relative_to(VR)}/")

for gz in sorted(VR.glob("*/**/*.gz")):
    out = gz.with_suffix("")
    with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
    gz.unlink()
    actions.append(f"[gz] {gz.relative_to(VR)}  ->  {out.name}")

print(f"=== {len(actions)} NESTED ACTIONS ===")
for a in actions:
    print(a)

print("\n=== FINAL: .xes per parent dir ===")
for d in sorted(dd for dd in VR.iterdir() if dd.is_dir()):
    n = len(list(d.glob("**/*.xes")))
    leftover_zip = len(list(d.glob("**/*.zip")))
    flag = f"  (+{leftover_zip} nested zip still present)" if leftover_zip else ""
    print(f"  {d.name}/ : {n} .xes{flag}")
print("DONE")
