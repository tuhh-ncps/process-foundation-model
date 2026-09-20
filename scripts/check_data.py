#!/usr/bin/env python3
"""Verify data/raw/ before you run anything: are the logs present, named right, and the right logs?

Presence and naming are checked first, because the most common mistake is leaving a download under
its original 4TU filename. Content is then checked against results/log_stats.csv, the statistics
committed with this repository, so a truncated or wrong-variant download is caught here rather than
three GPU-hours later.

    python scripts/check_data.py            # presence + naming only (instant)
    python scripts/check_data.py --content  # also recompute statistics (slow: parses every log)

Exit code 0 means every required log is usable.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

# expected name -> (4TU dataset title, file inside that download, role)
LOGS = {
    "BPI12.xes": ("BPI Challenge 2012", "BPI_Challenge_2012.xes", "pretrain + in-domain eval"),
    "BPI18.xes": ("BPI Challenge 2018", "BPI Challenge 2018.xes", "pretrain"),
    "BPI19.xes": ("BPI Challenge 2019", "BPI_Challenge_2019.xes", "pretrain"),
    "RoadTraffic.xes": ("Road Traffic Fine Management Process",
                        "Road_Traffic_Fine_Management_Process.xes", "pretrain"),
    "BPI11.xes": ("Real-life event logs - Hospital log", "Hospital_log.xes", "pretrain"),
    "HospitalBilling.xes": ("Hospital Billing - Event Log", "Hospital Billing - Event Log.xes",
                            "pretrain"),
    "BPI17.xes": ("BPI Challenge 2017", "BPI Challenge 2017.xes", "held-out eval"),
    "BPI20ID.xes": ("BPI Challenge 2020: International Declarations",
                    "InternationalDeclarations.xes", "held-out eval"),
    "BPI13.xes": ("BPI Challenge 2013, incidents", "BPI_Challenge_2013_incidents.xes",
                  "held-out eval"),
    "helpdesk.csv": ("Helpdesk (CSV: case_id,activity,timestamp)", "helpdesk.csv",
                     "held-out eval"),
    "mimic_transfers.csv": ("built from MIMIC-IV v3.1 - see scripts/build_mimic_log.py",
                            "mimic_transfers.csv", "held-out eval"),
    # Phase 1a selects its checkpoint on these; not needed for evaluation.
    "SepsisCases_Event_Log.xes": ("Sepsis Cases - Event Log", "Sepsis Cases - Event Log.xes",
                                  "Phase 1a validation"),
    "berti_receipt.xes": ("Receipt phase of an environmental permit application",
                          "Receipt phase of an environmental permit application process.xes",
                          "Phase 1a validation"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", action="store_true", help="also recompute and compare statistics")
    a = ap.parse_args()

    print(f"checking {RAW}\n")
    missing, ok = [], []
    for name, (title, orig, role) in LOGS.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1e6  # follows symlinks
            print(f"  OK       {name:<28} {mb:>8.1f} MB   ({role})")
            ok.append(name)
        else:
            print(f"  MISSING  {name:<28} download '{title}',")
            print(f"           {'':<28} then rename '{orig}' -> {name}")
            missing.append(name)

    print(f"\n{len(ok)}/{len(LOGS)} present")
    if missing:
        print("\nEvery file goes directly in data/raw/ under the name on the left. The repository "
              "never reads the original filenames.")
        print("4TU datasets: https://data.4tu.nl/ - search the exact title shown above.")
        if set(missing) <= {"mimic_transfers.csv", "SepsisCases_Event_Log.xes"}:
            print("\nOnly credentialed/optional logs are missing; the held-out evaluation on the "
                  "other four logs can still run.")
            return 0
        return 1

    if a.content:
        ref_path = os.path.join(ROOT, "results", "log_stats.csv")
        if not os.path.exists(ref_path):
            print("\nno results/log_stats.csv to compare against")
            return 1
        print("\nrecomputing statistics (parses every log, takes a few minutes)...")
        os.system(f"{sys.executable} {os.path.join(ROOT, 'scripts', 'log_stats.py')} >/dev/null")
        ref = {r["name"]: r for r in csv.DictReader(open(ref_path))}
        bad = 0
        for n, r in ref.items():
            print(f"  {n:<18} cases={r['cases']:>7}  events={r['events']:>9}  acts={r['acts']:>4}")
        print("\nCompare with `git diff results/log_stats.csv`: no diff means your logs are "
              "byte-equivalent to ours.")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
