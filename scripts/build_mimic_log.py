#!/usr/bin/env python3
"""Convert MIMIC-IV (Story A: patient flow through care units) into a PM event log.

Reads two MIMIC-IV *hosp* tables and emits a flat CSV that the project's CSV reader
loads with no column mapping (canonical ``case_id,activity,timestamp`` columns):

    * ``hosp/transfers.csv.gz``  -> one row per ward movement (careunit + intime)
    * ``hosp/admissions.csv.gz`` -> mortality + 30-day readmission (subject/admit/disch times)

One admission (``hadm_id``) = one trace. Its activities are the ordered care units the
patient passed through. We then append up to two terminal events per admission, both at
``dischtime``:

    1. mortality terminal — ``EXPIRED`` if the patient died in hospital, else ``DISCHARGED``.
    2. readmission terminal (SURVIVORS only) — ``READMIT_30D`` if the SAME patient has a
       later admission whose ``admittime`` is within 30 days of this discharge, else
       ``NO_READMIT_30D``. Expired admissions get no readmission terminal (they cannot be
       readmitted, so they are excluded from the readmission cohort).

The terminals are what the ``mimic_mortality`` / ``mimic_readmission_30d`` labelers read (and
strip) to supervise outcome heads leak-free, mirroring how BPI'12 uses its terminal ``A_``
activities. The care-unit events themselves drive the next-activity / next-Δt / remaining-time
pretext heads, and the length-of-stay / ICU-admission labelers derive their targets from them.

Run it on the cluster (where the data lives) inside the container:

    apptainer exec pmfoundation.sif python scripts/build_mimic_log.py \
        --mimic data/raw/mimic-iv-v3.1/physionet.org/files/mimiciv/3.1 \
        --out   data/raw/mimic_transfers.csv

Add ``--max-cases 2000`` for a fast smoke run before committing to the full log. (Readmission
is computed over ALL admissions, so a capped run still labels the kept admissions correctly.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_READMISSION_DAYS = 30  # a subsequent admission within this many days of discharge = readmission


def _hosp(root: Path, name: str) -> Path:
    """Locate a hosp table, tolerating both gzipped and plain CSV."""
    for cand in (root / "hosp" / f"{name}.csv.gz", root / "hosp" / f"{name}.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"could not find hosp/{name}.csv[.gz] under {root}")


def build_log(mimic_root: Path, max_cases: int | None = None) -> pd.DataFrame:
    """Return a long event-log DataFrame with columns case_id, activity, timestamp."""
    transfers = pd.read_csv(
        _hosp(mimic_root, "transfers"),
        usecols=["hadm_id", "careunit", "intime"],
        parse_dates=["intime"],
    )
    # Keep only real ward stays: an admission-linked row with a named care unit and a
    # timestamp. This drops ED-only rows (null hadm_id) and discharge rows (null careunit) —
    # our own terminal event replaces the latter.
    transfers = transfers.dropna(subset=["hadm_id", "careunit", "intime"])
    transfers["hadm_id"] = transfers["hadm_id"].astype("int64")

    if max_cases is not None:
        keep = transfers["hadm_id"].drop_duplicates().head(max_cases)
        transfers = transfers[transfers["hadm_id"].isin(keep)]

    events = transfers.rename(
        columns={"hadm_id": "case_id", "careunit": "activity", "intime": "timestamp"}
    )[["case_id", "activity", "timestamp"]]
    kept = set(events["case_id"].unique())

    # Admissions table drives BOTH terminals. Read subject_id + admittime so 30-day readmission
    # can be computed as a same-patient join, over ALL admissions (not just the kept subset, so a
    # --max-cases smoke run still sees each kept admission's next stay).
    adm = pd.read_csv(
        _hosp(mimic_root, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime"],
    )
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    dated = adm.dropna(subset=["admittime", "dischtime"]).sort_values(["subject_id", "admittime"])
    next_admit = dated.groupby("subject_id")["admittime"].shift(-1)
    gap_days = (next_admit - dated["dischtime"]).dt.total_seconds() / 86_400.0
    readmit = (gap_days >= 0) & (gap_days <= _READMISSION_DAYS)
    readmit_by_hadm = pd.Series(readmit.to_numpy(), index=dated["hadm_id"].to_numpy())

    # Restrict terminals to admissions that actually have >=1 care-unit event (so every trace
    # gets a terminal), keeping a valid dischtime.
    admk = adm[adm["hadm_id"].isin(kept)].dropna(subset=["dischtime"])

    mortality = pd.DataFrame(
        {
            "case_id": admk["hadm_id"],
            "activity": admk["hospital_expire_flag"].map({1: "EXPIRED", 0: "DISCHARGED"}),
            "timestamp": admk["dischtime"],
            "_kind": 1,
        }
    )
    # Readmission terminal for SURVIVORS only (expired patients cannot be readmitted).
    surv = admk[admk["hospital_expire_flag"] == 0]
    surv_readmit = surv["hadm_id"].map(readmit_by_hadm).fillna(False)
    readmission = pd.DataFrame(
        {
            "case_id": surv["hadm_id"],
            "activity": surv_readmit.map({True: "READMIT_30D", False: "NO_READMIT_30D"}),
            "timestamp": surv["dischtime"],
            "_kind": 2,
        }
    )

    events = events.assign(_kind=0)
    log = pd.concat([events, mortality, readmission], ignore_index=True)
    # Stable order within a case: care-unit events by time (_kind=0), then the mortality terminal
    # (_kind=1), then the readmission terminal (_kind=2). Both terminals share dischtime, so the
    # kind rank fixes their order and breaks any exact tie against the last care unit.
    log = log.sort_values(["case_id", "timestamp", "_kind"]).drop(columns="_kind")
    return log.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mimic", required=True, type=Path, help="MIMIC-IV 3.1 root (contains hosp/)")
    ap.add_argument("--out", required=True, type=Path, help="output event-log CSV path")
    ap.add_argument("--max-cases", type=int, default=None, help="cap #admissions (smoke test)")
    args = ap.parse_args()

    log = build_log(args.mimic, max_cases=args.max_cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(args.out, index=False)

    n_cases = log["case_id"].nunique()
    n_expired = int((log["activity"] == "EXPIRED").sum())
    n_readmit = int((log["activity"] == "READMIT_30D").sum())
    n_noreadmit = int((log["activity"] == "NO_READMIT_30D").sum())
    n_survivors = max(n_readmit + n_noreadmit, 1)
    terminals = ["EXPIRED", "DISCHARGED", "READMIT_30D", "NO_READMIT_30D"]
    print(
        f"wrote {len(log):,} events / {n_cases:,} admissions -> {args.out}\n"
        f"  care units: {log.loc[~log['activity'].isin(terminals), 'activity'].nunique()} distinct\n"
        f"  mortality:  {n_expired:,} expired ({n_expired / max(n_cases, 1):.1%}), "
        f"{n_cases - n_expired:,} survived\n"
        f"  readmit30:  {n_readmit:,} readmitted ({n_readmit / n_survivors:.1%} of survivors), "
        f"{n_noreadmit:,} not"
    )


if __name__ == "__main__":
    main()
