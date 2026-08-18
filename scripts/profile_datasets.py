"""Profile every event log under data/very-raw/ into a stats JSON for the dataset portfolio.

Memory-safe streaming (ET.iterparse + root.clear) so the 1.8 GB BPI'18 / 704 MB BPI'19
logs scan with flat memory. Per file we compute: case/event counts, case-length distribution,
activity vocabulary (+ top activities by event frequency), event/case attribute schema,
lifecycle transitions, timespan, and #variants (distinct control-flow sequences).

Single-log datasets are scanned in full. Multi-log dirs (Activities/NASA/Apache) scan every
file. Process-Discovery-Contest collections are SAMPLED (first N log files) and the collection's
total file count + bytes are reported separately, since they hold hundreds of homogeneous
synthetic logs.

Usage:
  uv run python scripts/profile_datasets.py [--only SUBSTR] [--out PATH] [--sample-pdc N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

VR = Path(__file__).resolve().parents[1] / "data/very-raw"
_ATTR_TAGS = frozenset({"string", "date", "int", "float", "boolean", "id"})


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ts(raw: str) -> datetime | None:
    try:
        v = raw.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.fromisoformat(v)
    except Exception:
        return None


def _pct(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def scan_file(path: Path, max_cases: int | None = None) -> dict:
    """Stream one .xes and return summary stats (naive-tz timestamps for global min/max)."""
    n_cases = 0
    n_events = 0
    case_lens: list[int] = []
    act_counter: Counter[str] = Counter()
    ev_keys: set[str] = set()
    case_keys: set[str] = set()
    lifecycles: set[str] = set()
    variants: set[int] = set()
    durations_s: list[float] = []
    t_min: datetime | None = None
    t_max: datetime | None = None
    has_ts = False

    ctx = ET.iterparse(str(path), events=("start", "end"))
    _, root = next(ctx)
    for event, elem in ctx:
        if event != "end" or _ln(elem.tag) != "trace":
            continue
        seq: list[str] = []
        first_ts_raw: str | None = None
        last_ts_raw: str | None = None
        ev_count = 0
        for child in elem:
            ctag = _ln(child.tag)
            if ctag == "event":
                ev_count += 1
                act = None
                for a in child:
                    atag = _ln(a.tag)
                    if atag not in _ATTR_TAGS:
                        continue
                    key = a.get("key")
                    if key is None:
                        continue
                    if key == "concept:name":
                        act = a.get("value")
                    elif key == "time:timestamp":
                        raw = a.get("value")
                        if raw is not None:
                            if first_ts_raw is None:
                                first_ts_raw = raw
                            last_ts_raw = raw
                    else:
                        ev_keys.add(key)
                        if key == "lifecycle:transition":
                            lv = a.get("value")
                            if lv is not None:
                                lifecycles.add(lv)
                if act is not None:
                    act_counter[act] += 1
                    seq.append(act)
            elif ctag in _ATTR_TAGS:
                key = child.get("key")
                if key is not None and key != "concept:name":
                    case_keys.add(key)

        if ev_count > 0:
            n_cases += 1
            n_events += ev_count
            case_lens.append(ev_count)
            variants.add(hash(tuple(seq)))
            if first_ts_raw is not None:
                has_ts = True
                t0 = _ts(first_ts_raw)
                t1 = _ts(last_ts_raw) if last_ts_raw else t0
                if t0 is not None and t1 is not None:
                    d0 = t0.replace(tzinfo=None)
                    d1 = t1.replace(tzinfo=None)
                    durations_s.append(abs((d1 - d0).total_seconds()))
                    lo, hi = (d0, d1) if d0 <= d1 else (d1, d0)
                    t_min = lo if t_min is None or lo < t_min else t_min
                    t_max = hi if t_max is None or hi > t_max else t_max

        elem.clear()
        root.clear()  # drop processed trace from root -> flat memory
        if max_cases is not None and n_cases >= max_cases:
            break

    case_lens.sort()
    durations_s.sort()
    span_days = None
    if t_min is not None and t_max is not None:
        span_days = round((t_max - t_min).total_seconds() / 86400.0, 1)
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_cases": n_cases,
        "n_events": n_events,
        "n_activities": len(act_counter),
        "n_variants": len(variants),
        "variant_ratio": round(len(variants) / n_cases, 3) if n_cases else None,
        "case_len": {
            "min": case_lens[0] if case_lens else 0,
            "median": round(statistics.median(case_lens), 1) if case_lens else 0,
            "mean": round(statistics.fmean(case_lens), 1) if case_lens else 0,
            "p90": _pct(case_lens, 0.90),
            "max": case_lens[-1] if case_lens else 0,
        },
        "avg_case_duration_days": round(statistics.fmean(durations_s) / 86400.0, 2)
        if durations_s
        else None,
        "median_case_duration_days": round(statistics.median(durations_s) / 86400.0, 2)
        if durations_s
        else None,
        "has_timestamps": has_ts,
        "time_start": t_min.date().isoformat() if t_min else None,
        "time_end": t_max.date().isoformat() if t_max else None,
        "span_days": span_days,
        "top_activities": act_counter.most_common(20),
        "event_attr_keys": sorted(ev_keys),
        "case_attr_keys": sorted(case_keys),
        "lifecycles": sorted(lifecycles),
    }


def find_logs(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*.xes") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on parent dir name")
    ap.add_argument("--out", default=str(VR / "_portfolio_stats.json"))
    ap.add_argument(
        "--sample-pdc", type=int, default=3, help="log files to scan per PDC collection"
    )
    args = ap.parse_args()

    parents = sorted(d for d in VR.iterdir() if d.is_dir())
    if args.only:
        parents = [d for d in parents if args.only.lower() in d.name.lower()]

    result: dict[str, dict] = {}
    for d in parents:
        logs = find_logs(d)
        if not logs:
            continue
        is_pdc = "process discovery contest" in d.name.lower()
        total_bytes = sum(p.stat().st_size for p in logs)
        to_scan = logs
        sampled = False
        if is_pdc:
            # prefer the Training Logs subdir; sample a few files
            train = [p for p in logs if "training logs" in str(p).lower()] or logs
            to_scan = train[: args.sample_pdc]
            sampled = True

        print(f"[{d.name}] scanning {len(to_scan)}/{len(logs)} logs ...", flush=True)
        files = []
        for p in to_scan:
            t0 = time.time()
            try:
                st = scan_file(p)
            except Exception as e:
                st = {"file": p.name, "error": repr(e)}
            st["scan_s"] = round(time.time() - t0, 1)
            files.append(st)
            print(
                f"    {p.name}: {st.get('n_cases', '?')} cases, "
                f"{st.get('n_events', '?')} events, {st.get('n_activities', '?')} acts "
                f"({st['scan_s']}s)",
                flush=True,
            )

        result[d.name] = {
            "n_log_files": len(logs),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1e6, 1),
            "sampled": sampled,
            "is_collection": is_pdc or len(logs) > 1,
            "files": files,
        }
        # write incrementally so partial progress survives a long run
        Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"\nWROTE {args.out}  ({len(result)} datasets)")


if __name__ == "__main__":
    main()
