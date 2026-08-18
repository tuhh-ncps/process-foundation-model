"""Aggregate STRUCTURAL fingerprint per log — quantifies loopiness/rework to explain cross-domain
transfer. Confirms whether logs cluster by process STRUCTURE (not business domain), and ranks each
log's structural distance to Sepsis (the transfer target). See docs/dataset_portfolio.md.

Metrics per log (all control-flow only):
  events_per_case, variant_ratio, self_loop_rate (immediate repeats), rework_rate (revisits an
  earlier activity), repetition_factor (events/case ÷ distinct acts/case), back_edge_frac (DFG edges
  against mean-position order = cyclicity), succ_entropy (branching).
Distance-to-Sepsis = Euclidean distance in z-scored loop-feature space.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pm_foundation.data.readers.csv_reader import CsvLogReader
from pm_foundation.data.readers.xes_reader import XesLogReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
VR = ROOT / "data/very-raw"
CAP = 8000  # cap traces per log for a stable structural sample (big logs)

# (label, path, format, group)
LOGS = [
    ("mimic", RAW / "mimic_transfers.csv", "csv", "worked"),
    ("BPI20ID", RAW / "BPI20ID.xes", "xes", "worked"),
    ("Sepsis", RAW / "SepsisCases_Event_Log.xes", "xes", "target"),
    ("BPI17", RAW / "BPI17.xes", "xes", "loopy?"),
    (
        "BPI13-inc",
        VR / "BPI Challenge 2013, incidents_1_all/BPI_Challenge_2013_incidents.xes",
        "xes",
        "loopy?",
    ),
    ("BPI12", RAW / "BPI12.xes", "xes", "corpus"),
    ("BPI18", RAW / "BPI18.xes", "xes", "corpus"),
    ("BPI19", RAW / "BPI19.xes", "xes", "corpus"),
    ("RoadTraffic", RAW / "RoadTraffic.xes", "xes", "corpus"),
    ("HospBilling", RAW / "HospitalBilling.xes", "xes", "corpus"),
    ("BPI11", RAW / "BPI11.xes", "xes", "corpus"),
]

FEATURES = [
    "events_per_case",
    "variant_ratio",
    "self_loop_rate",
    "rework_rate",
    "repetition_factor",
    "back_edge_frac",
    "succ_entropy",
]


def read_traces(path: Path, fmt: str):
    if fmt == "csv":
        traces = CsvLogReader().read(path).traces
        return traces[:CAP]
    return XesLogReader(max_traces=CAP).read(path).traces


def profile(traces) -> dict:
    n_cases = len(traces)
    n_events = 0
    self_loops = trans = 0
    rework = 0
    uniq_per_case = []
    variants = set()
    edge_ct: Counter = Counter()
    succ: dict = defaultdict(Counter)
    pos_sum: dict = defaultdict(float)
    pos_n: Counter = Counter()

    for t in traces:
        seq = [e.activity for e in t.events]
        if not seq:
            continue
        n_events += len(seq)
        variants.add(tuple(seq))
        uniq_per_case.append(len(set(seq)))
        seen: set = set()
        L = len(seq)
        for i, a in enumerate(seq):
            if a in seen:
                rework += 1
            else:
                seen.add(a)
            pos_sum[a] += i / max(L - 1, 1)
            pos_n[a] += 1
            if i + 1 < L:
                b = seq[i + 1]
                trans += 1
                if a == b:
                    self_loops += 1
                edge_ct[(a, b)] += 1
                succ[a][b] += 1

    mean_pos = {a: pos_sum[a] / pos_n[a] for a in pos_n}
    back = tot = 0
    for (a, b), c in edge_ct.items():
        if a == b:
            continue
        tot += c
        if mean_pos.get(b, 0) < mean_pos.get(a, 0):
            back += c
    # frequency-weighted successor entropy
    act_freq = Counter()
    for a, d in succ.items():
        act_freq[a] = sum(d.values())
    ent_num = ent_den = 0.0
    for a, d in succ.items():
        tot_a = sum(d.values())
        H = -sum((c / tot_a) * math.log2(c / tot_a) for c in d.values() if c)
        ent_num += H * act_freq[a]
        ent_den += act_freq[a]

    epc = n_events / n_cases if n_cases else 0
    upc = sum(uniq_per_case) / len(uniq_per_case) if uniq_per_case else 1
    return {
        "n_cases": n_cases,
        "events_per_case": round(epc, 2),
        "variant_ratio": round(len(variants) / n_cases, 3) if n_cases else 0,
        "self_loop_rate": round(self_loops / trans, 3) if trans else 0,
        "rework_rate": round(rework / n_events, 3) if n_events else 0,
        "repetition_factor": round(epc / upc, 2) if upc else 0,
        "back_edge_frac": round(back / tot, 3) if tot else 0,
        "succ_entropy": round(ent_num / ent_den, 3) if ent_den else 0,
    }


def main() -> None:
    rows = []
    for label, path, fmt, group in LOGS:
        if not path.exists():
            print(f"[skip] {label}: {path} not found", file=sys.stderr)
            continue
        try:
            m = profile(read_traces(path, fmt))
            m.update(label=label, group=group)
            rows.append(m)
            print(
                f"[ok] {label:12} cases={m['n_cases']:>6} rework={m['rework_rate']:.3f} "
                f"selfloop={m['self_loop_rate']:.3f} repeat={m['repetition_factor']:.2f} "
                f"back={m['back_edge_frac']:.3f}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[err] {label}: {e!r}", file=sys.stderr)

    # z-score features across logs, distance to Sepsis
    means = {f: sum(r[f] for r in rows) / len(rows) for f in FEATURES}
    stds = {
        f: (sum((r[f] - means[f]) ** 2 for r in rows) / len(rows)) ** 0.5 or 1.0 for f in FEATURES
    }

    def z(r):
        return [(r[f] - means[f]) / stds[f] for f in FEATURES]

    zs = {r["label"]: z(r) for r in rows}
    for r in rows:
        r["dist_to_sepsis"] = round(math.dist(zs[r["label"]], zs["Sepsis"]), 2)

    order = sorted(rows, key=lambda r: r["dist_to_sepsis"])
    print("\n=== STRUCTURAL FINGERPRINT (sorted by distance to Sepsis) ===")
    hdr = (
        f"{'log':12} {'group':8} {'dist→Sepsis':>11} | {'ev/case':>7} {'var_ratio':>9} "
        f"{'selfloop':>8} {'rework':>7} {'repeat×':>7} {'backedge':>8} {'branch_H':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in order:
        print(
            f"{r['label']:12} {r['group']:8} {r['dist_to_sepsis']:>11} | "
            f"{r['events_per_case']:>7} {r['variant_ratio']:>9} {r['self_loop_rate']:>8} "
            f"{r['rework_rate']:>7} {r['repetition_factor']:>7} {r['back_edge_frac']:>8} "
            f"{r['succ_entropy']:>8}"
        )

    # scatter: rework_rate (x) vs repetition_factor (y), sized by variant_ratio
    colors = {"corpus": "#1f77b4", "worked": "#2ca02c", "target": "#d62728", "loopy?": "#ff7f0e"}
    plt.figure(figsize=(9, 6.5))
    for r in rows:
        plt.scatter(
            r["rework_rate"],
            r["repetition_factor"],
            s=80 + 300 * r["variant_ratio"],
            c=colors[r["group"]],
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
        )
        plt.annotate(
            r["label"],
            (r["rework_rate"], r["repetition_factor"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    for g, c in colors.items():
        plt.scatter([], [], c=c, label=g, s=80, edgecolors="k", linewidths=0.5)
    plt.xlabel("rework_rate  (fraction of events revisiting an earlier activity) →  loopier")
    plt.ylabel("repetition_factor  (events per case ÷ distinct acts) →  loopier")
    plt.title(
        "Process structure, not business domain: Sepsis is the high-rework outlier\n"
        "(marker size ∝ variant ratio)"
    )
    plt.legend(title="role in pipeline")
    plt.grid(alpha=0.3)
    out = ROOT / "docs/structural_fingerprint.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
