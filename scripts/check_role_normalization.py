"""Confirm the role-fingerprint train/eval mismatch.

Training fits ONE role graph over the concatenated multi-dataset corpus (union vocab);
eval fits the graph on the target log ALONE. Fingerprints are rank-normalized per column,
so a feature value means "rank among whatever population it was fit against".

NOTE: marginals are uniform by construction (ranks), so comparing marginals proves nothing.
The real test: does the SAME activity get a DIFFERENT fingerprint solo vs in-union?
That displacement is exactly the input shift the GIN suffers at eval time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from pm_foundation.data.preprocessing import fit_feature_spec
from pm_foundation.data.readers.csv_reader import CsvLogReader
from pm_foundation.data.readers.xes_reader import XesLogReader
from pm_foundation.data.roles import fit_role_graph
from pm_foundation.data.schema import EventLog

RAW = Path(__file__).resolve().parents[1] / "data/raw"
FEATNAMES = [
    "in_deg",
    "out_deg",
    "pagerank",
    "betweenness",
    "in_cycle",
    "self_loop_p",
    "in_gap_med",
    "in_gap_std",
    "in_gap_p90",
    "out_gap_med",
    "out_gap_std",
    "out_gap_p90",
    "p_start",
    "p_terminal",
    "support",
    "pred_H",
    "succ_H",
    "mean_pos",
    "std_pos",
    "rework_p",
]

# corpus members (as trained) + Sepsis (the eval target) — capped for speed
SETS = [
    ("BPI12", RAW / "BPI12.xes", "xes", 3000),
    ("BPI18", RAW / "BPI18.xes", "xes", 2000),
    ("BPI19", RAW / "BPI19.xes", "xes", 3000),
    ("RoadTraffic", RAW / "RoadTraffic.xes", "xes", 3000),
    ("HospBilling", RAW / "HospitalBilling.xes", "xes", 3000),
    ("BPI11", RAW / "BPI11.xes", "xes", 1143),
    ("Sepsis", RAW / "SepsisCases_Event_Log.xes", "xes", 1050),
]


def load(path: Path, fmt: str, cap: int):
    if fmt == "csv":
        return CsvLogReader().read(path).traces[:cap]
    return XesLogReader(max_traces=cap).read(path).traces


def fingerprints(traces) -> dict[str, torch.Tensor]:
    """name -> 20-dim rank-normalized fingerprint, for the graph fit on exactly these traces."""
    acts = sorted({e.activity for t in traces for e in t.events})
    log = EventLog(traces=list(traces), activity_vocab=acts)
    spec = fit_feature_spec(log, max_seq_len=64)
    g = fit_role_graph(list(traces), spec.activity_vocab)
    names = spec.activity_vocab.to_list()
    feats, mask = g["feats"], g["real_mask"]
    return {names[i]: feats[i].clone() for i in range(len(names)) if bool(mask[i])}


def main() -> None:
    data = {}
    for name, path, fmt, cap in SETS:
        if not path.exists():
            print(f"[skip] {name}", file=sys.stderr)
            continue
        data[name] = load(path, fmt, cap)
        print(f"[load] {name}: {len(data[name])} traces", file=sys.stderr)

    # SOLO fingerprints (what EVAL computes for a target log)
    solo = {n: fingerprints(tr) for n, tr in data.items()}
    for n in solo:
        print(f"[solo] {n}: {len(solo[n])} activities", file=sys.stderr)

    # UNION fingerprints (what TRAINING computes over the concatenated corpus)
    corpus = [n for n in data if n != "Sepsis"]
    union_traces = [t for n in corpus for t in data[n]]
    union = fingerprints(union_traces)
    print(f"[union] corpus union: {len(union)} activities", file=sys.stderr)

    # union INCLUDING Sepsis -> lets us measure the shift for the actual eval target too
    union_s = fingerprints(union_traces + list(data["Sepsis"]))
    print(f"[union+Sepsis] {len(union_s)} activities", file=sys.stderr)

    print("\n=== SAME ACTIVITY, SOLO-fit vs UNION-fit fingerprint displacement ===")
    print("(features are rank-normalized to [0,1]; 0.33 = a third of the whole range)\n")
    hdr = f"{'dataset':12} {'#acts':>6} {'mean|Δ|':>8} {'max|Δ|':>7} {'%acts Δ>0.25':>13} {'worst feature':>16}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for n in corpus:
        u = union
        common = [a for a in solo[n] if a in u]
        if not common:
            continue
        D = torch.stack([(u[a] - solo[n][a]).abs() for a in common])  # (A,20)
        per_feat = D.mean(0)
        frac_big = float((D.mean(1) > 0.25).float().mean())
        rows.append(
            (
                n,
                len(common),
                float(D.mean()),
                float(D.max()),
                frac_big,
                FEATNAMES[int(per_feat.argmax())],
                per_feat,
            )
        )
        print(
            f"{n:12} {len(common):>6} {float(D.mean()):>8.3f} {float(D.max()):>7.3f} "
            f"{frac_big * 100:>12.1f}% {FEATNAMES[int(per_feat.argmax())]:>16}"
        )

    # the eval target itself
    commonS = [a for a in solo["Sepsis"] if a in union_s]
    DS = torch.stack([(union_s[a] - solo["Sepsis"][a]).abs() for a in commonS])
    print(
        f"{'Sepsis*':12} {len(commonS):>6} {float(DS.mean()):>8.3f} {float(DS.max()):>7.3f} "
        f"{float((DS.mean(1) > 0.25).float().mean()) * 100:>12.1f}% "
        f"{FEATNAMES[int(DS.mean(0).argmax())]:>16}"
    )
    print(
        "  (*Sepsis measured against a union that includes it — the shift its activities would"
        "\n   undergo between eval-style solo fitting and training-style union fitting)"
    )

    # BPI11 dominance check: how much of the union vocab does it own?
    print("\n=== vocab composition of the training union ===")
    tot = len(union)
    for n in corpus:
        own = len([a for a in solo[n] if a in union])
        print(
            f"  {n:12} {own:>5} activities = {own / tot * 100:>5.1f}% of the {tot}-activity union"
            f"   | {len(data[n]):>5} traces"
        )

    print("\n=== per-feature mean |Δ| (union-fit vs solo-fit), averaged over corpus datasets ===")
    stack = torch.stack([r[6] for r in rows]).mean(0)
    for i in torch.argsort(stack, descending=True):
        print(f"  {FEATNAMES[int(i)]:>14}: {float(stack[int(i)]):.3f}")


if __name__ == "__main__":
    main()
