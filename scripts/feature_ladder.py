#!/usr/bin/env python3
"""Phase A of protocols/feature_ladder.md: freeze the fingerprint-descriptor order from pretraining data only.

A1  fingerprints of the six pretraining logs, each read exactly as its Phase-1b data config specifies
    (70% chronological training partition, cases with >= 2 events, control-flow stripped, no trace caps),
    computed per log with that log's own vocabulary and directly-follows graph;
A2  per-log correlation matrices R_l (standardised within log; constant descriptor -> 0 off-diagonal, 1 on
    the diagonal);
A3  R_bar = mean over the six logs;
A4  greedy order maximising J(S) = tr(R_bar[:,S] pinv(R_bar[S,S]) R_bar[S,:]), Moore-Penrose pseudo-inverse
    at rcond = 1e-10, exact ties resolved by the Table-1 letter order A..O;
A4a invariants of R_bar and J(S) (any violation -> exit 1, nothing is frozen);
A4b the same ordering at rcond in {1e-8, 1e-10, 1e-12} (reported only);
A5  leave-one-log-out orderings (reported only).

No held-out evaluation log is ever read.

Usage (inside the project container, from the repository root):
    python scripts/feature_ladder.py --data-dir /workspace/data/raw \
        --out outputs/feature_ladder/feature_ladder.json \
        --fingerprints outputs/feature_ladder/feature_ladder_fingerprints.npz
    python scripts/feature_ladder.py --from-fingerprints results/feature_ladder_fingerprints.npz --out ...
    python scripts/feature_ladder.py --selftest          # synthetic checks, no data needed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Code order of the 15 fingerprint columns (pm_foundation.data.roles, pruned `feats`).
NAMES = ["pagerank", "betweenness", "self_loop_p", "in_gap_median", "in_gap_std", "out_gap_median",
         "out_gap_std", "p_start", "p_terminal", "support", "pred_entropy", "succ_entropy",
         "mean_pos", "std_pos", "rework_p"]
N = len(NAMES)
# Table-1 letters (family order). Used ONLY to resolve exact ties.
TABLE1 = ["pagerank", "betweenness", "self_loop_p", "p_start", "p_terminal", "support", "rework_p",
          "pred_entropy", "succ_entropy", "in_gap_median", "in_gap_std", "out_gap_median", "out_gap_std",
          "mean_pos", "std_pos"]
LETTER = {name: chr(ord("A") + i) for i, name in enumerate(TABLE1)}
TIE_ORDER = [NAMES.index(n) for n in TABLE1]           # code indices in A..O order
P_START, P_TERMINAL = NAMES.index("p_start"), NAMES.index("p_terminal")

# Six pretraining logs -> their Phase-1b data configs (configs/data/<name>.yaml).
PRETRAIN = [("BPI12", "bpi12_solo"), ("BPI19", "bpi19"), ("BPI18", "bpi18"),
            ("RoadTraffic", "road_traffic"), ("BPI11", "bpi11"), ("HospitalBilling", "hospital_billing")]
HELD_OUT_FILES = {"BPI17.xes", "BPI20ID.xes", "BPI13.xes", "helpdesk.csv", "mimic_transfers.csv"}

RCOND = 1e-10
RCOND_GRID = [1e-8, 1e-10, 1e-12]
EPS = 1e-8


# ----------------------------------------------------------------------------- A1
def load_fingerprints(data_dir: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    import yaml

    from pm_foundation.data.roles import fit_role_graph
    from pm_foundation.data.vocabulary import Vocabulary
    from pm_foundation.training.ar_pretrain import _read_log, _train_traces

    X, meta = {}, {}
    for log, cfg_name in PRETRAIN:
        cfg = yaml.safe_load(open(ROOT / "configs" / "data" / f"{cfg_name}.yaml"))
        (spec,) = cfg["train_logs"]                        # exactly one training log per config
        assert "max_traces" not in spec, f"{cfg_name}: Phase-1b config unexpectedly caps traces"
        path = spec["path"].replace("${paths.data_dir}", data_dir)
        assert Path(path).name not in HELD_OUT_FILES, f"refusing to read held-out log {path}"
        strip, min_len, split = bool(cfg["strip_to_control_flow"]), int(cfg["min_trace_len"]), tuple(cfg["split"])
        t0 = time.time()
        traces = _train_traces(_read_log({"path": path}, strip=strip), min_trace_len=min_len, split=split)
        acts = sorted({e.activity for t in traces for e in t.events})
        g = fit_role_graph(traces, Vocabulary.build(acts))
        real = g["real_mask"].numpy()
        X[log] = g["feats"].numpy()[real].astype(np.float64)
        meta[log] = {"config": f"configs/data/{cfg_name}.yaml", "path": path, "strip_to_control_flow": strip,
                     "min_trace_len": min_len, "split": list(split), "partition": "train", "trace_cap": None,
                     "n_train_traces": len(traces), "n_activities": int(real.sum()),
                     "seconds": round(time.time() - t0, 1)}
        print(f"[A1] {log}: {len(traces)} train traces, {int(real.sum())} activities ({meta[log]['seconds']} s)",
              flush=True)
    return X, meta


# ----------------------------------------------------------------------------- A2/A3
def log_correlation(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    const = sd < 1e-12
    Z = np.zeros_like(X)
    Z[:, ~const] = (X[:, ~const] - mu[~const]) / sd[~const]
    R = Z.T @ Z / X.shape[0]
    R[const, :] = 0.0
    R[:, const] = 0.0
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R, [NAMES[i] for i in np.flatnonzero(const)]


def pinv_sym(A: np.ndarray, rcond: float) -> np.ndarray:
    """Moore-Penrose pseudo-inverse of a symmetric matrix: singular values = |eigenvalues|, cut at rcond*max."""
    w, V = np.linalg.eigh(A)
    cut = rcond * np.max(np.abs(w))
    inv = np.zeros_like(w)
    keep = np.abs(w) > cut
    inv[keep] = 1.0 / w[keep]
    return (V * inv) @ V.T


def recovered(R: np.ndarray, S: list[int], rcond: float) -> np.ndarray:
    """Per-descriptor recovered standardised variance diag(R[:,S] pinv(R[S,S]) R[S,:]); J(S) is its sum."""
    if not S:
        return np.zeros(R.shape[0])
    A = R[:, S]
    return np.einsum("ij,ij->i", A @ pinv_sym(R[np.ix_(S, S)], rcond), A)


# ----------------------------------------------------------------------------- A4
class Invariants:
    def __init__(self, gate: bool):
        self.gate, self.violations, self.worst = gate, [], {}

    def _note(self, key: str, value: float) -> None:
        self.worst[key] = max(self.worst.get(key, 0.0), float(value))

    def fail(self, msg: str) -> None:
        self.violations.append(msg)

    def matrix(self, R: np.ndarray) -> None:
        asym, diag, lam = np.abs(R - R.T).max(), np.abs(np.diag(R) - 1).max(), np.linalg.eigvalsh(R).min()
        self._note("max_asymmetry", asym)
        self._note("max_diag_error", diag)
        self.worst["min_eigenvalue"] = float(lam)
        if asym > 1e-12: self.fail(f"R_bar not symmetric: {asym:.3g}")
        if diag > 1e-12: self.fail(f"R_bar diagonal not 1: {diag:.3g}")
        if lam < -1e-10: self.fail(f"R_bar not PSD: min eigenvalue {lam:.3g}")

    def subset(self, S: list[int], rec: np.ndarray) -> None:
        J = rec.sum()
        lo, hi = len(S) - J, J - N
        self._note("J_below_size", lo)
        self._note("J_above_15", hi)
        if lo > EPS or hi > EPS: self.fail(f"J({S})={J:.12g} outside [{len(S)}, {N}]")
        below, above = -rec.min(), rec.max() - 1
        self._note("rec_below_0", below)
        self._note("rec_above_1", above)
        if below > EPS or above > EPS: self.fail(f"recovered variance outside [0,1] for S={S}")
        if S:
            sel = np.abs(rec[S] - 1).max()
            self._note("selected_not_1", sel)
            if sel > EPS: self.fail(f"selected descriptor not fully recovered for S={S}: {sel:.3g}")


def greedy(R: np.ndarray, rcond: float, inv: Invariants | None = None, rng=None) -> dict:
    S: list[int] = []
    steps, prev = [], 0.0
    if inv is not None:
        rec0 = recovered(R, [], rcond)
        if abs(rec0.sum()) > EPS: inv.fail("J(empty) != 0")
    for k in range(1, N + 1):
        cands = [f for f in TIE_ORDER if f not in S]          # candidates in A..O order
        scored = []
        for f in cands:
            rec = recovered(R, S + [f], rcond)
            if inv is not None:
                inv.subset(S + [f], rec)
            scored.append((rec.sum(), f, rec))
        best = max(J for J, _, _ in scored)
        tied = [f for J, f, _ in scored if J == best]           # exact ties
        choice = tied[0]                                        # first in A..O order
        runner = max((J for J, f, _ in scored if f != choice), default=None)
        S.append(choice)
        J = best
        dJ = J - prev
        if inv is not None:
            inv._note("negative_dJ", -dJ)
            if dJ < -EPS: inv.fail(f"J decreased at step {k}: {dJ:.3g}")
            perm = list(rng.permutation(S))
            pdiff = abs(recovered(R, perm, rcond).sum() - J)
            inv._note("permutation_diff", pdiff)
            if pdiff > EPS: inv.fail(f"J not order-invariant at step {k}: {pdiff:.3g}")
        steps.append({"k": k, "code_index": choice, "name": NAMES[choice], "letter": LETTER[NAMES[choice]],
                      "J": J, "J_over_15": J / N, "delta_J": dJ,
                      "margin_to_runner_up": (J - runner) if runner is not None else None,
                      "n_exact_ties": len(tied) - 1})
        prev = J
    if inv is not None:
        full = recovered(R, list(range(N)), rcond).sum()
        inv._note("J_full_error", abs(full - N))
        if abs(full - N) > EPS: inv.fail(f"J(all 15) = {full:.12g} != 15")
    return {"order": S, "steps": steps}


def kendall_tau(a: list[int], b: list[int]) -> float:
    pa, pb = {f: i for i, f in enumerate(a)}, {f: i for i, f in enumerate(b)}
    items, c, d = list(a), 0, 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (pa[items[i]] - pa[items[j]]) * (pb[items[i]] - pb[items[j]])
            c, d = c + (s > 0), d + (s < 0)
    return (c - d) / (len(items) * (len(items) - 1) / 2)


def enters(order: list[int]) -> dict:
    ps, pt = order.index(P_START) + 1, order.index(P_TERMINAL) + 1
    return {"p_start": ps, "p_terminal": pt, "first_of_either": min(ps, pt)}


def compare(primary: list[int], other: list[int], R_bar: np.ndarray) -> dict:
    out = {"kendall_tau": kendall_tau(primary, other), "order": other,
           "order_letters": "".join(LETTER[NAMES[f]] for f in other), "enters": enters(other)}
    for k in (3, 5, 10):
        out[f"top{k}_overlap"] = len(set(primary[:k]) & set(other[:k])) / k
        out[f"top{k}_chance"] = k / N
        out[f"top{k}_J_ratio"] = recovered(R_bar, other[:k], RCOND).sum() / recovered(R_bar, primary[:k], RCOND).sum()
    return out


def git_blob_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ----------------------------------------------------------------------------- driver
def analyse(X: dict[str, np.ndarray], meta: dict) -> tuple[dict, list[str]]:
    logs = [log for log, _ in PRETRAIN]
    Rs, consts = {}, {}
    for log in logs:
        Rs[log], consts[log] = log_correlation(X[log])
    R_bar = sum(Rs[log] for log in logs) / len(logs)
    lam = np.linalg.eigvalsh(R_bar)

    inv = Invariants(gate=True)
    inv.matrix(R_bar)
    rng = np.random.default_rng(0)
    primary = greedy(R_bar, RCOND, inv, rng)
    P = primary["order"]

    robustness = {}
    for rc in RCOND_GRID:
        o = greedy(R_bar, rc)["order"]
        diff = next((i + 1 for i, (x, y) in enumerate(zip(P, o)) if x != y), None)
        robustness[f"{rc:g}"] = {"identical_to_primary": o == P, "first_differing_position": diff,
                                 "kendall_tau": kendall_tau(P, o),
                                 "order_letters": "".join(LETTER[NAMES[f]] for f in o),
                                 "J_over_15": [recovered(R_bar, o[:k], rc).sum() / N for k in range(1, N + 1)]}

    lolo = {}
    for out_log in logs:
        R_minus = sum(Rs[log] for log in logs if log != out_log) / (len(logs) - 1)
        lolo[out_log] = compare(P, greedy(R_minus, RCOND)["order"], R_bar)
    taus = {log: v["kendall_tau"] for log, v in lolo.items()}
    worst_log = min(taus, key=taus.get)

    art = {
        "protocol": "protocols/feature_ladder.md",
        "protocol_git_blob": git_blob_hash(ROOT / "protocols" / "feature_ladder.md"),
        "script": "scripts/feature_ladder.py",
        "script_git_blob": git_blob_hash(Path(__file__).resolve()),
        "status": "OK",
        "order": [{"k": s["k"], "name": s["name"], "code_index": s["code_index"], "letter": s["letter"]}
                  for s in primary["steps"]],
        "order_code_indices": P,
        "order_letters": "".join(s["letter"] for s in primary["steps"]),
        "steps": primary["steps"],
        "p_start_p_terminal_enter_at": enters(P),
        "settings": {"pretraining_logs": logs, "per_log_data": meta,
                     "standardisation": "within log, population SD (ddof=0)",
                     "constant_descriptor_rule": "SD < 1e-12 within log -> correlation 0 off-diagonal, 1 on diagonal",
                     "consensus": "R_bar = mean of the six per-log correlation matrices",
                     "objective": "J(S) = tr(R_bar[:,S] pinv(R_bar[S,S]) R_bar[S,:])",
                     "pseudo_inverse": "Moore-Penrose via symmetric eigendecomposition, cutoff rcond*max|eigenvalue|",
                     "rcond": RCOND, "tie_break": "exact ties -> Table-1 letter order A..O",
                     "invariant_tolerance": EPS},
        "R_bar": R_bar.tolist(),
        "R_bar_eigenvalues": lam.tolist(),
        "R_bar_condition_number": float(lam.max() / lam.min()) if lam.min() > 0 else float("inf"),
        "constant_descriptors_per_log": consts,
        "invariants": {"passed": not inv.violations, "violations": inv.violations[:50],
                       "n_violations": len(inv.violations), "worst": inv.worst},
        "pinv_robustness": robustness,
        "leave_one_log_out": {"per_log": lolo, "min_kendall_tau": taus[worst_log], "min_tau_left_out_log": worst_log},
        "table1_letters": LETTER,
        "code_order_names": NAMES,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    return art, inv.violations


def selftest() -> None:
    rng = np.random.default_rng(1)
    X = {}
    for i, (log, _) in enumerate(PRETRAIN):
        n = 20 + 7 * i
        base = rng.normal(size=(n, 4))
        M = np.column_stack([base @ rng.normal(size=4) + 0.3 * rng.normal(size=n) for _ in range(N)])
        M[:, NAMES.index("in_gap_median")] = M[:, NAMES.index("pagerank")]      # exact duplicate: tie
        if i == 0:
            M[:, NAMES.index("self_loop_p")] = 0.5                               # constant within one log
        X[log] = M
    art, viol = analyse(X, {log: {"synthetic": True} for log, _ in PRETRAIN})
    assert not viol, viol
    R = np.eye(N)
    assert abs(recovered(R, [0, 3, 7], RCOND).sum() - 3) < 1e-12              # identity: J(S) = |S|
    A = rng.normal(size=(6, 6)); A = A @ A.T; A[:, 0] = A[0, :] = 0; A[0, 0] = 0
    assert np.allclose(pinv_sym(A, RCOND), np.linalg.pinv(A, rcond=RCOND, hermitian=True))
    order = art["order_code_indices"]
    pr, dup = NAMES.index("pagerank"), NAMES.index("in_gap_median")
    assert order.index(pr) < order.index(dup), "exact tie must be resolved by A..O order (A before J)"
    assert art["invariants"]["passed"] and len(order) == N and sorted(order) == list(range(N))
    print("selftest OK | order:", art["order_letters"], "| worst:", {k: f"{v:.2e}" for k, v in art["invariants"]["worst"].items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "feature_ladder.json"))
    ap.add_argument("--fingerprints", help="write the per-log fingerprint matrices here (.npz)")
    ap.add_argument("--from-fingerprints", help="skip A1 and load fingerprint matrices from this .npz")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.from_fingerprints:
        z = np.load(a.from_fingerprints, allow_pickle=False)
        X = {log: z[f"{log}__X"] for log, _ in PRETRAIN}
        meta = json.loads(str(z["meta"]))
    else:
        X, meta = load_fingerprints(a.data_dir)
        if a.fingerprints:
            Path(a.fingerprints).parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.fingerprints, meta=json.dumps(meta), **{f"{log}__X": X[log] for log in X})
    art, viol = analyse(X, meta)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if viol:
        art["status"] = "FAILED_INVARIANTS"
        out = out.with_suffix(".FAILED.json")
    out.write_text(json.dumps(art, indent=2))
    print(f"[A4] order {art['order_letters']} | J/15 per step "
          + " ".join(f"{s['J_over_15']:.3f}" for s in art["steps"]))
    print(f"[A4a] invariants {'PASSED' if not viol else 'FAILED (%d)' % len(viol)} | cond(R_bar) {art['R_bar_condition_number']:.3g}")
    print("[A4b] rcond:", {k: v["identical_to_primary"] for k, v in art["pinv_robustness"].items()})
    print(f"[A5] min Kendall tau {art['leave_one_log_out']['min_kendall_tau']:.3f} "
          f"(leaving out {art['leave_one_log_out']['min_tau_left_out_log']})")
    print("wrote", out)
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
