"""Standalone role-encoder pretraining - trains the vocabulary-free ActivityEncoder BY ITSELF.

No backbone, no transformer, no AR heads: this optimizes the role encoder alone so a backbone can
later just *read* its ``e(a)`` table (``role_encoder.pt``). Objective is **role-aware**:

* contrastive view-consistency (``ActivityEncoder.contrastive_loss`` - anti-collapse / stability),
* cross-log start/end **SupCon** (pull same-role activities together ACROSS logs, on the fused e(a)),
* a linear **role-CE** head (sharpen the linear start/end axis; train-time only, not saved).

Quality is *controlled*: every epoch scores a COMPOSITE on HELD-OUT logs -
``cross-log same-role@5 + start-vs-end transfer + subsampling stability`` - EMA-smoothed, and the
best checkpoint is kept. A raw-fingerprint reference composite is recorded for context (the hand
fingerprint is a strong baseline; see docs). Metrics are numpy-only (no sklearn dependency); the
encoder is tiny so this runs single-process (no DDP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pm_foundation.data.roles import P_START_COL, P_TERMINAL_COL, apply_aggregator, fit_role_graph
from pm_foundation.data.schema import Trace
from pm_foundation.data.vocabulary import Vocabulary
from pm_foundation.experiments import RunRegistry, write_learning_curve
from pm_foundation.models.role_encoder import build_role_module
from pm_foundation.training.ar_pretrain import _read_log, _train_traces

_ENDS = ("start", "end")
_R2I = {"start": 0, "middle": 1, "end": 2}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _roles_of(graph: dict[str, torch.Tensor], thresh: float = 0.85) -> np.ndarray:
    """Per-activity role from the (rank-normalized) start/terminal fingerprints."""
    f = graph["feats"].numpy()
    ps, pt = f[:, P_START_COL], f[:, P_TERMINAL_COL]
    return np.where(ps >= thresh, "start", np.where(pt >= thresh, "end", "middle"))


def _log_traces(spec: dict[str, Any], *, strip: bool, min_len: int, split) -> list[Trace]:
    return _train_traces(_read_log(spec, strip=strip), min_trace_len=min_len, split=split)


def _graph_of(traces: list[Trace], vocab: Vocabulary | None = None) -> dict[str, torch.Tensor]:
    # A subsample must reuse the FULL log's vocab so row indices stay aligned (else a dropped
    # activity shrinks V and the stability mask mismatches).
    if vocab is None:
        vocab = Vocabulary.build(sorted({e.activity for t in traces for e in t.events}))
    return fit_role_graph(traces, vocab)


# --------------------------------------------------------------------------- #
# quality metrics (numpy-only; the HELD-OUT composite that controls selection)
# --------------------------------------------------------------------------- #
def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _crosslog_same_role_at_k(reps, roles, logs, val_logs, k=5) -> float:
    """For each held-out start/end activity, fraction of its k nearest OTHER-log neighbours
    sharing its role. This is 'same role, different log -> near?', measured on held-out logs."""
    s = _unit(_standardize(reps))
    sim = s @ s.T
    anchors = [i for i in range(len(roles)) if roles[i] in _ENDS and logs[i] in val_logs]
    hit = tot = 0
    for i in anchors:
        other = np.array([j for j in range(len(roles)) if logs[j] != logs[i]])
        nn = other[np.argsort(-sim[i, other])[:k]]
        hit += int((roles[nn] == roles[i]).sum())
        tot += len(nn)
    return hit / max(tot, 1)


def _startend_transfer(reps, roles, logs, train_logs, val_logs) -> float:
    """Nearest-class-mean start-vs-end probe: means from TRAIN logs, tested on VAL logs
    (balanced accuracy). A dependency-free stand-in for a linear role probe."""
    xs = _standardize(reps)
    m = np.isin(roles, _ENDS)
    trm, tem = m & np.isin(logs, list(train_logs)), m & np.isin(logs, list(val_logs))
    if len(set(roles[trm])) < 2 or tem.sum() == 0:
        return 0.0
    means = {r: xs[trm][roles[trm] == r].mean(0) for r in _ENDS}
    recalls = []
    for r in _ENDS:
        idx = tem & (roles == r)
        if idx.sum() == 0:
            continue
        d_start = np.linalg.norm(xs[idx] - means["start"], axis=1)
        d_end = np.linalg.norm(xs[idx] - means["end"], axis=1)
        pred = np.where(d_start <= d_end, "start", "end")
        recalls.append(float((pred == r).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _cos_stability(a: np.ndarray, b: np.ndarray) -> float:
    return float((_unit(a) * _unit(b)).sum(1).mean()) if len(a) else 1.0


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _supcon_startend(z: torch.Tensor, roles: list[str], tau: float) -> torch.Tensor:
    """Supervised contrastive on start/end anchors: pull same-role together, push all else apart."""
    zn = F.normalize(z, dim=1)
    sim = (zn @ zn.T) / tau
    n = z.shape[0]
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool), -1e9)
    exp = torch.exp(sim - sim.max(1, keepdim=True).values.detach())
    denom = exp.sum(1)
    terms = []
    for i in range(n):
        if roles[i] not in _ENDS:
            continue
        pos = [j for j in range(n) if j != i and roles[j] == roles[i]]
        if pos:
            terms.append(-(torch.log(exp[i, pos] / denom[i] + 1e-12)).mean())
    return torch.stack(terms).mean() if terms else z.new_zeros(())


def train_role_encoder(config: dict[str, Any]) -> Path:
    """Train the role encoder standalone; save the best ``role_encoder.pt`` + learning curve.

    Returns the run directory. Selection is by an EMA-smoothed HELD-OUT composite quality metric.
    """
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    strip = bool(config.get("strip_to_control_flow", True))
    min_len = int(config.get("min_trace_len", 2))
    split = tuple(config.get("split", (0.70, 0.15, 0.15)))
    mcfg = dict(config.get("model", {}))
    ocfg = dict(config.get("objective", {}))
    pcfg = dict(config.get("optim", {}))
    w_con = float(ocfg.get("w_contrastive", 0.5))
    w_sup = float(ocfg.get("w_supcon", 0.5))
    w_ce = float(ocfg.get("w_ce", 1.0))
    tau_sup = float(ocfg.get("tau_supcon", 0.2))
    epochs = int(pcfg.get("epochs", 120))
    ema_a = float(pcfg.get("ema", 0.35))

    train_specs = list(config["train_logs"])
    val_specs = list(config["val_logs"])
    train_names = [s.get("name") or Path(s["path"]).stem for s in train_specs]
    val_names = [s.get("name") or Path(s["path"]).stem for s in val_specs]

    # ---- build role graphs (train-split only; leak-free). val logs also get a 50% subsample ----
    # Aggregator (mean|sum): see roles.apply_aggregator / DirectedGinLayer. Default mean; 'sum' is
    # true-GIN. Must match the consumer's aggregator - here trainer and eval are the same process.
    aggregator = str(mcfg.get("aggregator", "mean"))

    graphs: dict[str, dict[str, Any]] = {}
    for spec, nm in zip(train_specs + val_specs, train_names + val_names, strict=True):
        traces = _log_traces(spec, strip=strip, min_len=min_len, split=split)
        vocab = Vocabulary.build(sorted({e.activity for t in traces for e in t.events}))
        g = apply_aggregator(_graph_of(traces, vocab), aggregator)
        entry = {"full": g, "roles": _roles_of(g), "sub": None}
        if nm in val_names and len(traces) >= 4:
            sub = [traces[i] for i in sorted(rng.choice(len(traces), len(traces) // 2, replace=False))]
            entry["sub"] = apply_aggregator(_graph_of(sub, vocab), aggregator)  # SAME vocab -> aligned
        graphs[nm] = entry

    registry = RunRegistry(config.get("output_dir", "outputs"))
    data_summary = {
        "train_logs": [{"name": n, "n_activities": int(graphs[n]["full"]["real_mask"].sum())}
                       for n in train_names],
        "val_logs": [{"name": n, "n_activities": int(graphs[n]["full"]["real_mask"].sum())}
                     for n in val_names],
    }
    ctx = registry.start("role_encoder", config, name=config.get("name"), data=data_summary)
    run_dir = ctx.dir

    # ---- encoder (fused ActivityEncoder or a raw|mlp|gin RoleEmbedder) + train-time role head ----
    enc = build_role_module(mcfg, 1)
    role_head = torch.nn.Linear(enc.role_dim, 3)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(role_head.parameters()), lr=float(pcfg.get("lr", 1e-3))
    )
    # inverse-frequency class weights over the TRAIN pool (start/end are rare vs middle)
    cnt = np.zeros(3)
    for nm in train_names:
        for r in graphs[nm]["roles"][graphs[nm]["full"]["real_mask"].numpy()]:
            cnt[_R2I[r]] += 1
    classw = torch.tensor(cnt.sum() / (3 * np.clip(cnt, 1, None)), dtype=torch.float32)

    def out(augment: bool) -> torch.Tensor:  # the e(a) table the backbone will read
        return enc(augment=augment)

    def encode_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reps, roles, logs = [], [], []
        for nm in train_names + val_names:
            enc.set_graph(graphs[nm]["full"])
            rm = graphs[nm]["full"]["real_mask"]
            reps.append(out(False)[rm].detach().numpy())
            roles += list(graphs[nm]["roles"][rm.numpy()])
            logs += [nm] * int(rm.sum())
        return np.concatenate(reps), np.array(roles), np.array(logs)

    def stability(nm: str) -> float:
        g, gs = graphs[nm]["full"], graphs[nm]["sub"]
        if gs is None:
            return 1.0
        enc.set_graph(g)
        ef = out(False)
        enc.set_graph(gs)
        es = out(False)
        keep = (g["real_mask"] & gs["real_mask"]).numpy()
        return _cos_stability(ef.detach().numpy()[keep], es.detach().numpy()[keep])

    def composite() -> tuple[float, float, float, float]:
        reps, roles, logs = encode_all()
        p = _crosslog_same_role_at_k(reps, roles, logs, val_names)
        t = _startend_transfer(reps, roles, logs, train_names, val_names)
        s = float(np.mean([stability(nm) for nm in val_names]))
        return float(np.mean([p, t, s])), p, t, s

    # raw-fingerprint reference composite (context; the fingerprint is a strong baseline)
    def _fp_reps():
        reps, roles, logs = [], [], []
        for nm in train_names + val_names:
            g = graphs[nm]["full"]
            rm = g["real_mask"]
            reps.append(g["feats"][rm].numpy())
            roles += list(graphs[nm]["roles"][rm.numpy()])
            logs += [nm] * int(rm.sum())
        return np.concatenate(reps), np.array(roles), np.array(logs)

    fr, fro, frl = _fp_reps()
    fp_stab = []
    for nm in val_names:
        g, gs = graphs[nm]["full"], graphs[nm]["sub"]
        if gs is None:
            continue
        keep = (g["real_mask"] & gs["real_mask"]).numpy()
        fp_stab.append(_cos_stability(g["feats"].numpy()[keep], gs["feats"].numpy()[keep]))
    fp_comp = float(np.mean([
        _crosslog_same_role_at_k(fr, fro, frl, val_names),
        _startend_transfer(fr, fro, frl, train_names, val_names),
        float(np.mean(fp_stab)) if fp_stab else 1.0,
    ]))

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "sd": None, "epoch": -1}
    ema = None
    for ep in range(1, epochs + 1):
        enc.train()
        opt.zero_grad()
        con, zs, rs = [], [], []
        for nm in train_names:
            enc.set_graph(graphs[nm]["full"])
            cl = enc.contrastive_loss()
            if cl is not None:
                con.append(cl)
            rm = graphs[nm]["full"]["real_mask"]
            zs.append(out(True)[rm])
            rs += list(graphs[nm]["roles"][rm.numpy()])
        z = torch.cat(zs)
        con_l = torch.stack(con).mean() if con else z.new_zeros(())
        sup_l = _supcon_startend(z, rs, tau_sup)
        ce_l = F.cross_entropy(role_head(z), torch.tensor([_R2I[r] for r in rs]), weight=classw)
        loss = w_con * con_l + w_sup * sup_l + w_ce * ce_l
        loss.backward()
        opt.step()

        enc.eval()
        with torch.no_grad():
            comp, p, t, s = composite()
        ema = comp if ema is None else ema_a * comp + (1 - ema_a) * ema
        history.append({
            "epoch": ep, "loss": loss.item(), "contrastive": con_l.item(), "supcon": sup_l.item(),
            "role_ce": ce_l.item(), "composite": round(comp, 4), "composite_ema": round(ema, 4),
            "xlog_same_role": round(p, 4), "startend_transfer": round(t, 4),
            "stability": round(s, 4), "fingerprint_ref": round(fp_comp, 4),
        })
        if ema > best["score"]:
            best = {"score": ema, "raw": comp, "epoch": ep,
                    "sd": {k: v.clone() for k, v in enc.state_dict().items()}}

    torch.save(best["sd"], run_dir / "role_encoder.pt")
    write_learning_curve(run_dir / "role_learning_curve.csv", history)
    _plot_quality(run_dir / "role_quality.png", history, fp_comp, best["epoch"],
                  train_names, val_names)
    registry.finish(ctx, metrics={
        "aggregator": aggregator,
        "best_composite": round(best["score"], 4),
        "best_composite_raw": round(best.get("raw", best["score"]), 4),
        "best_epoch": best["epoch"], "fingerprint_ref": round(fp_comp, 4),
        "beats_fingerprint": bool(best.get("raw", 0) > fp_comp),
    })
    print(f"[{config.get('name', 'role')}] best composite {best['score']:.3f} "
          f"(raw {best.get('raw', 0):.3f}) @ epoch {best['epoch']}  "
          f"vs fingerprint {fp_comp:.3f}  -> {run_dir}")
    return run_dir


def _plot_quality(png: Path, history, fp_comp, best_epoch, train_names, val_names) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # plotting is best-effort; the CSV is the source of truth
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ep, [h["composite_ema"] for h in history], color="#7f77dd", lw=2, label="composite (EMA)")
    for key, color in [("xlog_same_role", "#e4572e"), ("startend_transfer", "#378add"),
                       ("stability", "#639922")]:
        ax.plot(ep, [h[key] for h in history], color=color, lw=1, ls="--", label=key.replace("_", "-"))
    ax.axhline(fp_comp, color="0.4", ls=":", lw=1.5)
    ax.text(ep[-1], fp_comp + 0.01, f"raw-fingerprint composite {fp_comp:.2f}",
            ha="right", fontsize=8, color="0.4")
    ax.axvline(best_epoch, color="#7f77dd", ls=":", lw=1, alpha=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out quality")
    ax.set_title(f"Standalone role-encoder quality - train {train_names} · held-out {val_names}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
