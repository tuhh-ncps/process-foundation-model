"""Zero-shot open-vocabulary next-activity via the pretrained matching head.

No training, no labels: a role backbone's pretrained query projection ``q_i = W h_i`` is
cosine-matched against the EVAL catalogue's candidate table ``e(c)`` — both live in the same
role space, so the head predicts activities of a process it never saw. This is the sharpest
test of cross-domain transfer: next-activity on the eval TEST split with the backbone (and
candidate encoder) fully frozen.

Outputs per backbone: top-k accuracy and a confusion matrix (CSV + PNG). The eval catalogue's
fingerprints/DFG are fit on the eval TRAIN split only (roles.py leakage contract); the metrics
are computed on the TEST split.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pm_foundation.data.dataset import (
    NEXT_ACTIVITY_IGNORE_INDEX,
    SupervisedTraceDataset,
    collate_supervised,
)
from pm_foundation.data.preprocessing import (
    FeatureSpec,
    SplitStrategy,
    build_traces,
    fit_feature_spec,
    split_log,
)
from pm_foundation.data.readers import get_reader
from pm_foundation.data.roles import fit_role_graph
from pm_foundation.evaluation.confusion import build_confusion, plot_confusion, write_confusion_csv
from pm_foundation.evaluation.label_efficiency import _strip_to_control_flow
from pm_foundation.experiments import RunManifest, RunRegistry
from pm_foundation.models import TraceBackbone


def _load_role_backbone(
    registry: RunRegistry, run_id: str, eval_role_graph: dict[str, torch.Tensor]
) -> tuple[TraceBackbone, torch.nn.Linear]:
    """Load a frozen role backbone + its pretrained matching query; install the eval catalogue."""
    run_dir = registry.run_dir("backbone", run_id)
    spec = FeatureSpec.load(run_dir / "feature_spec.json")
    model_cfg = dict(RunManifest.load(run_dir / "manifest.json").config["model"])
    if int(model_cfg.get("role_dim", 0)) <= 0:
        raise ValueError(
            f"backbone {run_id!r} has no role channel — zero-shot matching needs role_dim>0"
        )
    model_cfg["causal"] = True
    bb = TraceBackbone.from_config(model_cfg, spec)
    bb.load_state_dict(torch.load(run_dir / "backbone.pt", map_location="cpu"))
    bb.eval()
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.embedding.role_encoder.set_graph(eval_role_graph)  # swap in the eval catalogue

    heads = torch.load(run_dir / "ar_heads.pt", map_location="cpu")
    if "match_query" not in heads:
        raise ValueError(
            f"backbone {run_id!r} has no saved matching head (not a role/matching run)"
        )
    d_model = int(model_cfg["d_model"])
    role_dim = int(model_cfg["role_dim"])
    match_query = torch.nn.Linear(d_model, role_dim)
    match_query.load_state_dict(heads["match_query"])
    match_query.eval()
    return bb, match_query


def _adapt_role_encoder(bb: TraceBackbone, steps: int, lr: float) -> float | None:
    """Label-free adaptation: run ``steps`` of the role encoder's OWN self-supervised contrastive
    loss on the INSTALLED (target-domain) graph, so the GIN specializes to the target's structure —
    no labels used. Only the role encoder is updated; the rest of the frozen backbone is untouched.
    Returns the final contrastive loss (or None if the graph has < 2 real activities)."""
    enc = bb.embedding.role_encoder
    enc.train()
    for p in enc.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    last: float | None = None
    for _ in range(steps):
        opt.zero_grad()
        loss = enc.contrastive_loss()
        if loss is None:
            break
        loss.backward()
        opt.step()
        last = float(loss.detach())
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return last


@torch.no_grad()
def _score(
    bb: TraceBackbone, match_query: torch.nn.Linear, loader: DataLoader[Any], real_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictions_topk_sorted, targets) over all valid next-activity positions.

    Candidates are restricted to REAL activities; logits are cosine(q, e(c)) (rank-invariant to
    the learned scale). Predictions are returned as a (N, C) matrix of candidate ids sorted by
    score descending, so any top-k is a prefix slice.
    """
    table = F.normalize(bb.embedding.role_encoder(augment=False), dim=-1, eps=1e-8)  # (V, r)
    cand = table[real_ids]  # (C, r) — real candidates only
    all_ranked, all_targets = [], []
    for batch in loader:
        states = bb.forward_batch(batch).event_states  # (B, L, d)
        q = F.normalize(match_query(states), dim=-1)  # (B, L, r)
        sims = q @ cand.t()  # (B, L, C) cosine over real candidates
        targets = batch["next_activity"]  # (B, L), eval-vocab ids; IGNORE at pad/last
        valid = targets != NEXT_ACTIVITY_IGNORE_INDEX
        ranked_ids = real_ids[sims.argsort(dim=-1, descending=True)]  # (B, L, C) candidate ids
        all_ranked.append(ranked_ids[valid])  # (n_valid, C)
        all_targets.append(targets[valid])  # (n_valid,)
    return torch.cat(all_ranked), torch.cat(all_targets)


def run_zero_shot_matching(config: dict[str, Any]) -> Path:
    """Zero-shot matching eval for one or more role backbones; persists metrics + confusion."""
    backbones: dict[str, str] = dict(config["backbones"])
    topk = sorted({int(k) for k in config.get("topk", [1, 3, 5])})
    min_len = int(config.get("min_trace_len", 2))
    split = tuple(config.get("split", (0.70, 0.15, 0.15)))
    batch_size = int((config.get("probe") or {}).get("batch_size", 128))
    # Label-free GIN adaptation: >0 steps of the role encoder's own contrastive loss on the TARGET
    # graph before matching (no labels), so the GIN specializes to the target domain's structure.
    adapt_cfg = dict(config.get("adapt") or {})
    adapt_steps = int(adapt_cfg.get("steps", 0))
    adapt_lr = float(adapt_cfg.get("lr", 1.0e-3))

    log = get_reader(
        str(
            config["eval_log"].get("format") or Path(config["eval_log"]["path"]).suffix.lstrip(".")
        ),
        **(
            {"max_traces": config["eval_log"]["max_traces"]}
            if config["eval_log"].get("max_traces")
            else {}
        ),
    ).read(config["eval_log"]["path"])
    if bool(config.get("strip_to_control_flow", True)):
        log = _strip_to_control_flow(log)
    splits = split_log(
        build_traces(log, min_trace_len=min_len), SplitStrategy.TEMPORAL, split, seed=0
    )

    eval_spec = fit_feature_spec(
        splits.train, max_seq_len=int(config["model"].get("max_seq_len", 64))
    )
    eval_vocab = eval_spec.activity_vocab
    # Leakage contract: fingerprints/DFG from the eval TRAIN split only (never the scored TEST).
    eval_role_graph = fit_role_graph(list(splits.train.traces), eval_vocab)
    real_ids = torch.nonzero(eval_role_graph["real_mask"], as_tuple=False).squeeze(-1)
    names = [eval_vocab.to_list()[i] for i in real_ids.tolist()]

    registry = RunRegistry(config.get("output_dir", "outputs"))
    ctx = registry.start(
        "zero_shot",
        config,
        name=config.get("name"),
        data={
            "eval_log": config["eval_log"]["path"],
            "role_corpus": {"split": "train", "n_traces": len(splits.train.traces)},
            "n_candidates": len(real_ids),
            "topk": topk,
            "adapt": ({"steps": adapt_steps, "lr": adapt_lr} if adapt_steps > 0 else None),
        },
    )

    rows: list[dict[str, Any]] = []
    for alias, run_id in backbones.items():
        bb, match_query = _load_role_backbone(registry, run_id, eval_role_graph)
        adapt_note = ""
        if adapt_steps > 0:  # label-free: adapt the GIN to the target graph before matching
            fl = _adapt_role_encoder(bb, adapt_steps, adapt_lr)
            adapt_note = f"  adapted({adapt_steps} steps, loss={fl:.3f})" if fl is not None else ""
        # role_ids encode the TEST traces with the EVAL catalogue so e(a_i) is looked up there.
        ds = SupervisedTraceDataset(
            splits.test, eval_spec, target_activity_vocab=eval_vocab, role_vocab=eval_vocab
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_supervised)
        ranked, targets = _score(bb, match_query, loader, real_ids)
        n = targets.shape[0]
        rec = {
            "backbone_alias": alias,
            "backbone_run_id": run_id,
            "n_predictions": int(n),
            "adapt_steps": adapt_steps,
        }
        for k in topk:
            kk = min(k, ranked.shape[1])
            hit = (ranked[:, :kk] == targets.unsqueeze(1)).any(dim=1).float().mean()
            rec[f"top{k}_acc"] = round(float(hit), 4)
        rows.append(rec)

        cm = build_confusion(ranked[:, 0], targets, real_ids)
        write_confusion_csv(cm, names, ctx.dir / f"confusion_{alias}.csv")
        plot_confusion(
            cm,
            names,
            ctx.dir / f"confusion_{alias}.png",
            f"{alias}: zero-shot next-activity (top-1 recall)",
        )
        print(
            f"[zero-shot] {alias}: n={n:,}  "
            + "  ".join(f"top{k}={rec[f'top{k}_acc']:.3f}" for k in topk)
            + adapt_note,
            flush=True,
        )

    # Reference baselines on the SAME test targets, so a backbone's top-k is directly interpretable:
    #   frequency = rank candidates by marginal next-activity frequency in the TRAIN split (leak-free)
    #               and predict that fixed ranking everywhere — the "always guess the common step" floor;
    #   uniform   = k / n_candidates (chance). A backbone only shows real matching signal above these.
    if (
        rows
    ):  # `targets` (from the loop) = the valid test next-activity ids, identical across backbones
        from collections import Counter

        freq: Counter[int] = Counter()
        for t in splits.train.traces:
            ids = [eval_vocab.encode(e.activity) for e in t.events]
            for nxt in ids[1:]:
                freq[nxt] += 1
        ranked_freq = sorted(real_ids.tolist(), key=lambda c: -freq.get(c, 0))
        tgt = targets.tolist()
        nt = max(len(tgt), 1)
        nc = max(len(real_ids), 1)
        freq_rec = {
            "backbone_alias": "frequency",
            "backbone_run_id": "frequency",
            "n_predictions": len(tgt),
        }
        unif_rec = {
            "backbone_alias": "uniform",
            "backbone_run_id": "uniform",
            "n_predictions": len(tgt),
        }
        for k in topk:
            top_set = set(ranked_freq[:k])
            freq_rec[f"top{k}_acc"] = round(sum(1 for x in tgt if x in top_set) / nt, 4)
            unif_rec[f"top{k}_acc"] = round(min(k, len(real_ids)) / nc, 4)
        rows.extend([freq_rec, unif_rec])
        print(
            "[zero-shot] frequency: "
            + "  ".join(f"top{k}={freq_rec[f'top{k}_acc']:.3f}" for k in topk)
            + f"   uniform top1={unif_rec['top1_acc']:.3f}",
            flush=True,
        )

    with (ctx.dir / "metrics.json").open("w") as fh:
        json.dump({"n_candidates": len(real_ids), "topk": topk, "results": rows}, fh, indent=2)
    with (ctx.dir / "topk.csv").open("w", newline="") as fh:
        fields = [
            "backbone_alias",
            "backbone_run_id",
            "n_predictions",
            "adapt_steps",
            *(f"top{k}_acc" for k in topk),
        ]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    real_backbones = {a: r for a, r in backbones.items() if r != "random"}
    for alias, run_id in real_backbones.items():
        registry.link_backbone_to_eval(
            run_id, ctx.run_id, info={"alias": alias, "mode": "zero_shot"}
        )
    registry.finish(ctx, links={"backbones": backbones})
    return ctx.dir
