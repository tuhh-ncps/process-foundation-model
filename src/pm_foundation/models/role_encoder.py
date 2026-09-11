"""Candidate-record activity encoder: fingerprints -> directed message-passing -> fused e(a).

Encodes every activity of a catalogue as a **record** of four channels:
    role      — directed message-passing embedding of its fingerprint over the DFG (structural;
                GIN-style update with a mean aggregator by default — see DirectedGinLayer)
    name      — hashed char-trigram bag embedding of the label (lexical)
    stats     — the raw fingerprint through a small MLP (un-smoothed evidence)
    graph ctx — P(in)/P(out)-weighted mean of neighbor roles (1-hop context)
fused into one ``role_dim`` vector per activity. The table serves BOTH sides of the
model (tied): the hybrid input channel ``ID (+) e(a_i)`` and the candidate bank of the
open-vocabulary matching head.

Vocabulary-freedom: inputs are corpus statistics + label text only — never activity
ids — so a NEW dataset's catalogue maps through the frozen encoder with no retraining
(``set_graph`` swaps the graph buffers; see ``pm_foundation.data.roles`` for the
train-split-only leakage contract).

Training: end-to-end from the backbone losses, plus a node-level **contrastive** loss
(InfoNCE between two augmented views: edge dropout on the DFG + channel dropout) that
shapes the space independently of any dataset-specific head and prevents role collapse.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pm_foundation.data.roles import N_ROLE_FEATURES, NAME_HASH_BUCKETS, NAME_TRIGRAM_SLOTS


class DirectedGinLayer(nn.Module):
    """GIN update (Xu et al. 2019) with separate directed in-/out aggregation.

    ``agg = (1+eps)*h + adj_in @ h + adj_out @ h``. The AGGREGATOR is set by the adjacency the caller
    installs (``model.aggregator`` via ``roles.apply_aggregator``): ``sum`` (DEFAULT) uses a binary
    adjacency so ``adj @ h`` is the neighbour SUM that gives GIN its Weisfeiler-Leman power — true
    GIN. ``mean`` uses ``fit_role_graph``'s ROW-NORMALIZED transition matrix, a probability-weighted
    mean (strictly weaker, but scale-invariant across logs of different sizes). The encoder learns to
    its adjacency scale, so the SAME aggregator is used at pretrain and inference — it is persisted in
    the manifest and wired across pretrain / eval / role_pretrain. Config default is ``sum``; the code
    fallback stays ``mean`` so pre-flip (mean-trained) backbones without the key still evaluate
    correctly. A 5-seed standalone A/B found the two tied on role-space quality (~0.81; depths 1-3
    tied, 4-5 over-smooth) — sum is chosen as the theoretically-correct form; a joint-training GPU
    check remains pending."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj_in: torch.Tensor, adj_out: torch.Tensor) -> torch.Tensor:
        agg = (1 + self.eps) * h + adj_in @ h + adj_out @ h
        out: torch.Tensor = self.norm(h + self.mlp(agg))  # residual keeps features flowing
        return out


class ActivityEncoder(nn.Module):
    """The candidate-record encoder; holds the current catalogue's graph as buffers."""

    def __init__(
        self,
        n_activities: int,
        role_dim: int = 64,
        n_layers: int = 2,
        edge_dropout: float = 0.2,
        channel_dropout: float = 0.15,
        tau: float = 0.2,
    ) -> None:
        super().__init__()
        self.role_dim = role_dim
        self.edge_dropout = edge_dropout
        self.channel_dropout = channel_dropout
        self.tau = tau

        self.inp = nn.Linear(N_ROLE_FEATURES, role_dim)
        self.gin = nn.ModuleList(DirectedGinLayer(role_dim) for _ in range(n_layers))
        self.name_emb = nn.Embedding(NAME_HASH_BUCKETS, role_dim, padding_idx=0)
        self.stats_mlp = nn.Sequential(
            nn.Linear(N_ROLE_FEATURES, role_dim), nn.GELU(), nn.Linear(role_dim, role_dim)
        )
        self.ctx_proj = nn.Linear(2 * role_dim, role_dim)
        self.fuse = nn.Sequential(
            nn.Linear(4 * role_dim, 2 * role_dim), nn.GELU(), nn.Linear(2 * role_dim, role_dim)
        )
        self.out_norm = nn.LayerNorm(role_dim)

        # Catalogue buffers — swapped per dataset via set_graph (checkpointed with the model).
        self.register_buffer("feats", torch.zeros(n_activities, N_ROLE_FEATURES))
        self.register_buffer("adj_in", torch.zeros(n_activities, n_activities))
        self.register_buffer("adj_out", torch.zeros(n_activities, n_activities))
        self.register_buffer(
            "name_ids", torch.zeros(n_activities, NAME_TRIGRAM_SLOTS, dtype=torch.long)
        )
        self.register_buffer("real_mask", torch.zeros(n_activities, dtype=torch.bool))

    @torch.no_grad()
    def set_graph(self, graph: dict[str, torch.Tensor]) -> None:
        """Install a catalogue (from ``fit_role_graph``); resizes buffers if V differs."""
        dev = self.feats.device
        self.feats = graph["feats"].to(dev)
        self.adj_in = graph["adj_in"].to(dev)
        self.adj_out = graph["adj_out"].to(dev)
        self.name_ids = graph["name_ids"].to(dev)
        self.real_mask = graph["real_mask"].to(dev)

    def _channels(self, augment: bool) -> torch.Tensor:
        feats, a_in, a_out = self.feats, self.adj_in, self.adj_out
        if augment:
            feats = F.dropout(feats, p=0.1, training=True)
            keep_in = torch.rand_like(a_in) >= self.edge_dropout
            keep_out = torch.rand_like(a_out) >= self.edge_dropout
            a_in = a_in * keep_in / (1 - self.edge_dropout)
            a_out = a_out * keep_out / (1 - self.edge_dropout)

        h = self.inp(feats)
        for layer in self.gin:
            h = layer(h, a_in, a_out)  # role channel (V, d)
        ctx = self.ctx_proj(torch.cat([a_in @ h, a_out @ h], dim=-1))  # neighbor-role summary
        name_vecs = self.name_emb(self.name_ids)  # (V, T, d)
        name_mask = (self.name_ids != 0).unsqueeze(-1).float()
        name = (name_vecs * name_mask).sum(1) / name_mask.sum(1).clamp(min=1.0)
        stats = self.stats_mlp(feats)

        channels = [h, name, stats, ctx]
        if augment and self.channel_dropout > 0:
            # Drop whole channels (never all four) so no single channel — especially
            # name — becomes a shortcut the others can't cover for.
            drop = torch.rand(len(channels)) < self.channel_dropout
            if bool(drop.all()):
                drop[0] = False
            channels = [
                torch.zeros_like(c) if bool(d) else c for c, d in zip(channels, drop, strict=True)
            ]
        fused: torch.Tensor = self.out_norm(self.fuse(torch.cat(channels, dim=-1)))
        # Reserved rows (PAD/UNK/CLS/MASK) stay zero so padded positions embed to zero.
        return fused * self.real_mask.unsqueeze(-1)

    def forward(self, augment: bool = False) -> torch.Tensor:
        """Return the full catalogue table ``(V, role_dim)``."""
        return self._channels(augment)

    def contrastive_loss(self) -> torch.Tensor | None:
        """Node-level InfoNCE between two augmented views (real activities only)."""
        mask = self.real_mask
        if int(mask.sum()) < 2:
            return None
        z1 = F.normalize(self._channels(augment=True)[mask], dim=-1)
        z2 = F.normalize(self._channels(augment=True)[mask], dim=-1)
        logits = z1 @ z2.t() / self.tau
        labels = torch.arange(z1.shape[0], device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# The candidate/role encoder is a first-class, swappable component (its own role_encoder.pt).
RoleEncoder = ActivityEncoder


# Feature subsets for the role-embedding ablation candidates (indices into the 15-dim fingerprint).
ROLE_FEATURE_SUBSETS = {
    "all": list(range(N_ROLE_FEATURES)),  # 15
    # 5 non-positional feats: self_loop_p, in_gap_median, out_gap_median, support, rework_p
    "five": [2, 3, 5, 9, 14],
    # 11 strongest features (PCA-guided backward elimination; reconstructs 87% of the full 15-dim
    # signal). Drops the mutually-correlated PC1 centrality cluster (pagerank, betweenness,
    # self_loop_p, pred_entropy) -- a GIN can re-derive degree/centrality from the adjacency it
    # already receives -- and keeps timing, position, frequency and rework, which it cannot.
    "eleven": [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
    # featureless ablation: NO fingerprint input at all. The GIN is fed a constant, so e(a)
    # carries only DFG topology (with sum-aggregation, depth-1 recovers degree; deeper layers
    # build WL-style structural signatures). Isolates "how much is the fingerprint worth?".
    "none": [],
}


class RoleEmbedder(nn.Module):
    """Structural-only activity embedder for the role-embedding ablation — NO name/stats/ctx fusion.

    Produces ``e(a)`` in ``R^role_dim`` from the fingerprint (+ DFG for ``gin``), a drop-in for
    ``ActivityEncoder`` (same ``set_graph`` / ``forward(augment)`` / ``contrastive_loss`` interface,
    same swappable ``role_encoder.pt``). ``arch``:
      * ``raw`` — the raw fingerprint itself (frozen, no params; ``role_dim`` must equal #features),
      * ``mlp`` — a per-node MLP over the fingerprint (no message passing),
      * ``gin`` — directed GIN over the DFG (the current structural channel, in isolation).
    ``feature_subset`` selects the input columns ("all"=15 or "five"=the 5 non-positional feats).
    """

    def __init__(
        self,
        n_activities: int,
        role_dim: int = 64,
        arch: str = "gin",
        n_layers: int = 1,
        feature_subset: str = "all",
        edge_dropout: float = 0.2,
        feat_dropout: float = 0.1,
        tau: float = 0.2,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.feature_idx = list(ROLE_FEATURE_SUBSETS.get(feature_subset, feature_subset))
        self.featureless = not self.feature_idx
        n_in = 1 if self.featureless else len(self.feature_idx)  # featureless: constant input
        self.edge_dropout = edge_dropout
        self.feat_dropout = feat_dropout
        self.tau = tau
        if arch == "raw":
            if self.featureless:
                raise ValueError("role_arch='raw' needs features; feature_subset='none' is GIN/MLP only")
            if role_dim != n_in:
                raise ValueError(f"raw role_dim must equal #features ({n_in}), got {role_dim}")
            self.role_dim = n_in
        elif arch == "mlp":
            self.role_dim = role_dim
            self.net = nn.Sequential(nn.Linear(n_in, role_dim), nn.GELU(), nn.Linear(role_dim, role_dim))
        elif arch == "gin":
            self.role_dim = role_dim
            self.inp = nn.Linear(n_in, role_dim)
            self.gin = nn.ModuleList(DirectedGinLayer(role_dim) for _ in range(n_layers))
        else:
            raise ValueError(f"unknown role arch {arch!r} (expected raw|mlp|gin)")
        # Catalogue buffers (no name_ids — this variant has no name channel).
        self.register_buffer("feats", torch.zeros(n_activities, N_ROLE_FEATURES))
        self.register_buffer("adj_in", torch.zeros(n_activities, n_activities))
        self.register_buffer("adj_out", torch.zeros(n_activities, n_activities))
        self.register_buffer("real_mask", torch.zeros(n_activities, dtype=torch.bool))

    @torch.no_grad()
    def set_graph(self, graph: dict[str, torch.Tensor]) -> None:
        dev = self.feats.device
        self.feats = graph["feats"].to(dev)
        self.adj_in = graph["adj_in"].to(dev)
        self.adj_out = graph["adj_out"].to(dev)
        self.real_mask = graph["real_mask"].to(dev)

    def _embed(self, augment: bool) -> torch.Tensor:
        if self.featureless:
            f = self.feats.new_ones(self.feats.shape[0], 1)
        else:
            f = self.feats[:, self.feature_idx]
            if augment and self.feat_dropout > 0.0:
                f = F.dropout(f, p=self.feat_dropout, training=True)
        if self.arch == "raw":
            e = f
        elif self.arch == "mlp":
            e = self.net(f)
        else:  # gin
            ai, ao = self.adj_in, self.adj_out
            if augment and self.edge_dropout > 0.0:
                ai = ai * (torch.rand_like(ai) >= self.edge_dropout).float() / (1 - self.edge_dropout)
                ao = ao * (torch.rand_like(ao) >= self.edge_dropout).float() / (1 - self.edge_dropout)
            h = self.inp(f)
            for layer in self.gin:
                h = layer(h, ai, ao)
            e = h
        return e * self.real_mask.unsqueeze(-1)

    def forward(self, augment: bool = False) -> torch.Tensor:
        return self._embed(augment)

    def contrastive_loss(self) -> torch.Tensor | None:
        """InfoNCE between two augmented views (for Phase-1 SSL). None for the raw (untrained) arch."""
        if self.arch == "raw":
            return None
        mask = self.real_mask
        if int(mask.sum()) < 2:
            return None
        z1 = F.normalize(self._embed(augment=True)[mask], dim=-1)
        z2 = F.normalize(self._embed(augment=True)[mask], dim=-1)
        logits = z1 @ z2.t() / self.tau
        labels = torch.arange(z1.shape[0], device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def build_role_module(model_cfg: dict, n_activities: int) -> nn.Module:
    """Factory: build the role module from ``model_cfg`` (shared by from_config and role_pretrain).

    ``role_arch`` selects it: ``fused`` (default) -> the full 4-channel ActivityEncoder; else a
    structural-only RoleEmbedder (raw|mlp|gin) for the role-embedding ablation.
    """
    arch = str(model_cfg.get("role_arch", "fused"))
    role_dim = int(model_cfg.get("role_dim", 64))
    if arch == "fused":
        return ActivityEncoder(
            n_activities,
            role_dim=role_dim,
            n_layers=int(model_cfg.get("role_layers", 2)),
            edge_dropout=float(model_cfg.get("edge_dropout", 0.2)),
            channel_dropout=float(model_cfg.get("channel_dropout", 0.15)),
            tau=float(model_cfg.get("tau", 0.2)),
        )
    return RoleEmbedder(
        n_activities,
        role_dim=role_dim,
        arch=arch,
        n_layers=int(model_cfg.get("role_layers", 1)),
        feature_subset=str(model_cfg.get("role_feature_subset", "all")),
        edge_dropout=float(model_cfg.get("edge_dropout", 0.2)),
        feat_dropout=float(model_cfg.get("feat_dropout", 0.1)),
        tau=float(model_cfg.get("tau", 0.2)),
    )
