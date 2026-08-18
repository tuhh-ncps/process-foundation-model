"""Candidate-record activity encoder: fingerprints -> directed GIN -> fused e(a).

Encodes every activity of a catalogue as a **record** of four channels:
    role      — directed-GIN embedding of its fingerprint over the DFG (structural)
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
    """GIN layer with separate in-/out-neighbor aggregation (directed, weighted)."""

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
