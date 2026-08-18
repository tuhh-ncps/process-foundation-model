"""Per-event embedding: fuse activity, temporal, and attribute features.

Consumes the batch tensors produced by the data layer (see ``docs/features.md``
§4) and produces ``(B, L+1, d_model)`` event vectors with a learned ``CLS`` trace
token prepended at position 0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from pm_foundation.data.preprocessing import FeatureSpec


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019): a LEARNABLE time embedding — the trainable generalization
    of fixed Fourier bands. For each source scalar τ it emits one linear term ``w0·τ + b0`` and
    ``k`` sinusoids ``sin(w_i·τ + b_i)`` with learnable frequencies ``w`` and phases ``b``
    (sin-with-phase subsumes cos). Output dim per source = ``k + 1``."""

    def __init__(self, n_sources: int, k: int) -> None:
        super().__init__()
        self.n_sources = n_sources
        self.k = k
        self.out_dim = n_sources * (k + 1)
        self.w0 = nn.Parameter(torch.ones(n_sources))  # linear slope ~1 (near-identity)
        self.b0 = nn.Parameter(torch.zeros(n_sources))
        self.w = nn.Parameter(torch.randn(n_sources, k))  # learnable frequencies
        self.b = nn.Parameter(torch.zeros(n_sources, k))  # learnable phases

    def forward(
        self, tau: torch.Tensor
    ) -> torch.Tensor:  # (..., n_sources) -> (..., n_sources*(k+1))
        lin = self.w0 * tau + self.b0  # (..., n_sources)
        ang = tau.unsqueeze(-1) * self.w + self.b  # (..., n_sources, k)
        per = torch.sin(ang).flatten(start_dim=-2)  # (..., n_sources*k)
        return torch.cat([lin, per], dim=-1)


class TimeEncoder(nn.Module):
    """Dedicated time encoder: an MLP over the normalized scalar time features -> d_model.

    V1/V2 use it plain. It optionally expands only the continuous monotonic columns
    (``fourier_indices`` — z_log_delta, z_log_elapsed); cyclic (hour/dow) and binary features
    are left untouched (already periodic/discrete — re-encoding them is feature laundering):
      - V3 (``fourier_bands``): fixed bands, appends ``sin(π·b·x), cos(π·b·x)``.
      - V4 (``time2vec_k``): learnable Time2Vec over those columns."""

    def __init__(
        self,
        n_time_features: int,
        d_model: int,
        dropout: float = 0.1,
        *,
        fourier_bands: tuple[float, ...] = (),
        fourier_indices: tuple[int, ...] = (0, 1),
        time2vec_k: int = 0,
    ) -> None:
        super().__init__()
        self.fourier_bands = tuple(fourier_bands)
        self.fourier_indices = list(fourier_indices)
        self.time2vec = Time2Vec(len(self.fourier_indices), time2vec_k) if time2vec_k > 0 else None
        n_fourier = 2 * len(self.fourier_indices) * len(self.fourier_bands)
        n_t2v = self.time2vec.out_dim if self.time2vec is not None else 0
        self.net = nn.Sequential(
            nn.Linear(n_time_features + n_fourier + n_t2v, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:  # (B, L, T) -> (B, L, d_model)
        x = time_features
        if self.fourier_bands or self.time2vec is not None:
            cont = x[..., self.fourier_indices]  # continuous monotonic columns
            parts = [x]
            for band in self.fourier_bands:  # V3 fixed Fourier
                ang = math.pi * band * cont
                parts.extend((torch.sin(ang), torch.cos(ang)))
            if self.time2vec is not None:  # V4 learnable Time2Vec
                parts.append(self.time2vec(cont))
            x = torch.cat(parts, dim=-1)
        return self.net(x)


class EventEmbedding(nn.Module):
    """Maps raw per-event features to ``d_model`` vectors with a CLS token.

    Fuses a learned activity embedding, the precomputed temporal features, one
    learned embedding per categorical attribute, and the standardized numeric
    attributes; projects the concatenation to ``d_model``; adds a learned
    positional embedding; and prepends a learned ``CLS`` token.
    """

    def __init__(
        self,
        n_activities: int,
        categorical_cardinalities: list[int],
        n_numeric: int,
        n_time_features: int,
        d_model: int = 256,
        activity_embedding_dim: int = 128,
        attribute_embedding_dim: int = 32,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        position: str = "learned",
        time_encoder: str = "none",
        fourier_bands: tuple[float, ...] = (1, 2, 4, 8),
        time2vec_k: int = 8,
        role_dim: int = 0,
        role_layers: int = 2,
        id_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if position not in ("learned", "rope"):
            raise ValueError(f"position must be 'learned' or 'rope', got {position!r}")
        if time_encoder not in ("none", "mlp", "gated", "fourier", "time2vec"):
            raise ValueError(
                "time_encoder must be 'none', 'mlp', 'gated', 'fourier', or 'time2vec', "
                f"got {time_encoder!r}"
            )
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.position = position
        self.time_encoder_mode = time_encoder
        self.n_numeric = n_numeric
        self.id_dropout = id_dropout  # train-time UNK-masking of the ID channel (role-reliance)

        # Hybrid role channel (vocabulary-free): the ActivityEncoder maps the catalogue's
        # fingerprints/DFG/names to e(a); its table joins the input as ID (+) e(a_i) and is
        # the tied candidate bank of the AR matching head. Graph buffers are installed via
        # set_graph (train-split-only corpus — see data/roles.py leakage contract).
        self.role_encoder = None
        if role_dim > 0:
            from pm_foundation.models.role_encoder import ActivityEncoder

            self.role_encoder = ActivityEncoder(
                n_activities, role_dim=role_dim, n_layers=role_layers
            )

        # PAD id is 0 across activity and categorical vocabularies, so padding_idx=0
        # keeps padded positions at a zero (untrained) vector.
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(card, attribute_embedding_dim, padding_idx=0)
            for card in categorical_cardinalities
        )
        attr_dim = len(categorical_cardinalities) * attribute_embedding_dim + n_numeric

        if time_encoder in ("mlp", "gated", "fourier", "time2vec"):
            # Additive fusion: each stream is projected to d_model, then summed.
            #   V1 (mlp)      : event_emb = activity_emb + time_emb + attr_emb
            #   V2 (gated)    : event_emb = LayerNorm(activity_emb + gate * time_emb + attr_emb)
            #   V3 (fourier)  : V2 + the time encoder Fourier-expands z_log_delta / z_log_elapsed
            #   V4 (time2vec) : V2 + a learnable Time2Vec over those same columns
            self.activity_embedding = nn.Embedding(n_activities, d_model, padding_idx=0)
            bands = tuple(fourier_bands) if time_encoder == "fourier" else ()
            t2v_k = time2vec_k if time_encoder == "time2vec" else 0
            self.time_mlp = TimeEncoder(
                n_time_features, d_model, dropout, fourier_bands=bands, time2vec_k=t2v_k
            )
            self.attr_proj = nn.Linear(attr_dim, d_model) if attr_dim else None
            self.role_proj = nn.Linear(role_dim, d_model) if role_dim > 0 else None
            if time_encoder in ("gated", "fourier", "time2vec"):
                # Per-channel learned gate on the time stream. Bias -2.0 -> sigmoid(-2)≈0.12 at
                # init, so control-flow leads and time enters gently (the known-good prior); the
                # model opens specific channels only if temporal signal helps. LayerNorm keeps
                # the additive sum well-scaled.
                self.time_gate = nn.Parameter(torch.full((d_model,), -2.0))
                self.fused_norm = nn.LayerNorm(d_model)
        else:
            # Baseline: concatenate raw features (time included) and project once.
            self.activity_embedding = nn.Embedding(
                n_activities, activity_embedding_dim, padding_idx=0
            )
            concat_dim = activity_embedding_dim + n_time_features + attr_dim + role_dim
            self.fusion = nn.Linear(concat_dim, d_model)
        # Learned absolute positions (+1 for the CLS token). Skipped for RoPE, which encodes
        # position inside attention instead — so RoPE has no position table and no length cap.
        self.position_embedding = (
            nn.Embedding(max_seq_len + 1, d_model) if position == "learned" else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        activity_ids: torch.Tensor,  # (B, L) long
        time_features: torch.Tensor,  # (B, L, T) float
        categorical_ids: torch.Tensor,  # (B, L, n_cat) long
        numeric_features: torch.Tensor,  # (B, L, n_num) float
        role_ids: torch.Tensor | None = None,  # (B, L) long, CURRENT catalogue's ids
    ) -> torch.Tensor:
        """Return event vectors with a prepended CLS token: ``(B, L+1, d_model)``."""
        bsz, length = activity_ids.shape

        # Hybrid role channel: e(a_i) looked up in the CURRENT catalogue's table. In-domain
        # role_ids == activity_ids (same vocab); cross-domain the batch carries role_ids
        # encoded with the eval catalogue, so the channel survives the vocabulary change.
        role_emb = None
        if self.role_encoder is not None:
            table = self.role_encoder()  # (V_role, role_dim); reserved rows are zero
            ids = role_ids if role_ids is not None else activity_ids
            role_emb = F.embedding(ids, table)

        # ID-channel dropout: during TRAINING, replace a fraction of REAL activity ids with UNK so
        # the model must predict from the (vocabulary-free) role channel — making the role space
        # load-bearing and thus transferable to disjoint-vocab domains. role_emb was computed from
        # the ORIGINAL ids above, so dropped positions become exactly the cross-domain input
        # pattern (ID=UNK, role=real). No-op at eval, without a role channel, or when id_dropout=0.
        act_ids = activity_ids
        if self.training and self.id_dropout > 0.0 and self.role_encoder is not None:
            drop = (torch.rand_like(activity_ids, dtype=torch.float) < self.id_dropout) & (
                activity_ids != 0
            )
            act_ids = activity_ids.masked_fill(drop, 1)  # 1 = UNK id (reserved: PAD=0, UNK=1)

        if self.time_encoder_mode in ("mlp", "gated", "fourier", "time2vec"):
            # Additive: activity_emb + (gate *) time_emb + attr_emb (each already d_model).
            gated = self.time_encoder_mode in ("gated", "fourier", "time2vec")
            time_emb = self.time_mlp(time_features)  # (V3: Fourier-expanded inside the encoder)
            if gated:
                time_emb = torch.sigmoid(self.time_gate) * time_emb  # V2/V3: gated time stream
            fused = self.activity_embedding(act_ids) + time_emb
            if role_emb is not None:
                fused = fused + self.role_proj(role_emb)  # hybrid: ID emb (+) role channel
            if self.attr_proj is not None:
                attr_parts = [
                    emb(categorical_ids[:, :, j])
                    for j, emb in enumerate(self.categorical_embeddings)
                ]
                if self.n_numeric:
                    attr_parts.append(numeric_features)
                fused = fused + self.attr_proj(torch.cat(attr_parts, dim=-1))
            if gated:
                fused = self.fused_norm(fused)  # V2/V3: LayerNorm the additive sum
        else:
            parts: list[torch.Tensor] = [self.activity_embedding(act_ids), time_features]
            if role_emb is not None:
                parts.append(role_emb)  # hybrid: ID emb || role channel
            for j, embedding in enumerate(self.categorical_embeddings):
                parts.append(embedding(categorical_ids[:, :, j]))
            if self.n_numeric:
                parts.append(numeric_features)
            fused = self.fusion(torch.cat(parts, dim=-1))  # (B, L, d_model)

        cls = self.cls_token.expand(bsz, 1, self.d_model)
        sequence = torch.cat([cls, fused], dim=1)  # (B, L+1, d_model)

        if self.position_embedding is not None:  # learned absolute positions (RoPE adds none here)
            positions = torch.arange(length + 1, device=activity_ids.device)
            sequence = sequence + self.position_embedding(positions)
        out: torch.Tensor = self.dropout(sequence)
        return out

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any], feature_spec: FeatureSpec) -> EventEmbedding:
        return cls(
            n_activities=feature_spec.n_activities,
            categorical_cardinalities=feature_spec.categorical_cardinalities,
            n_numeric=feature_spec.n_numeric,
            n_time_features=feature_spec.n_time_features,
            d_model=int(model_cfg.get("d_model", 256)),
            activity_embedding_dim=int(model_cfg.get("activity_embedding_dim", 128)),
            attribute_embedding_dim=int(model_cfg.get("attribute_embedding_dim", 32)),
            max_seq_len=int(model_cfg.get("max_seq_len", feature_spec.max_seq_len)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            position=str(model_cfg.get("position", "learned")),
            time_encoder=str(model_cfg.get("time_encoder", "none")),
            fourier_bands=tuple(model_cfg.get("fourier_bands", (1, 2, 4, 8))),
            time2vec_k=int(model_cfg.get("time2vec_k", 8)),
            role_dim=int(model_cfg.get("role_dim", 0)),
            role_layers=int(model_cfg.get("role_layers", 2)),
            id_dropout=float(model_cfg.get("id_dropout", 0.0)),
        )
