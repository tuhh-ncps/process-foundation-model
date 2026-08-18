"""The reusable trace backbone: embeddings + encoder.

This is the unit of transfer — pretrained once via DINO, then frozen or finetuned
under task heads. Checkpoints store the (teacher) backbone plus its FeatureSpec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from pm_foundation.models.embeddings import EventEmbedding
from pm_foundation.models.encoder import EncoderOutput, TraceEncoder

if TYPE_CHECKING:
    from pm_foundation.data.preprocessing import FeatureSpec


class TraceBackbone(nn.Module):
    """Composes :class:`EventEmbedding` and :class:`TraceEncoder`.

    Accepts the batch tensors produced by the data layer and returns an
    :class:`EncoderOutput` (pooled trace embedding + per-event states).
    """

    def __init__(self, embedding: EventEmbedding, encoder: TraceEncoder) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder

    def forward(
        self,
        activity_ids: torch.Tensor,  # (B, L)
        time_features: torch.Tensor,  # (B, L, T)
        categorical_ids: torch.Tensor,  # (B, L, n_cat)
        numeric_features: torch.Tensor,  # (B, L, n_num)
        padding_mask: torch.Tensor,  # (B, L) bool, True where padded
        role_ids: torch.Tensor | None = None,  # (B, L) current-catalogue ids (role channel)
    ) -> EncoderOutput:
        event_vectors = self.embedding(
            activity_ids, time_features, categorical_ids, numeric_features, role_ids=role_ids
        )
        # CLS (position 0) is never padded; prepend a False column to the mask.
        cls_mask = torch.zeros(
            padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
        )
        extended_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, L+1)
        # time_features[..., 1] is z_since (normalized log elapsed-since-start); the encoder uses
        # it for the V5 temporal attention bias (ignored otherwise).
        out: EncoderOutput = self.encoder(
            event_vectors, padding_mask=extended_mask, elapsed_z=time_features[..., 1]
        )
        return out

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> EncoderOutput:
        """Convenience: run :meth:`forward` on a collated batch dict."""
        out: EncoderOutput = self(
            batch["activity_ids"],
            batch["time_features"],
            batch["categorical_ids"],
            batch["numeric_features"],
            batch["padding_mask"],
            role_ids=batch.get("role_ids"),
        )
        return out

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any], feature_spec: FeatureSpec) -> TraceBackbone:
        """Build a backbone from a model config dict and a fitted feature spec."""
        embedding = EventEmbedding.from_config(model_cfg, feature_spec)
        encoder = TraceEncoder(
            d_model=int(model_cfg.get("d_model", 256)),
            n_layers=int(model_cfg.get("n_layers", 6)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            ffn_dim=int(model_cfg.get("ffn_dim", 1024)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            causal=bool(model_cfg.get("causal", False)),
            position=str(model_cfg.get("position", "learned")),
            temporal_bias=bool(model_cfg.get("temporal_bias", False)),
            temporal_bias_buckets=int(model_cfg.get("temporal_bias_buckets", 32)),
            temporal_bias_max_log=float(model_cfg.get("temporal_bias_max_log", 14.0)),
            temporal_bias_gated=bool(model_cfg.get("temporal_bias_gated", False)),
            log_since_mean=float(feature_spec.time_mean.get("log_since", 0.0)),
            log_since_std=float(feature_spec.time_std.get("log_since", 1.0)),
        )
        return cls(embedding, encoder)
