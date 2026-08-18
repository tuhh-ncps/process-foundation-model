"""Masked-event self-supervised pretraining (BERT-style MLM for traces).

A granularity-matched alternative to DINO: instead of a trace-level invariance
objective, the model predicts **masked activities from their context**, which keeps
and sharpens the local/positional signal that per-event tasks (next-activity,
remaining-time) depend on. Reuses the same :class:`TraceBackbone`.
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from pm_foundation.models.foundation_model import TraceBackbone

IGNORE_INDEX = -100


class MaskedEventLitModule(L.LightningModule):
    """Pretrains a backbone by predicting masked activity ids from context."""

    backbone: TraceBackbone

    def __init__(
        self,
        backbone: TraceBackbone,
        n_activities: int,
        mask_id: int,
        mask_prob: float = 0.15,
        optimizer_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embedding.d_model, n_activities)
        self.n_activities = n_activities
        self.mask_id = mask_id
        self.mask_prob = mask_prob
        self.optimizer_cfg = optimizer_cfg or {}

    @staticmethod
    def apply_masking(
        activity_ids: torch.Tensor, padding_mask: torch.Tensor, mask_prob: float, mask_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_ids, targets)``; targets are ``IGNORE_INDEX`` except at
        masked (real, selected) positions, where they hold the original activity."""
        selected = (torch.rand(activity_ids.shape, device=activity_ids.device) < mask_prob) & (
            ~padding_mask
        )
        masked_ids = activity_ids.clone()
        masked_ids[selected] = mask_id
        targets = torch.full_like(activity_ids, IGNORE_INDEX)
        targets[selected] = activity_ids[selected]
        return masked_ids, targets

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        padding_mask = batch["padding_mask"]
        masked_ids, targets = self.apply_masking(
            batch["activity_ids"], padding_mask, self.mask_prob, self.mask_id
        )
        out = self.backbone(
            masked_ids,
            batch["time_features"],
            batch["categorical_ids"],
            batch["numeric_features"],
            padding_mask,
        )
        logits = self.head(out.event_states)  # (B, L, n_activities)
        loss = F.cross_entropy(
            logits.reshape(-1, self.n_activities), targets.reshape(-1), ignore_index=IGNORE_INDEX
        )
        self.log("train/mlm_loss", loss, prog_bar=True, batch_size=padding_mask.shape[0])
        with torch.no_grad():
            scored = targets != IGNORE_INDEX
            if scored.any():
                acc = (logits.argmax(-1)[scored] == targets[scored]).float().mean()
                self.log("train/mlm_acc", acc, batch_size=padding_mask.shape[0])
        return loss

    def configure_optimizers(self) -> Any:
        cfg = self.optimizer_cfg
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(cfg.get("lr", 5e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs = self.trainer.max_epochs or 1
        warmup = min(
            int(cfg.get("warmup_epochs", 0) * total_steps / max(1, max_epochs)),
            max(0, total_steps - 1),
        )

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_masked_event_module(
    model_cfg: dict[str, Any], mlm_cfg: dict[str, Any], feature_spec: Any
) -> MaskedEventLitModule:
    """Assemble a :class:`MaskedEventLitModule` from config + a fitted feature spec."""
    backbone = TraceBackbone.from_config(model_cfg, feature_spec)
    return MaskedEventLitModule(
        backbone,
        n_activities=feature_spec.n_activities,
        mask_id=feature_spec.activity_vocab.mask_id,
        mask_prob=float(mlm_cfg.get("mask_prob", 0.15)),
        optimizer_cfg=mlm_cfg.get("optimizer"),
    )
