"""Remaining-time prediction head (prefix-level, scalar regression).

At every event position the head predicts the time remaining until case
completion. Targets are typically standardized/log-scaled by the data layer; the
loss is a masked MAE over non-padded positions.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection

from pm_foundation.evaluation.metrics import regression_metrics
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.tasks.base import TaskHead


class RemainingTimeHead(TaskHead):
    """Predicts remaining time from each event state. Loss: masked MAE."""

    target_key = "remaining_time"

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.regressor = RegressionHead(d_model, out_dim=1, hidden_dim=hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor = self.regressor(backbone_out.event_states).squeeze(-1)  # (B, L)
        return preds

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = ~padding_mask
        return F.l1_loss(outputs[valid], targets[valid])

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return regression_metrics(prefix=prefix)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        valid = ~padding_mask
        if valid.any():
            # Report MAE in DAYS (the PPM convention). Preds/targets are log1p(seconds);
            # invert to seconds and rescale. The loss above stays on the log scale (stable
            # training, outlier-robust); only the reported/monitored metric is in days.
            def _days(z: torch.Tensor) -> torch.Tensor:
                return torch.expm1(z.clamp(0.0, 25.0)) / 86400.0

            metrics.update(_days(outputs[valid]), _days(targets[valid]))
