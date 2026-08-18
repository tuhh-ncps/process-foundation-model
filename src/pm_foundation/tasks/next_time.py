"""Next-time prediction head (prefix-level, scalar regression).

At every event position the head predicts the time until the *next* event (the
inter-event gap), targeted as ``log1p`` seconds by the data layer. This is a
per-event *timing* task -- the immediate-gap analogue of remaining-time, and the
exact quantity the autoregressive pretext optimizes. The final event of a trace
has no successor: its target is ``NaN`` and masked out, alongside padding. Loss is
a masked MAE over valid (non-pad, non-NaN) positions.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection

from pm_foundation.evaluation.metrics import regression_metrics
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.tasks.base import TaskHead


class NextTimeHead(TaskHead):
    """Predicts the next inter-event time from each event state. Loss: masked MAE."""

    target_key = "next_time"

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.regressor = RegressionHead(d_model, out_dim=1, hidden_dim=hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor = self.regressor(backbone_out.event_states).squeeze(-1)  # (B, L)
        return preds

    @staticmethod
    def _valid(targets: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Non-pad positions that have a successor (target is not NaN)."""
        return ~padding_mask & ~torch.isnan(targets)

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = self._valid(targets, padding_mask)
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
        valid = self._valid(targets, padding_mask)
        if valid.any():
            # Report MAE in DAYS (the PPM convention). Preds/targets are log1p(seconds);
            # invert to seconds and rescale. The loss above stays on the log scale.
            # (Inter-event gaps are short: switch 86400.0 -> 3600.0 to report in hours.)
            def _days(z: torch.Tensor) -> torch.Tensor:
                return torch.expm1(z.clamp(0.0, 25.0)) / 86400.0

            metrics.update(_days(outputs[valid]), _days(targets[valid]))
