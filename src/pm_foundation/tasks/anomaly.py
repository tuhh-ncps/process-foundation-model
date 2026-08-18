"""Anomaly detection head (trace-level scoring).

The default ``supervised`` mode learns a trace-level anomaly score against binary
labels (BCE). An unsupervised ``one_class`` objective is left as future work; the
scoring module is the same, so only the loss differs.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.tasks.base import TaskHead


class AnomalyHead(TaskHead):
    """Produces a trace-level anomaly score (logit) from the trace embedding."""

    target_key = "anomaly"

    def __init__(
        self, d_model: int, hidden_dim: int | None = None, mode: str = "supervised"
    ) -> None:
        super().__init__()
        self.mode = mode
        self.scorer = RegressionHead(d_model, out_dim=1, hidden_dim=hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        score: torch.Tensor = self.scorer(backbone_out.trace_embedding).squeeze(-1)  # (B,) logit
        return score

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(outputs, targets.float())

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return MetricCollection(
            {"auroc": BinaryAUROC(), "auprc": BinaryAveragePrecision()}, prefix=prefix
        )

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        metrics.update(torch.sigmoid(outputs), targets.int())
