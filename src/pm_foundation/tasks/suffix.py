"""Suffix-prediction event-level heads: remaining event count, future activity set.

Both are per-event tasks read off the causal states ``h_1..h_L`` and, like
:class:`NextKActivitiesHead`, DERIVE their targets from information already in the batch
(``padding_mask`` / the eval-vocab ``next_activity`` target), so they need no new dataset or
collate plumbing. ``target_key`` is set to ``"next_activity"`` only to satisfy the multi-task
module's target lookup; the real target is computed inside each head.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection

from pm_foundation.evaluation.metrics import multilabel_metrics, regression_metrics
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.classification import ClassificationHead
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.tasks.base import TaskHead

IGNORE_INDEX = -100


class RemainingCountHead(TaskHead):
    """Predict the number of events remaining until case end, from each event state. Loss: MAE.

    Target at position ``i`` is ``(#real events - 1) - i`` (0 at the last event), derived from the
    padding mask — a control-flow analogue of remaining-time.
    """

    target_key = "next_activity"  # plumbing only; the target comes from padding_mask

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.regressor = RegressionHead(d_model, out_dim=1, hidden_dim=hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor = self.regressor(backbone_out.event_states).squeeze(-1)  # (B, L)
        return preds

    @staticmethod
    def _targets(padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ~padding_mask  # (B, L) real events
        n = valid.sum(dim=1, keepdim=True)  # (B, 1) events per trace
        pos = torch.arange(padding_mask.size(1), device=padding_mask.device).unsqueeze(0)  # (1, L)
        remaining = (n - 1 - pos).clamp(min=0).to(torch.float32)  # (B, L) events after position i
        return remaining, valid

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        remaining, valid = self._targets(padding_mask)
        return F.l1_loss(outputs[valid], remaining[valid])

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return regression_metrics(prefix=prefix)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        remaining, valid = self._targets(padding_mask)
        if valid.any():
            metrics.update(outputs[valid], remaining[valid])


class FutureActivitySetHead(TaskHead):
    """Predict the SET of activities occurring in the suffix (multi-label) from each event state.

    Target at position ``i`` is the multi-hot of ``{activity_{i+1}, .., activity_L}`` in the eval
    vocabulary — derived from the ``next_activity`` target sequence by a reverse cumulative OR
    (``next_activity[i]`` is the activity at ``i+1``). Loss: BCE-with-logits over real positions.
    """

    target_key = "next_activity"

    def __init__(self, d_model: int, n_activities: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.n_activities = n_activities
        self.classifier = ClassificationHead(d_model, n_activities, hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(backbone_out.event_states)  # (B, L, n_activities)
        return logits

    def _future_multihot(self, next_activity: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` eval-vocab next-activity ids -> ``(B, L, n)`` suffix multi-hot."""
        valid = (next_activity != IGNORE_INDEX).unsqueeze(-1)  # (B, L, 1)
        onehot = F.one_hot(next_activity.clamp(min=0), self.n_activities)  # (B, L, n)
        onehot = (onehot * valid).to(torch.float32)
        # reverse cumulative max == cumulative OR from the end: future[i] = OR of onehot[i:]
        future = torch.flip(torch.cummax(torch.flip(onehot, dims=[1]), dim=1).values, dims=[1])
        return future

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        future = self._future_multihot(targets)
        valid = ~padding_mask
        return F.binary_cross_entropy_with_logits(outputs[valid], future[valid])

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return multilabel_metrics(self.n_activities, prefix=prefix)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        future = self._future_multihot(targets)
        valid = ~padding_mask
        if valid.any():
            metrics.update(outputs[valid], future[valid].to(torch.long))
