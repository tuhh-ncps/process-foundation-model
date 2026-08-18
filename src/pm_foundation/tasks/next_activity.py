"""Next-activity prediction head (prefix-level, multi-class).

At every event position the head predicts the *next* event's activity, which
covers all trace prefixes at once (teacher-forcing style). The target at a given
position is the following event's activity id; positions whose next event is
padding (and pad positions themselves) are ignored.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection

from pm_foundation.evaluation.metrics import classification_metrics
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.classification import ClassificationHead
from pm_foundation.tasks.base import TaskHead

IGNORE_INDEX = -100


class NextActivityHead(TaskHead):
    """Predicts the next activity from each event state. Loss: cross-entropy."""

    target_key = "next_activity"

    def __init__(self, d_model: int, n_activities: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.n_activities = n_activities
        self.classifier = ClassificationHead(d_model, n_activities, hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(backbone_out.event_states)  # (B, L, n_activities)
        return logits

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        n_classes = outputs.shape[-1]
        return F.cross_entropy(
            outputs.reshape(-1, n_classes), targets.reshape(-1), ignore_index=IGNORE_INDEX
        )

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return classification_metrics(self.n_activities, prefix=prefix)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        valid = targets != IGNORE_INDEX
        if valid.any():
            metrics.update(outputs[valid], targets[valid])

    @staticmethod
    def build_targets(activity_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Next-activity targets from a batch: ``target[b, i] = activity_ids[b, i+1]``.

        Positions whose next event is padding, and pad positions, are set to
        ``IGNORE_INDEX``. Shape ``(B, L)``.
        """
        targets = torch.full_like(activity_ids, IGNORE_INDEX)
        targets[:, :-1] = activity_ids[:, 1:]
        next_is_pad = torch.zeros_like(padding_mask)
        next_is_pad[:, :-1] = padding_mask[:, 1:]
        targets[next_is_pad] = IGNORE_INDEX
        targets[padding_mask] = IGNORE_INDEX
        return targets
