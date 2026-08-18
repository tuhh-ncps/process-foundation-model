"""Next-K-activities head — multi-step activity prediction (prefix-level).

At every event position the head predicts the next ``K`` activities (positions
``i+1 .. i+K``) from that event's causal state, in one shot. This is a strictly
harder, longer-horizon variant of :class:`NextActivityHead` (which is ``K=1``): it
probes whether the frozen representation encodes trajectory structure several steps
out, not just the immediate successor.

The K-step targets are DERIVED from the ordinary next-activity target sequence by
shifting, so this head reuses ``target_key = "next_activity"`` and needs no new
dataset/collate plumbing: if ``na[i]`` is the activity at ``i+1`` (already encoded in
the eval vocabulary for honest cross-dataset scoring), then the step-``j`` target at
position ``i`` is simply ``na[i+j]``. Positions running off the end of the trace are
``IGNORE_INDEX`` and excluded from the loss/metric.
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


class NextKActivitiesHead(TaskHead):
    """Predicts the next ``k`` activities from each event state. Loss: mean CE over steps.

    Metric ``acc`` is micro-accuracy pooled over all (position, step) pairs — i.e. the
    mean per-step top-1 accuracy across the K horizons.
    """

    target_key = "next_activity"  # reuse the eval-vocab next-activity target; shift for K steps

    def __init__(
        self, d_model: int, n_activities: int, k: int = 5, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        self.n_activities = n_activities
        self.k = k
        self.classifier = ClassificationHead(d_model, n_activities * k, hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(backbone_out.event_states)  # (B, L, k*n)
        b, length, _ = logits.shape
        return logits.view(b, length, self.k, self.n_activities)

    @staticmethod
    def _k_targets(next_activity: torch.Tensor, k: int) -> torch.Tensor:
        """``(B, L)`` immediate-next targets -> ``(B, L, k)`` by left-shifting each step.

        Step ``j`` at position ``i`` is ``next_activity[i+j]`` (the activity ``i+1+j``);
        positions that run past the trace end are ``IGNORE_INDEX``.
        """
        b, length = next_activity.shape
        tk = next_activity.new_full((b, length, k), IGNORE_INDEX)
        for j in range(k):
            if length - j > 0:
                tk[:, : length - j, j] = next_activity[:, j:]
        return tk

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        tk = self._k_targets(targets, self.k)  # (B, L, k)
        return F.cross_entropy(
            outputs.reshape(-1, self.n_activities), tk.reshape(-1), ignore_index=IGNORE_INDEX
        )

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        return classification_metrics(self.n_activities, top_k=0, prefix=prefix)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        tk = self._k_targets(targets, self.k)
        valid = tk != IGNORE_INDEX
        if valid.any():
            metrics.update(outputs[valid], tk[valid])
