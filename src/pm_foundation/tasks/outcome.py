"""Case-level prediction head with a SHARED pooling adapter (Template B).

Every case-level task (mortality, LOS, ICU, readmission, generic outcome) uses the SAME
adapter over the per-event states ``H = [h_1 .. h_L]``::

    H --Pool--> z --Linear--> y

so the pooling is identical across tasks and is the only thing between the frozen backbone and
the linear classifier — removing the confound of relying on whatever the backbone happened to
compress into its predefined ``trace_embedding``. Four poolings (``evaluate.pooling``):

  * ``trace``     (P0) — the backbone's own pooled ``trace_embedding`` (the *fixed-summary linear
                         probe*: does the predefined case vector already contain the target?)
  * ``mean``      (P1) — masked mean over event states, parameter-free
  * ``last``      (P2) — the last real event state, natural for causal models
  * ``max``       (P2') — masked element-wise max over event states, parameter-free
  * ``attention`` (P3) — learned additive attention over event states (the *event-state
                         aggregation probe*: is the target in the collection of event states,
                         even if not compressed into the predefined summary?)

The backbone stays frozen; only the pooling (P3 has params; P0-P2' are parameter-free) and the
linear classifier are trained — still a legitimate frozen-backbone probe. The parameter-free
poolings (mean/last/max) vs attention separate "using all event states" from "learning
task-specific event importance".
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection

from pm_foundation.evaluation.metrics import classification_metrics
from pm_foundation.models.encoder import EncoderOutput
from pm_foundation.models.heads.classification import ClassificationHead
from pm_foundation.tasks.base import TaskHead


class CasePooling(nn.Module):
    """Aggregate per-event states ``H = (B, L, d)`` into a case vector ``z = (B, d)``."""

    MODES = ("trace", "mean", "last", "max", "attention")

    def __init__(self, mode: str, d_model: int) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown pooling {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        if mode == "attention":  # additive attention: alpha_i = softmax(w^T tanh(W h_i))
            self.proj = nn.Linear(d_model, d_model)
            self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "trace":
            return out.trace_embedding  # (B, d) — the backbone's predefined case summary
        h = out.event_states  # (B, L, d)
        valid = ~padding_mask  # (B, L), True at real events (padding is right-aligned)
        if self.mode == "mean":
            m = valid.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.mode == "last":
            last = valid.sum(dim=1).clamp(min=1) - 1  # index of the last real event per row
            return h[torch.arange(h.size(0), device=h.device), last]
        if self.mode == "max":  # masked element-wise max: z_j = max_i h_ij over real events
            neg = torch.finfo(h.dtype).min
            return h.masked_fill(~valid.unsqueeze(-1), neg).max(dim=1).values
        # attention (P3)
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        return (alpha * h).sum(dim=1)


class OutcomeHead(TaskHead):
    """Case-level classifier: pool ``H`` (shared adapter) then a linear layer. CE loss."""

    target_key = "outcome"

    def __init__(
        self, d_model: int, n_classes: int, pooling: str = "trace", hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.pool = CasePooling(pooling, d_model)
        self.classifier = ClassificationHead(d_model, n_classes, hidden_dim)

    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        z = self.pool(backbone_out, padding_mask)  # (B, d)
        logits: torch.Tensor = self.classifier(z)  # (B, n_classes)
        return logits

    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(outputs, targets)

    def build_metrics(self, prefix: str = "") -> MetricCollection:
        # ranking=True adds auroc/auprc — the honest readout for imbalanced case tasks.
        return classification_metrics(self.n_classes, top_k=0, prefix=prefix, ranking=True)

    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        metrics.update(outputs, targets)
