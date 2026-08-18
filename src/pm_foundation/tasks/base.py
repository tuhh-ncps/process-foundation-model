"""Common task-head interface (uniform contract for the multi-task module).

Every head maps the backbone's :class:`EncoderOutput` to task predictions and
defines its own loss and metrics. The ``MultiTaskLitModule`` (M7) reads each head's
``target_key`` from the batch and calls ``loss`` / ``update_metrics`` uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torchmetrics import MetricCollection

from pm_foundation.models.encoder import EncoderOutput


class TaskHead(nn.Module, ABC):
    """Maps backbone outputs to task predictions, with loss and metrics.

    Subclasses read whichever part of :class:`EncoderOutput` they need (pooled
    trace embedding and/or per-event states) and set ``target_key`` to the batch
    key holding their supervision target.
    """

    #: Batch key holding this task's target tensor.
    target_key: str = "target"

    @abstractmethod
    def forward(self, backbone_out: EncoderOutput, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return task predictions (logits / scores)."""
        raise NotImplementedError

    @abstractmethod
    def loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Scalar training loss for a batch."""
        raise NotImplementedError

    @abstractmethod
    def build_metrics(self, prefix: str = "") -> MetricCollection:
        """A fresh metric collection for this task (the caller owns it per split)."""
        raise NotImplementedError

    @abstractmethod
    def update_metrics(
        self,
        metrics: MetricCollection,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        """Flatten/mask predictions as needed and update ``metrics``."""
        raise NotImplementedError
