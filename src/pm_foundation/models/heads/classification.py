"""Generic classification head (trace- or event-level)."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Linear (optionally one hidden layer) classifier over a ``d_model`` input.

    Operates on the last dimension, so it applies to pooled trace embeddings
    ``(B, in_dim)`` or per-event states ``(B, L, in_dim)`` alike.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., in_dim) -> (..., n_classes)
        out: torch.Tensor = self.net(x)
        return out
