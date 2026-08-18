"""Generic scalar regression head (e.g. remaining time, anomaly score)."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """Maps a ``d_model`` input to one (or more) continuous outputs.

    Operates on the last dimension, so it applies to pooled trace embeddings
    ``(B, in_dim)`` or per-event states ``(B, L, in_dim)`` alike.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., in_dim) -> (..., out_dim)
        out: torch.Tensor = self.net(x)
        return out
