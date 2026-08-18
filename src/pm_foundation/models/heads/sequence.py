"""Generic per-event (sequence labeling) head."""

from __future__ import annotations

import torch
from torch import nn


class SequenceHead(nn.Module):
    """Applies a shared linear projection to every event state.

    Useful for per-event predictions (e.g. event-level anomaly scores) where each
    position emits its own output of size ``out_dim``.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self, event_states: torch.Tensor
    ) -> torch.Tensor:  # (B, L, in_dim) -> (B, L, out_dim)
        out: torch.Tensor = self.proj(event_states)
        return out
