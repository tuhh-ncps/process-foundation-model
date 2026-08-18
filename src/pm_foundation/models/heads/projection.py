"""DINO projection head: MLP -> L2-normalize -> normalized prototypes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DinoProjectionHead(nn.Module):
    """Projects a trace embedding to ``out_dim`` prototype logits for DINO.

    Architecture (per DINO): an MLP (GELU) down to a bottleneck, an L2-normalized
    bottleneck, then a linear layer to ``out_dim`` prototypes whose weight vectors
    are unit-normalized — equivalent to a weight-normalized layer with the
    magnitude fixed to 1, which stabilizes training and avoids a deprecated API.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        if n_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.prototypes = nn.Parameter(torch.empty(out_dim, bottleneck_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) -> (B, out_dim)
        hidden: torch.Tensor = self.mlp(x)
        hidden = F.normalize(hidden, dim=-1, p=2)
        weight = F.normalize(self.prototypes, dim=-1, p=2)
        return F.linear(hidden, weight)
