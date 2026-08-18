"""Rotary position embedding (RoPE).

RoPE encodes position by *rotating* the query/key vectors inside attention by an angle
proportional to the token's position, so relative position falls out of the Q·K dot product.
Unlike a learned absolute position table it needs **no parameters and no maximum length** —
rotations are computed on the fly for any position — which is what lets the trace encoder run
on sequences of arbitrary length (and extrapolate beyond the lengths seen in training).
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin rotation tables for a given (head) dimension on demand."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(seq_len, head_dim)`` for positions ``0..seq_len-1``."""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (L, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: ``[x1, x2] -> [-x2, x1]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors ``(B, H, L, head_dim)``; ``cos``/``sin`` are ``(L, head_dim)``."""
    cos = cos[None, None]  # (1, 1, L, head_dim)
    sin = sin[None, None]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
