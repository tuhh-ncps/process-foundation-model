"""Transformer trace encoder.

Consumes event vectors (with a CLS token at position 0) and returns per-event
states plus a pooled trace embedding. This is the architectural core of the backbone.

Two modes:
- **bidirectional** (default): each event attends to the whole trace; the pooled
  trace embedding is the CLS state. Ideal for understanding a finished trace, but
  the per-event states "see the future" — so next-activity must be evaluated on
  prefixes to avoid leakage.
- **causal** (``causal=True``): a triangular mask lets each event attend only to
  itself and earlier events, so per-event states are leak-free by construction
  (great for next-activity in a single full-trace pass). Under causal masking the
  front CLS sees nothing, so the trace embedding is pooled from the **last real
  event** (which has attended to the whole prefix).

Position handling is selectable (``position``): ``"learned"`` (default) uses the
absolute position table in :class:`EventEmbedding` and the stock ``nn.TransformerEncoder``;
``"rope"`` uses a rotary-position encoder (below) that needs no position table and no
maximum length, so traces of arbitrary length are handled (see ``models/rope.py``, docs §18).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pm_foundation.models.rope import RotaryEmbedding, apply_rotary


class _RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding on Q/K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, head_dim)
        cos, sin = self.rope(seq_len, x.device, q.dtype)
        q, k = apply_rotary(q, k, cos, sin)
        # MPS has no attention-dropout SDPA kernel (raises NotImplementedError); disable dropout
        # there only. CPU/CUDA keep the configured dropout, so cluster training is unchanged.
        dropout_p = self.dropout if self.training else 0.0
        if dropout_p and q.device.type == "mps":
            dropout_p = 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(bsz, seq_len, d_model)
        proj: torch.Tensor = self.proj(out)
        return proj


class _RoPEEncoderLayer(nn.Module):
    """Pre-norm transformer layer using rotary-position self-attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RoPESelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class _RoPEEncoder(nn.Module):
    """Stack of RoPE pre-norm layers + final norm (drop-in for ``nn.TransformerEncoder``)."""

    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_RoPEEncoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        normed: torch.Tensor = self.norm(x)
        return normed


class TemporalAttentionBias(nn.Module):
    """V5: learnable per-head additive attention bias over the bucketed log real-time gap
    ``log1p(|t_i - t_j|)`` — the T5 relative-position-bias pattern, but on **real time** instead
    of index position (RoPE already covers relative index). Bias table inits to zero, so at init
    it's a no-op (== plain RoPE) and only departs if temporal distance helps.

    ``times`` are per-token elapsed seconds (CLS at index 0 has t=0). Buckets are fixed log-spaced
    edges over ``[0, max_log_gap]``; the diagonal (gap 0) falls in bucket 0."""

    def __init__(
        self, n_heads: int, n_buckets: int = 32, max_log_gap: float = 14.0, gated: bool = False
    ) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.gated = gated
        # n_buckets-1 interior edges -> bucketize returns indices in [0, n_buckets-1].
        self.register_buffer("boundaries", torch.linspace(0.0, max_log_gap, n_buckets - 1))
        if gated:
            # Explicit per-head gate: contribution = tanh(a_h) * B_h[bucket], a_h init 0 -> α=0 at
            # init (no-op). B init NON-zero so a_h has a gradient (∂(αB)/∂a ∝ B); the gate opens
            # first, then B refines. Control-flow-first, temporal bias enters gently.
            self.bias = nn.Parameter(torch.randn(n_heads, n_buckets) * 0.02)
            self.alpha = nn.Parameter(torch.zeros(n_heads))
        else:
            # Implicit gate: B init zero -> no-op at init, learns the bias directly.
            self.bias = nn.Parameter(torch.zeros(n_heads, n_buckets))

    def forward(self, times: torch.Tensor) -> torch.Tensor:  # (B, L) seconds -> (B, n_heads, L, L)
        gap = (times[:, :, None] - times[:, None, :]).abs()  # (B, L, L) pairwise |t_i - t_j|
        idx = torch.bucketize(torch.log1p(gap), self.boundaries)  # (B, L, L) in [0, n_buckets-1]
        bias = self.bias[:, idx].permute(1, 0, 2, 3)  # (n_heads,B,L,L) -> (B, n_heads, L, L)
        if self.gated:
            bias = torch.tanh(self.alpha)[None, :, None, None] * bias  # α_h = tanh(a_h)
        return bias


@dataclass
class EncoderOutput:
    """Backbone outputs shared by SSL and all downstream heads."""

    trace_embedding: torch.Tensor  # (B, d_model) — pooled trace summary
    event_states: torch.Tensor  # (B, L, d_model) — per-event states (CLS excluded)


class TraceEncoder(nn.Module):
    """A pre-norm (norm-first) transformer encoder over event sequences."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
        position: str = "learned",
        temporal_bias: bool = False,
        temporal_bias_buckets: int = 32,
        temporal_bias_max_log: float = 14.0,
        temporal_bias_gated: bool = False,
        log_since_mean: float = 0.0,
        log_since_std: float = 1.0,
    ) -> None:
        super().__init__()
        if position not in ("learned", "rope"):
            raise ValueError(f"position must be 'learned' or 'rope', got {position!r}")
        self.causal = causal
        self.position = position
        # V5 temporal attention bias only applies to the RoPE path (manual attention scores).
        self.temporal_bias = bool(temporal_bias) and position == "rope"
        self._log_since_mean = log_since_mean
        self._log_since_std = log_since_std
        if self.temporal_bias:
            self.tbias = TemporalAttentionBias(
                n_heads, temporal_bias_buckets, temporal_bias_max_log, gated=temporal_bias_gated
            )
        if position == "rope":
            # No position table, no max length — handles arbitrary sequence lengths.
            self.encoder: nn.Module = _RoPEEncoder(d_model, n_layers, n_heads, ffn_dim, dropout)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=n_layers,
                norm=nn.LayerNorm(d_model),
                # The nested-tensor fast path is incompatible with norm_first=True.
                enable_nested_tensor=False,
            )

    def _rope_attn_mask(
        self, seq_len: int, padding_mask: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        """Boolean attend-mask ``(B, 1, L, L)`` (True = attend) for the RoPE encoder.

        Masks padded keys and, when causal, future keys. CLS (index 0) is never padded, so
        every query row keeps at least one valid key — no fully-masked rows / NaNs.
        """
        bsz = 1 if padding_mask is None else padding_mask.shape[0]
        if padding_mask is None:
            key_valid = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
        else:
            key_valid = ~padding_mask
        allowed = key_valid[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
        if self.causal:
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            allowed = allowed & causal
        return allowed

    def forward(
        self,
        event_vectors: torch.Tensor,  # (B, L+1, d_model), incl. CLS at index 0
        padding_mask: torch.Tensor | None = None,  # (B, L+1) bool, True where padded
        elapsed_z: torch.Tensor | None = None,  # (B, L) normalized log elapsed (z_since), for V5
    ) -> EncoderOutput:
        if self.position == "rope":
            allowed = self._rope_attn_mask(
                event_vectors.shape[1], padding_mask, event_vectors.device
            )  # (B, 1, L+1, L+1) bool
            if self.temporal_bias and elapsed_z is not None:
                # Recover real elapsed seconds from the normalized z_since column and prepend the
                # CLS token at t=0; build a per-head float bias, -inf on disallowed (pad/future).
                sec = torch.expm1(elapsed_z * self._log_since_std + self._log_since_mean)  # (B, L)
                cls0 = torch.zeros(sec.shape[0], 1, device=sec.device, dtype=sec.dtype)
                bias = self.tbias(torch.cat([cls0, sec], dim=1))  # (B, n_heads, L+1, L+1)
                attn_mask: torch.Tensor = bias.masked_fill(~allowed, float("-inf"))
            else:
                attn_mask = allowed
            hidden = self.encoder(event_vectors, attn_mask)
        else:
            causal_mask = None
            if self.causal:
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    event_vectors.shape[1], device=event_vectors.device
                )
            hidden = self.encoder(
                event_vectors, mask=causal_mask, src_key_padding_mask=padding_mask
            )
        event_states = hidden[:, 1:]  # the L event positions

        if self.causal:
            trace_embedding = self._last_event_state(event_states, padding_mask)
        else:
            trace_embedding = hidden[:, 0]  # CLS summary

        return EncoderOutput(trace_embedding=trace_embedding, event_states=event_states)

    @staticmethod
    def _last_event_state(
        event_states: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Pool the last real event's state (causal summary)."""
        bsz = event_states.shape[0]
        if padding_mask is None:
            last_idx = torch.full((bsz,), event_states.shape[1] - 1, device=event_states.device)
        else:
            # padding_mask is (B, L+1) incl. CLS at index 0; events are [:, 1:].
            lengths = (~padding_mask[:, 1:]).sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
        return event_states[torch.arange(bsz, device=event_states.device), last_idx]
