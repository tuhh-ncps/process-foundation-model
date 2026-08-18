"""Warm-start downstream heads from the pretrained AR pretext heads.

When a downstream task *is* one of the pretraining pretexts, the foundation model already
learned a head that solves it — so a fresh random probe wastes labels re-discovering a known
mapping. Warm-starting the downstream head from the persisted pretext head (``ar_heads.pt``)
gives near-optimal transfer at *zero* labels:

- **next-activity**: the pretext activity head *is* a next-activity classifier over the same
  vocab — copy its weights (dropping the extra END row if present).
- **next-time**: the pretext time head predicts the *standardized* log-Δt ``z``; the
  downstream target is ``log1p(Δt) = z·σ + μ`` (the same gap, unstandardized), so the
  downstream linear is an exact affine image of the pretext head: ``W ← σ·W``, ``b ← σ·b + μ``.

Only applies to a single-linear head (``hidden_dim=None``); returns ``False`` otherwise.
See docs §15.
"""

from __future__ import annotations

# The docstring uses Greek sigma/mu for the standardization affine.
# ruff: noqa: RUF002
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pm_foundation.tasks.next_activity import NextActivityHead
from pm_foundation.tasks.next_time import NextTimeHead

#: Downstream tasks that match a pretraining pretext and can be warm-started.
WARM_STARTABLE = ("next_activity", "next_time")


def load_pretext_heads(run_dir: str | Path) -> dict[str, Any]:
    """Load a backbone run's persisted pretext heads (``ar_heads.pt``)."""
    heads: dict[str, Any] = torch.load(Path(run_dir) / "ar_heads.pt", map_location="cpu")
    return heads


def _final_linear(module: nn.Module) -> nn.Linear | None:
    return module if isinstance(module, nn.Linear) else None


def warm_start_next_time(head: NextTimeHead, run_dir: str | Path) -> bool:
    """Affine-map the pretext time head into ``head``. Returns success."""
    lin = _final_linear(head.regressor.net)
    if lin is None:
        return False
    heads = load_pretext_heads(run_dir)
    # The pretext time head may be a bare Linear (keys weight/bias) or a RegressionHead
    # (net.weight/bias, or net.0/net.3 for an MLP). Take its FINAL linear's row 0 (= mu).
    sd = heads["time_head"]
    wk = sorted(k for k in sd if k.endswith("weight"))[-1]
    bk = wk[: -len("weight")] + "bias"
    w = sd[wk][:1]  # (1, d); row 0 = mu if the pretext was log-normal
    b = sd[bk][:1]
    sigma, mu = float(heads["log_delta_std"]), float(heads["log_delta_mean"])
    with torch.no_grad():
        lin.weight.copy_(sigma * w)
        lin.bias.copy_(sigma * b + mu)
    return True


def warm_start_next_activity(head: NextActivityHead, run_dir: str | Path) -> bool:
    """Copy the pretext activity head into ``head`` (dropping END). Returns success."""
    lin = _final_linear(head.classifier.net)
    if lin is None:
        return False
    heads = load_pretext_heads(run_dir)
    n = head.n_activities
    with torch.no_grad():
        lin.weight.copy_(heads["activity_head"]["weight"][:n])  # drop END row if present
        lin.bias.copy_(heads["activity_head"]["bias"][:n])
    return True


def warm_start(head: nn.Module, task: str, run_dir: str | Path) -> bool:
    """Warm-start ``head`` for ``task`` from the pretext heads at ``run_dir``.

    Returns ``True`` if a warm-start was applied, ``False`` otherwise (unsupported task,
    non-linear head, or missing/mismatched pretext head).
    """
    try:
        if task == "next_time" and isinstance(head, NextTimeHead):
            return warm_start_next_time(head, run_dir)
        if task == "next_activity" and isinstance(head, NextActivityHead):
            return warm_start_next_activity(head, run_dir)
    except (FileNotFoundError, KeyError, RuntimeError):
        return False
    return False
