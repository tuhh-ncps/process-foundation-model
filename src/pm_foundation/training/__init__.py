"""Training entrypoints: role-encoder pretraining (Phase 1a) and backbone pretraining (Phase 1b)."""

from __future__ import annotations

from pm_foundation.training.ar_pretrain import pretrain_autoregressive

__all__ = ["pretrain_autoregressive"]
