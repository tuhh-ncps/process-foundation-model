"""Self-supervised learning: DINO loss, EMA teacher, and the pretraining module."""

from __future__ import annotations

from pm_foundation.ssl.autoregressive import (
    AutoregressiveLitModule,
    build_autoregressive_module,
)
from pm_foundation.ssl.dino_loss import DinoLoss
from pm_foundation.ssl.dino_module import DinoEncoder, DinoLitModule, build_dino_module
from pm_foundation.ssl.masked_event import MaskedEventLitModule, build_masked_event_module
from pm_foundation.ssl.teacher_student import EmaTeacher

# Note: a masked-time (MTM) objective was implemented and evaluated but did not help
# remaining-time under a frozen probe (random was best); it was removed from the
# library. The negative result is documented in docs/ssl_dino.md §9.

__all__ = [
    "AutoregressiveLitModule",
    "DinoEncoder",
    "DinoLitModule",
    "DinoLoss",
    "EmaTeacher",
    "MaskedEventLitModule",
    "build_autoregressive_module",
    "build_dino_module",
    "build_masked_event_module",
]
