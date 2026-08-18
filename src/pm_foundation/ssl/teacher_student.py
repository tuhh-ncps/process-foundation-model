"""EMA teacher: a momentum copy of the student network.

The teacher is never updated by gradient descent; it tracks the student via an
exponential moving average and is the artifact we keep after pretraining.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class EmaTeacher(nn.Module):
    """Holds a teacher network and updates it as an EMA of a student.

    Args:
        teacher: A network architecturally identical to the student (and usually
            initialized with the student's weights).
        base_momentum: Starting ``lambda`` (e.g. 0.996), cosine-scheduled to 1.0.
    """

    teacher: nn.Module

    def __init__(self, teacher: nn.Module, base_momentum: float = 0.996) -> None:
        super().__init__()
        self.teacher = teacher
        self.base_momentum = base_momentum
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module, momentum: float | None = None) -> None:
        """``teacher = lambda*teacher + (1-lambda)*student`` for all params; copy buffers."""
        lam = self.base_momentum if momentum is None else momentum
        for tp, sp in zip(self.teacher.parameters(), student.parameters(), strict=True):
            tp.mul_(lam).add_(sp.detach(), alpha=1 - lam)
        for tb, sb in zip(self.teacher.buffers(), student.buffers(), strict=True):
            tb.copy_(sb)

    def momentum_at(self, step: int, total_steps: int) -> float:
        """Cosine schedule for ``lambda`` from ``base_momentum`` to 1.0."""
        if total_steps <= 0:
            return self.base_momentum
        progress = min(step / total_steps, 1.0)
        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * progress) + 1) / 2
