"""DINO loss: centered, temperature-sharpened cross-entropy between views.

See ``docs/ssl_dino.md`` §4. The teacher distribution is **centered** (running mean
subtracted) and **sharpened** (low temperature); the student is softened (higher
temperature). The loss is the average cross-entropy over all (teacher-global,
student-other) view pairs, excluding the matching same-view pair.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DinoLoss(nn.Module):
    """Cross-entropy over teacher (global) vs student (all) view distributions."""

    center: torch.Tensor  # registered buffer (1, out_dim)

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_outputs: list[torch.Tensor],  # one (B, out_dim) per view (all views)
        teacher_outputs: list[torch.Tensor],  # one (B, out_dim) per global view
    ) -> torch.Tensor:
        student_logp = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs]
        teacher_p = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_outputs
        ]

        total = torch.zeros((), device=student_outputs[0].device)
        n_terms = 0
        for ti, t in enumerate(teacher_p):
            for si, s in enumerate(student_logp):
                # Skip the matching crop: teacher global view ti aligns with
                # student global view si (global views are first in the student list).
                if ti == si:
                    continue
                total = total + -(t * s).sum(dim=-1).mean()
                n_terms += 1

        if n_terms == 0:
            raise ValueError(
                "DINO loss needs at least one cross-view pair; use n_global >= 2 "
                "or at least one local view."
            )
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_outputs: torch.Tensor) -> None:
        """EMA update of the centering buffer from a batch of teacher outputs.

        ``teacher_outputs`` is the concatenation of all teacher view outputs,
        shape ``(N, out_dim)``.
        """
        batch_center = teacher_outputs.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
