"""Downstream task heads and the multi-task LightningModule.

Each task exposes a :class:`TaskHead` (forward + loss + metrics) attachable to the
shared backbone. See ``docs/heads.md``.
"""

from __future__ import annotations

from pm_foundation.tasks.anomaly import AnomalyHead
from pm_foundation.tasks.base import TaskHead
from pm_foundation.tasks.multitask_module import MultiTaskLitModule, build_heads
from pm_foundation.tasks.next_activity import NextActivityHead
from pm_foundation.tasks.next_k_activities import NextKActivitiesHead
from pm_foundation.tasks.next_time import NextTimeHead
from pm_foundation.tasks.outcome import OutcomeHead
from pm_foundation.tasks.remaining_time import RemainingTimeHead
from pm_foundation.tasks.suffix import FutureActivitySetHead, RemainingCountHead

__all__ = [
    "AnomalyHead",
    "FutureActivitySetHead",
    "MultiTaskLitModule",
    "NextActivityHead",
    "NextKActivitiesHead",
    "NextTimeHead",
    "OutcomeHead",
    "RemainingCountHead",
    "RemainingTimeHead",
    "TaskHead",
    "build_heads",
]
