"""Reusable head primitives shared by SSL and downstream tasks."""

from __future__ import annotations

from pm_foundation.models.heads.classification import ClassificationHead
from pm_foundation.models.heads.projection import DinoProjectionHead
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.models.heads.sequence import SequenceHead

__all__ = [
    "ClassificationHead",
    "DinoProjectionHead",
    "RegressionHead",
    "SequenceHead",
]
