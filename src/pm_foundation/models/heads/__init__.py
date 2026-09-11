"""Reusable head primitives shared by the pretraining objectives and downstream tasks."""

from __future__ import annotations

from pm_foundation.models.heads.classification import ClassificationHead
from pm_foundation.models.heads.regression import RegressionHead

__all__ = ["ClassificationHead", "RegressionHead"]
