"""Experiment provenance and output artifacts.

Every backbone pretraining run and every label-efficiency evaluation is recorded as
a *run* under an output root (default ``outputs/``) with a JSON manifest, so results
are traceable both ways: an evaluation manifest names the backbone run(s) it used,
and each backbone directory accrues back-links to the evaluations that consumed it.
See ``docs/experiments.md`` for the on-disk layout.
"""

from __future__ import annotations

from pm_foundation.experiments.curves import (
    plot_label_efficiency,
    plot_learning_curve,
    write_label_efficiency,
    write_learning_curve,
)
from pm_foundation.experiments.provenance import RunContext, RunManifest, RunRegistry

__all__ = [
    "RunContext",
    "RunManifest",
    "RunRegistry",
    "plot_label_efficiency",
    "plot_learning_curve",
    "write_label_efficiency",
    "write_learning_curve",
]
