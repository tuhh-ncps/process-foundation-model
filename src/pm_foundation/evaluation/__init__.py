"""Evaluation: metrics, representation probing, and reporting.

``probing`` and ``report`` depend on ``tasks`` (which in turn import
``evaluation.metrics``), so they are exposed lazily to avoid an import cycle:
importing ``evaluation.metrics`` must not pull in ``probing``/``report``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pm_foundation.evaluation.benchmark import load_event_log, run_task_benchmark
    from pm_foundation.evaluation.label_efficiency import run_label_efficiency
    from pm_foundation.evaluation.probing import extract_trace_embeddings, linear_probe
    from pm_foundation.evaluation.report import (
        load_report,
        render_report,
        run_benchmark,
        save_report,
    )
    from pm_foundation.evaluation.rollout import SuffixRemainingTime

__all__ = [
    "SuffixRemainingTime",
    "extract_trace_embeddings",
    "linear_probe",
    "load_event_log",
    "load_report",
    "render_report",
    "run_benchmark",
    "run_label_efficiency",
    "run_task_benchmark",
    "save_report",
]

_LAZY = {
    "SuffixRemainingTime": "pm_foundation.evaluation.rollout",
    "extract_trace_embeddings": "pm_foundation.evaluation.probing",
    "linear_probe": "pm_foundation.evaluation.probing",
    "load_event_log": "pm_foundation.evaluation.benchmark",
    "load_report": "pm_foundation.evaluation.report",
    "render_report": "pm_foundation.evaluation.report",
    "run_benchmark": "pm_foundation.evaluation.report",
    "run_label_efficiency": "pm_foundation.evaluation.label_efficiency",
    "run_task_benchmark": "pm_foundation.evaluation.benchmark",
    "save_report": "pm_foundation.evaluation.report",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)
