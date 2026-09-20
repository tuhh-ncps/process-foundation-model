"""Evaluation: the label-efficiency protocol and the shared metric builders.

``run_label_efficiency`` is exposed LAZILY and deliberately. Task heads import
``evaluation.metrics`` for their metric collections, and ``label_efficiency`` imports those same
task heads - so importing this package eagerly would create a cycle
(``evaluation`` -> ``label_efficiency`` -> ``tasks`` -> ``evaluation.metrics``).
Keeping the re-export lazy means ``import pm_foundation.evaluation.metrics`` stays cheap and
cycle-free, which is what the task heads rely on.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pm_foundation.evaluation.label_efficiency import run_label_efficiency

__all__ = ["run_label_efficiency"]

_LAZY = {"run_label_efficiency": "pm_foundation.evaluation.label_efficiency"}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)
