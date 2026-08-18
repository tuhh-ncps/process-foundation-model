"""Reproducible evaluation reports and a small benchmark harness.

``run_benchmark`` trains the configured downstream task(s) under several variants
(e.g. frozen probe vs. full finetune, random vs. pretrained backbone) by reusing
``finetune``, and assembles a comparative report that can be saved to JSON and
rendered as a console table.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# A report maps a variant name to its flat {metric: value} dict.
Report = dict[str, dict[str, float]]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_benchmark(config: dict[str, Any], variants: list[dict[str, Any]] | None = None) -> Report:
    """Run each variant via ``finetune`` and collect test metrics into a report.

    Each variant is ``{"name", "freeze": bool, "checkpoint": str | None}`` and is
    deep-merged onto ``config`` before training. Defaults to a probe vs. finetune
    comparison of the given (possibly pretrained) backbone.
    """
    from pm_foundation.training.finetune import finetune  # local import avoids cycle

    if variants is None:
        variants = [
            {"name": "probe", "freeze": True},
            {"name": "finetune", "freeze": False},
        ]

    report: Report = {}
    for variant in variants:
        override: dict[str, Any] = {"task": {"backbone": {"freeze": variant["freeze"]}}}
        if variant.get("checkpoint") is not None:
            override["checkpoint"] = variant["checkpoint"]
        cfg = _deep_merge(config, override)
        results = finetune(cfg)
        report[variant["name"]] = {k: float(v) for k, v in results[0].items()}
    return report


def save_report(report: Report, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_report(path: str | Path) -> Report:
    data: Report = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def render_report(report: Report, console: Console | None = None) -> Table:
    """Render a report as a metric-by-variant table and print it."""
    variants = list(report)
    metrics = sorted({m for v in report.values() for m in v})

    table = Table(title="Evaluation report")
    table.add_column("metric", style="bold")
    for variant in variants:
        table.add_column(variant, justify="right")

    for metric in metrics:
        row = [metric]
        for variant in variants:
            value = report[variant].get(metric)
            row.append(f"{value:.4f}" if value is not None else "-")
        table.add_row(*row)

    (console or Console()).print(table)
    return table
