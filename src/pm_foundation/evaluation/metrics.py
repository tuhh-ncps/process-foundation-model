"""Shared metric builders (torchmetrics) used across tasks and evaluation."""

from __future__ import annotations

from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MultilabelAUROC,
    MultilabelF1Score,
)


def classification_metrics(
    n_classes: int, top_k: int = 5, prefix: str = "", ranking: bool = False
) -> MetricCollection:
    """Accuracy + macro-F1 (+ top-k accuracy when ``n_classes > top_k``).

    ``ranking=True`` adds threshold-free ``auroc`` / ``auprc`` (macro over classes). These are
    the right primary metric for heavily imbalanced case tasks (mortality, ICU, readmission),
    where macro-F1 collapses to the majority floor even when the model ranks well. Only enable it
    for small label spaces — a per-class AUROC over a large next-activity vocab is wasteful.
    """
    metrics: dict[str, Metric | MetricCollection] = {
        "acc": MulticlassAccuracy(num_classes=n_classes, average="micro"),
        "macro_f1": MulticlassF1Score(num_classes=n_classes, average="macro"),
    }
    if top_k and n_classes > top_k:
        metrics[f"top{top_k}_acc"] = MulticlassAccuracy(
            num_classes=n_classes, top_k=top_k, average="micro"
        )
    if ranking:
        metrics["auroc"] = MulticlassAUROC(num_classes=n_classes, average="macro")
        metrics["auprc"] = MulticlassAveragePrecision(num_classes=n_classes, average="macro")
    return MetricCollection(metrics, prefix=prefix)


def regression_metrics(prefix: str = "") -> MetricCollection:
    """MAE + RMSE for scalar regression tasks."""
    return MetricCollection(
        {"mae": MeanAbsoluteError(), "rmse": MeanSquaredError(squared=False)},
        prefix=prefix,
    )


def multilabel_metrics(n_labels: int, prefix: str = "") -> MetricCollection:
    """Micro/macro-F1 + macro-AUROC for multi-label tasks (e.g. future-activity-set).

    ``micro_f1`` pools all (position, label) decisions (robust set-overlap score); ``macro_f1``
    averages per activity; ``auroc`` is threshold-free.
    """
    return MetricCollection(
        {
            "micro_f1": MultilabelF1Score(num_labels=n_labels, average="micro"),
            "macro_f1": MultilabelF1Score(num_labels=n_labels, average="macro"),
            "auroc": MultilabelAUROC(num_labels=n_labels, average="macro"),
        },
        prefix=prefix,
    )
