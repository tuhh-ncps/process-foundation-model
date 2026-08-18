"""Shared confusion-matrix helpers for the evaluation modules (label-efficiency, zero-shot).

Rows = true, cols = predicted. Vocabulary ids are mapped to matrix positions through an explicit
``class_ids`` list, so a caller can restrict the matrix to *real* activities (reserved PAD/UNK/…
tokens excluded) or to the outcome classes, in a chosen display order.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # headless without matplotlib installed
    plt = None  # type: ignore[assignment]


def build_confusion(
    pred_ids: torch.Tensor, target_ids: torch.Tensor, class_ids: torch.Tensor
) -> torch.Tensor:
    """(C, C) integer confusion matrix over ``class_ids`` (rows = true, cols = predicted).

    ``class_ids`` lists the vocabulary ids to include, in display order. Any prediction or target
    id outside that set is dropped — targets are always in-set for next-activity/outcome, so the
    guard only catches a stray out-of-set prediction.
    """
    class_ids = class_ids.to(torch.long)
    n = int(class_ids.shape[0])
    hi = int(class_ids.max().item()) if n else 0
    id2row = torch.full((hi + 1,), -1, dtype=torch.long)
    id2row[class_ids] = torch.arange(n)

    def _map(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.long)
        out = torch.full_like(x, -1)
        inb = (x >= 0) & (x <= hi)
        out[inb] = id2row[x[inb]]
        return out

    t, p = _map(target_ids), _map(pred_ids)
    keep = (t >= 0) & (p >= 0)
    t, p = t[keep], p[keep]
    cm = torch.zeros(n, n, dtype=torch.long)
    if t.numel():
        cm.view(-1).scatter_add_(0, t * n + p, torch.ones_like(t))
    return cm


def write_confusion_csv(cm: torch.Tensor, names: list[str], path: Path) -> None:
    """Write the raw integer counts as CSV (header row/col = class names)."""
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *names])
        for i, name in enumerate(names):
            w.writerow([name, *cm[i].tolist()])


def plot_confusion(cm: torch.Tensor, names: list[str], png: Path, title: str) -> None:
    """Row-normalized (per-true recall) heatmap; annotates raw counts for small matrices."""
    if plt is None:
        return
    counts = cm.float()
    row = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)  # per-true recall
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), max(5, n * 0.3)))
    im = ax.imshow(row.numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = max(4, min(9, int(360 / max(n, 1))))
    ax.set_xticklabels(names, rotation=90, fontsize=fs)
    ax.set_yticklabels(names, fontsize=fs)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    if n <= 20:  # small enough to read: annotate each non-empty cell with its raw count
        for i in range(n):
            for j in range(n):
                c = int(cm[i, j])
                if c:
                    ax.text(
                        j,
                        i,
                        str(c),
                        ha="center",
                        va="center",
                        fontsize=max(5, fs - 1),
                        color="white" if row[i, j].item() < 0.6 else "black",
                    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized (recall)")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
