"""Write (and optionally plot) learning curves and label-efficiency curves.

CSV is the source of truth and always written; PNG plots are best-effort and skipped
if matplotlib is unavailable. Learning curves are per-epoch training metrics from a
backbone run; label-efficiency curves are per-task metric-vs-#labels results from an
evaluation run.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:  # matplotlib is an optional (viz) extra; plotting degrades gracefully without it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Learning curves (backbone pretraining)
# --------------------------------------------------------------------------- #
def write_learning_curve(path: str | Path, history: list[dict[str, Any]]) -> None:
    """Write per-epoch training metrics as CSV (``epoch`` first, then metric columns).

    ``history`` is a list of per-epoch dicts (e.g. from ``LearningCurveRecorder``);
    columns are the union of keys across rows so partially-logged metrics are kept.
    """
    path = Path(path)
    if not history:
        path.write_text("epoch\n", encoding="utf-8")
        return
    columns: OrderedDict[str, None] = OrderedDict()
    columns["epoch"] = None
    for row in history:
        for key in row:
            columns.setdefault(key, None)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_learning_curve(
    csv_path: str | Path, png_path: str | Path, *, title: str | None = None
) -> bool:
    """Plot every non-``epoch`` numeric column against ``epoch``. Returns success."""
    if plt is None:
        return False
    csv_path, png_path = Path(csv_path), Path(png_path)
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return False
    epochs = [float(r["epoch"]) for r in rows]
    series = [c for c in rows[0] if c != "epoch"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in series:
        ys = [float(r[col]) if r.get(col) not in (None, "") else float("nan") for r in rows]
        ax.plot(epochs, ys, marker=".", label=col)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title or "Backbone learning curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Label-efficiency curves (multi-head evaluation)
# --------------------------------------------------------------------------- #
_LE_FIELDS = (
    "task",
    "metric",
    "higher_is_better",
    "backbone_alias",
    "backbone_run_id",
    "mode",  # "frozen" (linear probe) | "finetune" (end-to-end, incl. from-scratch)
    "n_labels",
    "n_train_samples",  # ACTUAL #training cases used (label budget capped by availability)
    "seed",
    "value",
)


def write_label_efficiency(
    path: str | Path, records: list[dict[str, Any]], *, fields: tuple[str, ...] = _LE_FIELDS
) -> None:
    """Write label-efficiency results as long-format CSV (one row per measurement)."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float], mean: float) -> float:
    """Population std over seeds (0.0 for a single seed) — used for error bands."""
    if len(values) < 2:
        return 0.0
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def aggregate_label_efficiency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-seed rows to mean-over-seeds rows, keyed by task/metric/backbone/#labels."""
    groups: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    for r in records:
        key = (r["task"], r["metric"], r["backbone_alias"], r["n_labels"])
        g = groups.setdefault(
            key,
            {
                "task": r["task"],
                "metric": r["metric"],
                "higher_is_better": r["higher_is_better"],
                "backbone_alias": r["backbone_alias"],
                "backbone_run_id": r["backbone_run_id"],
                "n_labels": r["n_labels"],
                "n_train_samples": r.get("n_train_samples"),  # constant per (task, n_labels)
                "_values": [],
            },
        )
        g["_values"].append(float(r["value"]))
    out = []
    for g in groups.values():
        vals = g.pop("_values")
        g["mean_value"] = _mean(vals)
        g["std_value"] = _std(vals, g["mean_value"])
        g["n_seeds"] = len(vals)
        out.append(g)
    return out


def write_label_efficiency_means(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write the mean-over-seeds aggregate as CSV."""
    fields = (
        "task",
        "metric",
        "higher_is_better",
        "backbone_alias",
        "backbone_run_id",
        "n_labels",
        "n_train_samples",
        "mean_value",
        "std_value",
        "n_seeds",
    )
    write_label_efficiency(path, aggregate_label_efficiency(records), fields=fields)


def _size_sort_key(label: Any) -> tuple[int, float]:
    """Order label budgets numerically, with the ``"all"`` bucket always last."""
    if str(label) == "all":
        return (1, 0.0)
    return (0, float(label))


def _fmt_count(n: float) -> str:
    """Compact tick label for a sample count: 300 -> '300', 382217 -> '382k'."""
    return f"{round(n / 1000)}k" if n >= 1000 else str(int(n))


def _draw_task(
    ax: Any, task_rows: list[dict[str, Any]], *, x_scale: str = "log", title: str | None = None
) -> None:
    """Draw one task's label-efficiency curves (one line per backbone) onto ``ax``.

    The x-axis is the ACTUAL number of training cases (``n_train_samples``), so distances are
    proportional to real sample counts — a ``log`` axis (default) keeps budgets that span orders
    of magnitude legible while making the spacing proportional to sample-size *ratios* (so
    1000->all reads far wider than 300->1000); ``linear`` gives raw-proportional spacing. Falls
    back to evenly-spaced categorical positions for older runs without ``n_train_samples``.
    """
    metric = task_rows[0]["metric"]
    higher = str(task_rows[0]["higher_is_better"]).lower() in ("true", "1")
    backbones: OrderedDict[str, None] = OrderedDict((m["backbone_alias"], None) for m in task_rows)
    has_counts = all(m.get("n_train_samples") not in (None, "") for m in task_rows)
    use_log = has_counts and x_scale == "log"

    # A log x-axis cannot place the zero-shot (0-label) budget: log(0) = -inf, which masks that
    # point and collapses the autoscale (all other budgets pile at the right edge). Map 0 to a
    # sentinel one log-step left of the smallest positive budget, so it renders at the far left
    # (labeled "0") while the rest stay proportionally spaced on the log scale.
    zero_pos: float | None = None
    if use_log:
        pos = sorted(
            {float(m["n_train_samples"]) for m in task_rows if float(m["n_train_samples"]) > 0}
        )
        if pos:
            step = pos[1] / pos[0] if len(pos) > 1 else 10.0
            zero_pos = pos[0] / step

    def _x(v: Any) -> float:
        fv = float(v)
        return zero_pos if (zero_pos is not None and fv <= 0) else fv

    for alias in backbones:
        rows = [m for m in task_rows if m["backbone_alias"] == alias]
        if has_counts:
            rows.sort(key=lambda m: float(m["n_train_samples"]))
            xs = [_x(m["n_train_samples"]) for m in rows]
        else:  # legacy: evenly-spaced categorical fallback
            order = {
                lab: i
                for i, lab in enumerate(
                    sorted({m["n_labels"] for m in task_rows}, key=_size_sort_key)
                )
            }
            rows.sort(key=lambda m: order[m["n_labels"]])
            xs = [order[m["n_labels"]] for m in rows]
        ys = [float(m["mean_value"]) for m in rows]
        es = [float(m.get("std_value", 0.0) or 0.0) for m in rows]
        (line,) = ax.plot(xs, ys, marker="o", label=alias)
        if any(e > 0 for e in es):  # ±1 std over seeds (needs >=2 seeds to show)
            ax.fill_between(
                xs,
                [y - e for y, e in zip(ys, es, strict=True)],
                [y + e for y, e in zip(ys, es, strict=True)],
                color=line.get_color(),
                alpha=0.15,
            )

    if has_counts:
        if use_log:
            ax.set_xscale("log")
        raw_ticks = sorted({float(m["n_train_samples"]) for m in task_rows})
        ax.set_xticks([_x(t) for t in raw_ticks])
        ax.set_xticklabels([_fmt_count(t) for t in raw_ticks])
        ax.minorticks_off()
        ax.set_xlabel("# training cases")
    else:
        labels = sorted({m["n_labels"] for m in task_rows}, key=_size_sort_key)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlabel("# downstream labels")
    ax.set_ylabel(f"{metric} ({'higher' if higher else 'lower'} is better)")
    if title:  # off by default (clean figures for the paper); summary grid passes the task name
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_label_efficiency(
    records: list[dict[str, Any]], out_dir: str | Path, *, prefix: str = "", x_scale: str = "log"
) -> list[Path]:
    """Plot label-efficiency curves (mean metric vs #training cases, one line per backbone).

    Writes each per-task plot as a vector ``{task}.pdf`` and a high-res ``{task}.png`` (300 dpi),
    plus a combined ``summary.{pdf,png}`` (a grid with one panel per task, all backbones overlaid).
    Per-task plots are titleless (the caption carries the description); the summary panels keep the
    task name so they remain identifiable. Returns all written paths. ``x_scale`` is ``log``
    (default) or ``linear`` — see :func:`_draw_task`.
    """
    if plt is None:
        return []
    out_dir = Path(out_dir)
    means = aggregate_label_efficiency(records)
    tasks: OrderedDict[str, None] = OrderedDict((m["task"], None) for m in means)
    rows_by_task = {t: [m for m in means if m["task"] == t] for t in tasks}
    written: list[Path] = []

    def _save(fig: Any, stem: Path) -> None:
        """Vector PDF (for the paper) + high-res PNG (for quick viewing)."""
        pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
        fig.savefig(pdf, bbox_inches="tight")  # vector -> resolution-independent
        fig.savefig(png, dpi=300, bbox_inches="tight")  # high-res raster
        written.extend((pdf, png))

    for task, task_rows in rows_by_task.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _draw_task(ax, task_rows, x_scale=x_scale)  # no title
        fig.tight_layout()
        _save(fig, out_dir / task)
        plt.close(fig)

    # Combined single image: one panel per task, every backbone overlaid (panels keep task names).
    if len(tasks) > 1:
        ncols = 2
        nrows = (len(tasks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
        flat = axes.flatten()
        for ax, (task, task_rows) in zip(flat, rows_by_task.items(), strict=False):
            _draw_task(ax, task_rows, x_scale=x_scale, title=task)
        for ax in flat[len(tasks) :]:
            ax.axis("off")  # hide unused panels
        fig.tight_layout()
        _save(fig, out_dir / f"{prefix}summary")
        plt.close(fig)
    return written
