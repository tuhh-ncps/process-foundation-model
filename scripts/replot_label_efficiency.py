#!/usr/bin/env python3
"""Self-contained label-efficiency replotter — imports NOTHING from pm_foundation.

Reads each run's ``curves.csv`` and rewrites ``summary.png`` + per-task PNGs with a correct log
x-axis: the zero-shot (0-label) budget is placed at the far left (labeled "0") instead of
breaking the log scale (``log(0) = -inf`` masks the point and collapses the rest to the right
edge). No probe is recomputed.

Deliberately depends ONLY on matplotlib + stdlib, so it is immune to a stale/baked-in
pm_foundation inside the Apptainer image (jobs bind the live repo, but ``apptainer exec ...``
by itself imports the old /opt copy). Run it in the container (which has matplotlib):

    apptainer exec pmfoundation.sif python scripts/replot_label_efficiency.py outputs/label_efficiency/*
"""

import csv
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fmt(n: float) -> str:
    n = float(n)
    return f"{round(n / 1000)}k" if n >= 1000 else str(int(n))


def _load(run_dir: Path) -> list[dict]:
    """Mean-over-seeds rows, keyed by (task, backbone_alias, n_labels)."""
    rows = list(csv.DictReader((run_dir / "curves.csv").open()))
    agg: OrderedDict = OrderedDict()
    for r in rows:
        key = (r["task"], r["backbone_alias"], r["n_labels"])
        g = agg.setdefault(
            key,
            {
                "task": r["task"],
                "metric": r["metric"],
                "higher": str(r["higher_is_better"]).lower() in ("true", "1"),
                "bb": r["backbone_alias"],
                "nt": float(r["n_train_samples"]),
                "vals": [],
            },
        )
        g["vals"].append(float(r["value"]))
    for g in agg.values():
        g["mean"] = sum(g["vals"]) / len(g["vals"])
    return list(agg.values())


def _draw(ax, rows: list[dict]) -> None:
    higher = rows[0]["higher"]
    bbs = OrderedDict((g["bb"], None) for g in rows)
    pos = sorted({g["nt"] for g in rows if g["nt"] > 0})
    # place the 0-label point one log-step left of the smallest positive budget
    zero = (pos[0] / (pos[1] / pos[0] if len(pos) > 1 else 10.0)) if pos else None

    def x(v: float) -> float:
        return zero if (zero is not None and v <= 0) else v

    for bb in bbs:
        gs = sorted((g for g in rows if g["bb"] == bb), key=lambda g: g["nt"])
        ax.plot([x(g["nt"]) for g in gs], [g["mean"] for g in gs], marker="o", label=bb)
    ax.set_xscale("log")
    ticks = sorted({g["nt"] for g in rows})
    ax.set_xticks([x(t) for t in ticks])
    ax.set_xticklabels([_fmt(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("# training cases")
    ax.set_ylabel(f"{rows[0]['metric']} ({'higher' if higher else 'lower'} is better)")
    ax.set_title(f"{rows[0]['task']}: label efficiency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def replot(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    if not (run_dir / "curves.csv").exists():
        print(f"skip {run_dir} (no curves.csv)")
        return
    means = _load(run_dir)
    tasks = list(OrderedDict((g["task"], None) for g in means))
    by = {t: [g for g in means if g["task"] == t] for t in tasks}
    n = 0
    for t in tasks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _draw(ax, by[t])
        fig.tight_layout()
        fig.savefig(run_dir / f"{t}.png", dpi=120)
        plt.close(fig)
        n += 1
    if len(tasks) > 1:
        ncols = 2
        nrows = (len(tasks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
        flat = axes.flatten()
        for ax, t in zip(flat, tasks, strict=False):
            _draw(ax, by[t])
        for ax in flat[len(tasks) :]:
            ax.axis("off")
        fig.suptitle("label efficiency — all tasks & backbones", fontsize=13)
        fig.tight_layout()
        fig.savefig(run_dir / "summary.png", dpi=120)
        plt.close(fig)
        n += 1
    print(f"{run_dir.name}: rewrote {n} PNG(s)")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: replot_label_efficiency.py <run_dir> [<run_dir> ...]")
        raise SystemExit(1)
    for a in sys.argv[1:]:
        replot(Path(a))


if __name__ == "__main__":
    main()
