"""Fingerprint reconstruction vs feature budget k, one figure per downstream task.

Reads results/feature_ladder.json (Phase A), results/feature_ladder_summary.csv (next activity, C3/D) and
results/feature_ladder_tasks_summary.csv (the six further tasks, amendment A2). Writes
feature_ladder_recon_<task>.pdf / .png next to this file, one per task.

Left axis, pretraining logs only (identical in every figure): the frozen order's cumulative reconstruction
J(F_k)/15 and the gain Delta J/15 contributed by the descriptor added at step k. Letters mark which descriptor
enters (frozen order CHFNJBMLDOKGIEA).

Right axis, five held-out logs: that task's five-log mean with the D3 seed-wise SD (SD of the three five-log
means across evaluation seeds) as error bars. The axis is a different quantity on a different scale from J, so
where the task curve sits relative to the J line carries no meaning; only its SHAPE does. The range is set
from the task's own data and padded to ~2.5x its span: a tight fit magnifies the plateau, where the
budget-to-budget wiggles are about the size of the error bars.

The bars cover downstream-head seeds ONLY. Each budget is a single Phase-1b pretraining run (seed 0), so the
budget-to-budget wiggles also carry pretraining-seed variation that the bars do not show. D3b puts that scale
at sigma_15 = 0.0058 for next activity (the three GIN-15 replicas), which is larger than most of the bars
drawn here -- so a bar that clears its neighbour is not evidence that the two budgets differ. Next event time
is the clearest case: its swings exceed its bars, and it has no plateau to read.

MAE tasks (next event time, remaining time, remaining count) share a unit across logs -- log1p seconds, or
events -- so the five-log mean is well defined, but the per-log LEVELS differ by an order of magnitude
(remaining count: 0.30 events on Helpdesk, 8.1 on BPI17). The mean is therefore weighted towards the
largest-scale log, which is not always the log driving its shape. Each MAE panel states which log dominates
its level, computed from the data. Per-log curves are in figures/feature_ladder_tasks_plot.py.

A1 caveat: everything downstream comes from the cached evaluator, so budgets are compared with each other and
not with the main result tables.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
N = 15
CURVE = "#d55e00"
LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]
LOG_LABEL = {"bpi13_incidents": "BPI13", "BPI17": "BPI17", "BPI20ID": "BPI20ID",
             "helpdesk": "Helpdesk", "mimic_transfer": "MIMIC"}

# task key -> (figure title fragment, right-axis label, lower_is_better)
TASKS = {
    "next_activity": ("next-activity accuracy", "next-activity accuracy, five-log mean", False),
    "next_3_activities": ("next-3-activities accuracy", "next 3 activities: accuracy, five-log mean", False),
    "next_5_activities": ("next-5-activities accuracy", "next 5 activities: accuracy, five-log mean", False),
    "future_activity_set": ("future-activity-set micro-F1", "future activity set: micro-F1, five-log mean", False),
    "next_time": ("next-event-time MAE", "next event time: MAE (log1p s), five-log mean", True),
    "remaining_time": ("remaining-time MAE", "remaining time: MAE (log1p s), five-log mean", True),
    "remaining_count": ("remaining-count MAE", "remaining count: MAE (events), five-log mean", True),
}

art = json.load(open(os.path.join(RES, "feature_ladder.json")))
ks = list(range(N + 1))
J = [0.0] + [s["J_over_15"] for s in art["steps"]]                      # J(F_0) = 0 by construction
gain = [s["delta_J"] / N for s in art["steps"]]                         # bar at step k

rows = {"next_activity": {int(r["k"]): r
                          for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_summary.csv")))}}
for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_tasks_summary.csv"))):
    rows.setdefault(r["task"], {})[int(r["k"])] = r


def level_note(task: str) -> str:
    """Which log the five-log mean MAE is weighted towards, from the k = 15 per-log levels."""
    lv = {log: float(rows[task][15][f"value_{log}"]) for log in LOGS}
    top = max(lv, key=lv.get)
    return f"five-log mean is weighted towards {LOG_LABEL[top]} ({lv[top] / sum(lv.values()):.0%} of the total)"


def make(task: str) -> str:
    title_bit, ylab, lower_better = TASKS[task]
    acc = [float(rows[task][k]["mu"]) for k in ks]
    sd = [float(rows[task][k]["seedwise_sd"]) for k in ks]

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    ax.bar(ks[1:], gain, color="#9ecae1", width=0.7, zorder=1, label="gain at step $k$, $\\Delta J/15$")
    ax.plot(ks, J, "o-", color="#0072b2", lw=2.4, ms=5, zorder=4, label="frozen order, $J(F_k)/15$")
    for k in range(1, N + 1):
        ax.annotate(art["order_letters"][k - 1], (k, J[k]), textcoords="offset points", xytext=(-3, 13),
                    fontsize=8.5, color="#0072b2", zorder=5,
                    # A flat task curve must cross the rising J line, and the crossing moves with the task, so
                    # the letters carry a halo thick enough to mask a line passing behind a glyph.
                    path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
    ax.set_xticks(ks)
    ax.set_xlim(-0.4, N + 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("number of fingerprint descriptors $k$ (letters = descriptor added)")
    ax.set_ylabel("fraction of standardised fingerprint variance reconstructed")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    tx = ax.twinx()                               # different quantity and scale: compare shapes, not heights
    lab = f"{title_bit.split(':')[0]} (mean $\\pm$ SD over eval seeds)"
    tx.errorbar(ks, acc, yerr=sd, color=CURVE, lw=2.0, marker="s", ms=5, capsize=3, elinewidth=1.2,
                zorder=3, label=lab)
    span = max(acc) - min(acc)                    # padded to ~2.5x the span: a tight fit magnifies the plateau
    tx.set_ylim(min(acc) - 0.75 * span, max(acc) + 0.75 * span)
    tx.set_ylabel(ylab + ("\n(lower is better)" if lower_better else ""), color=CURVE)
    tx.tick_params(axis="y", colors=CURVE)
    tx.grid(False)

    handles = ax.get_legend_handles_labels()[0] + tx.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + tx.get_legend_handles_labels()[1]
    fig.legend(handles, labels, fontsize=9.5, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.015))      # outside: the interior has no gap that clears three entries
    ax.set_title(f"Fingerprint reconstruction and {title_bit} vs $k$", pad=22)  # room for the MAE note below
    if lower_better:                              # the mean is legitimate but unevenly weighted -- say so
        ax.text(0.0, 1.008, level_note(task), transform=ax.transAxes, fontsize=8.5, color="#666666")
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
        tx.spines[sp].set_visible(False)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    stem = os.path.join(HERE, f"feature_ladder_recon_{task}")
    fig.savefig(stem + ".pdf", bbox_inches="tight", transparent=True)
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return (f"{task:20s} range {min(acc):.4f}..{max(acc):.4f}  axis "
            f"{min(acc) - 0.75 * span:.4f}..{max(acc) + 0.75 * span:.4f}  max SD {max(sd):.4f}")


def make_combined(tasks: list[tuple[str, str, str]], name: str) -> str:
    """J(F_k)/15 and the gain bars, plus several task curves, each on its own right-hand axis.

    tasks: (task key, colour, marker); the first uses the inner right axis, later ones get offset spines.
    Every axis is scaled from its own task's data with the same ~2.5x padding as make(), so the curves are
    comparable in SHAPE only.
    """
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.bar(ks[1:], gain, color="#9ecae1", width=0.7, zorder=1, label="gain at step $k$, $\\Delta J/15$")
    ax.plot(ks, J, "o-", color="#0072b2", lw=2.4, ms=5, zorder=4, label="frozen order, $J(F_k)/15$")
    for k in range(1, N + 1):
        ax.annotate(art["order_letters"][k - 1], (k, J[k]), textcoords="offset points", xytext=(-3, 13),
                    fontsize=8.5, color="#0072b2", zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
    ax.set_xticks(ks)
    ax.set_xlim(-0.4, N + 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("number of fingerprint descriptors $k$ (letters = descriptor added)")
    ax.set_ylabel("fraction of standardised fingerprint variance reconstructed")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    notes = []
    for i, (task, colour, marker) in enumerate(tasks):
        title_bit, ylab, lower_better = TASKS[task]
        vals = [float(rows[task][k]["mu"]) for k in ks]
        sd = [float(rows[task][k]["seedwise_sd"]) for k in ks]
        tx = ax.twinx()
        if i:
            tx.spines["right"].set_position(("axes", 1.0 + 0.15 * i))
        h = tx.errorbar(ks, vals, yerr=sd, color=colour, lw=2.0, marker=marker, ms=5, capsize=3,
                        elinewidth=1.2, zorder=3)
        span = max(vals) - min(vals)
        tx.set_ylim(min(vals) - 0.75 * span, max(vals) + 0.75 * span)
        tx.set_ylabel(ylab + ("\n(lower is better)" if lower_better else ""), color=colour)
        tx.tick_params(axis="y", colors=colour)
        tx.spines["right"].set_color(colour)
        tx.spines["top"].set_visible(False)
        tx.grid(False)
        handles.append(h)
        labels.append(f"{title_bit} (mean $\\pm$ SD over eval seeds)")
        if lower_better:
            notes.append(f"{title_bit}: {level_note(task)}")

    fig.legend(handles, labels, fontsize=9.5, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    ax.set_title("Fingerprint reconstruction, " + " and ".join(TASKS[t][0] for t, _, _ in tasks) + " vs $k$",
                 pad=22)
    if notes:
        ax.text(0.0, 1.008, "; ".join(notes), transform=ax.transAxes, fontsize=8.5, color="#666666")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    stem = os.path.join(HERE, f"feature_ladder_recon_{name}")
    fig.savefig(stem + ".pdf", bbox_inches="tight", transparent=True)
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


for task in TASKS:
    print(make(task))
print(f"wrote {len(TASKS)} figures: feature_ladder_recon_<task>.pdf / .png")
print("wrote", make_combined([("next_activity", CURVE, "s"), ("remaining_time", "#009e73", "D")],
                             "next_activity_remaining_time"))
