"""Feature-budget ladder, six further tasks (protocols/feature_ladder.md, amendment A2).

Reads results/feature_ladder_tasks_summary.csv (scripts/feature_ladder_tasks_analysis.py).
Writes feature_ladder_tasks.pdf / .png next to this file.

One panel per task, showing the D5 paired difference to k = 15: five thin per-log curves, the five-log mean
and its t(4) 95% band, against a zero line that marks parity with the full 15-descriptor fingerprint.

Differences rather than levels because the raw per-log levels do not share an axis -- remaining-count MAE runs
from 0.30 events on Helpdesk to 8.1 on BPI17, so within-log variation would be invisible -- while the paired
differences share each task's own unit around a common origin. This also avoids averaging the MAE tasks across
logs, which A2 flags as scale-mixing. For the three higher-is-better tasks a negative difference means the
budget is worse than the full fingerprint; for the three MAE tasks a positive difference means worse. Each
panel says which.

A1 caveat: cached evaluator, so budgets are compared with each other, not with the main result tables.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]
LOG_LABEL = {"bpi13_incidents": "BPI13", "BPI17": "BPI17", "BPI20ID": "BPI20ID",
             "helpdesk": "Helpdesk", "mimic_transfer": "MIMIC"}
LOG_COLOR = {"bpi13_incidents": "#0072b2", "BPI17": "#d55e00", "BPI20ID": "#009e73",
             "helpdesk": "#cc79a7", "mimic_transfer": "#56b4e9"}
TASKS = ["next_3_activities", "next_5_activities", "future_activity_set",
         "next_time", "remaining_time", "remaining_count"]
TITLE = {"next_3_activities": "next 3 activities", "next_5_activities": "next 5 activities",
         "future_activity_set": "future activity set", "next_time": "next event time",
         "remaining_time": "remaining time", "remaining_count": "remaining count"}
PANEL = "abcdef"

rows = list(csv.DictReader(open(os.path.join(RES, "feature_ladder_tasks_summary.csv"))))
by_task = {t: {int(r["k"]): r for r in rows if r["task"] == t} for t in TASKS}
ks = sorted(by_task[TASKS[0]])

plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6})
fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8))

for i, (task, ax) in enumerate(zip(TASKS, axes.ravel())):
    rs = by_task[task]
    higher_better = rs[0]["higher_is_better"] == "True"
    worse = "below 0 = worse than $k=15$" if higher_better else "above 0 = worse than $k=15$"

    ax.axhline(0, color="#444444", lw=1.0, zorder=2)
    for log in LOGS:                                                      # D5, per log, task's own unit
        ax.plot(ks, [float(rs[k][f"delta_{log}"]) for k in ks], color=LOG_COLOR[log], lw=1.1,
                alpha=0.75, marker="o", ms=2.6, zorder=3, label=LOG_LABEL[log] if i == 0 else None)
    lo = [float(rs[k]["delta_ci_low"]) for k in ks]
    hi = [float(rs[k]["delta_ci_high"]) for k in ks]
    ax.fill_between(ks, lo, hi, facecolor="#c4c4c4", edgecolor="none", alpha=0.55, zorder=1,
                    label="five-log mean, $t(4)$ 95% CI" if i == 0 else None)
    ax.plot(ks, [float(rs[k]["delta_mean"]) for k in ks], color="#111111", lw=2.0, zorder=4)

    sig = [k for k in ks if rs[k]["delta_excludes_zero"] == "True"]
    for k in sig:                                                          # interval clear of zero
        ax.plot(k, float(rs[k]["delta_mean"]), marker="*", ms=11, color="#d55e00",
                mec="#111111", mew=0.6, zorder=5)
    ax.set_title(f"({PANEL[i]}) {TITLE[task]} - {rs[0]['unit']}\n{worse}", fontsize=10)
    ax.set_xticks([k for k in ks if k % 3 == 0] + [15])
    ax.set_xlim(-0.6, 15.6)
    if i >= 3:
        ax.set_xlabel("number of fingerprint descriptors $k$")
    if i % 3 == 0:
        ax.set_ylabel("difference to $k=15$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], marker="*", ms=11, color="#d55e00", mec="#111111", mew=0.6, ls="none"))
labels.append("95% CI excludes 0")
fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=9.5, framealpha=0.0,
           bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(os.path.join(HERE, "feature_ladder_tasks.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "feature_ladder_tasks.png"), dpi=200, bbox_inches="tight", transparent=True)
print("wrote feature_ladder_tasks.pdf / .png")
