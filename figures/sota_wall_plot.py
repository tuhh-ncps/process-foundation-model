"""Merged figure: SOTA comparison at full budget (accuracy tasks, normalised MAE tasks) + measured wall-clock.

Panels (a)/(b) reuse sota_agg_data.csv written by sota_agg_plot.py (mean over the five held-out logs; MAE tasks
divided by the Random-role MAE per log). Panel (c) reuses results/timing_pinned.csv (one full H200 per job,
common task set). Colour families: PFM / PFM-FT = oranges (light / dark), FM-v2 Proto / kNN = greens, SuTraN = purple.
Outputs: sota_wall.pdf/.png, sota_wall.tex.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root
# Our family = turquoise, deepening with how much of the model adapts to the new log
# (PFM frozen -> PFM-Scratch from random init -> PFM-FT pretrained and finetuned end to end);
# SuTraN = red; FM-v2 = yellow-green, kept clear of the turquoise ramp.
COL = {"PFM": "#7ecfcf", "PFM-Scratch": "#2f9ea6", "PFM-FT": "#0f6470", "SuTraN": "#d62728",
       "FM-v2 Proto": "#4d8b1f", "FM-v2 kNN": "#a8cf5c"}
BAR_METHODS = ["PFM", "PFM-Scratch", "PFM-FT", "SuTraN", "FM-v2 Proto", "FM-v2 kNN"]
ACC = [("next_activity", "Next\nactivity"), ("next_3_activities", "Next 3"), ("next_5_activities", "Next 5"),
       ("future_activity_set", "Future\nset")]
MAE = [("next_time", "Next\ntime"), ("remaining_time", "Remaining\ntime"), ("remaining_count", "Remaining\ncount")]
LOGS = [("helpdesk", "Helpdesk"), ("mimic_transfer", "MIMIC"), ("bpi13_incidents", "BPI13"), ("BPI20ID", "BPI20ID"), ("BPI17", "BPI17")]

norm = pd.read_csv(os.path.join(ROOT, "results", "sota_agg_data.csv"))
agg = norm.groupby("task")[BAR_METHODS].mean()
tp = pd.read_csv(os.path.join(ROOT, "results", "timing_pinned.csv"))


def minutes(lg, method, tasks):
    v = tp[(tp.log == lg) & (tp.method == method) & (tp.tasks == tasks)].minutes
    return float(v.iloc[0]) if len(v) else np.nan


plt.rcParams.update({"font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [4, 3, 3.2]})
w = 0.13
for ax, tasks, ylab, title in ((axes[0], ACC, r"Accuracy / micro-F1 $\uparrow$", "(a) Activity tasks"),
                               (axes[1], MAE, r"normalised MAE $\downarrow$", "(b) Time tasks")):
    x = np.arange(len(tasks))
    for j, meth in enumerate(BAR_METHODS):
        xs = x + (j - (len(BAR_METHODS) - 1) / 2) * w
        means = [agg.loc[t, meth] for t, _ in tasks]
        ax.bar(xs, [0 if np.isnan(v) else v for v in means], w, color=COL[meth], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab in tasks])
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.4, 1.0)
axes[1].set_yscale("log")
axes[1].set_ylim(0.5, 3.2)
axes[1].set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
axes[1].set_yticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
axes[1].yaxis.set_minor_formatter(NullFormatter())
axes[1].axhline(1.0, color="#bbb", lw=0.8, zorder=1)

# ---- (c) wall-clock ----
ax = axes[2]
x = np.arange(len(LOGS))
# \PFM's adaptation cost is measured with the frozen features CACHED once per log (the backbone is
# frozen, so it need not be re-run every epoch).
# PFM-Scratch is the same architecture trained end to end from random init on the target log.
# SuTraN's `7` is the timing file's task count: all seven tasks are SCORED from its outputs, but it
# trains only its three prediction heads in one joint run per log.
WALL = [("PFM", "PFM-cached", 2, COL["PFM"]), ("PFM-Scratch", "PFM-Scratch", 2, COL["PFM-Scratch"]),
        ("PFM-FT", "PFM-FT", 2, COL["PFM-FT"]), ("SuTraN", "SuTraN", 7, COL["SuTraN"]),
        ("FM-v2", "FM-v2", 2, COL["FM-v2 Proto"])]
bw = 0.16
for j, (name, method, tasks, color) in enumerate(WALL):
    y = [minutes(lg, method, tasks) for lg, _ in LOGS]
    ax.bar(x + (j - (len(WALL) - 1) / 2) * bw, y, width=bw, color=color, edgecolor="white",
           linewidth=0.4, label=name, zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.1, 1000)          # bars start at 0.1 min; headroom above for the panel legend
ax.set_xticks(list(x))
ax.set_xticklabels([("MIMIC-5k" if n == "MIMIC" else n) for _, n in LOGS], rotation=20)
ax.set_ylabel("wall-clock (minutes)", fontsize=10)
ax.set_title("(c) Cost of adaptation to a new log", fontsize=10.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

handles = [Patch(facecolor=COL[m], edgecolor="white", label=m) for m in BAR_METHODS]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.05),
           handlelength=1.4, columnspacing=1.6)
fig.tight_layout(w_pad=1.2)
# transparent page + axes background (the figure drops onto any slide/page colour)
fig.patch.set_alpha(0.0)
for a in axes:
    a.patch.set_alpha(0.0)
fig.savefig(os.path.join(HERE, "sota_wall.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "sota_wall.png"), dpi=200, bbox_inches="tight", transparent=True)
print("wrote sota_wall.pdf / .png")

tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/sota_wall.pdf}
\caption{Comparison with the baselines at full supervision, aggregated over the five held-out logs, and the measured
cost of adaptation. (a) Accuracy-type tasks (raw metric, mean over logs). (b) MAE tasks, each log's error divided by the
full-budget MAE of a randomly initialised frozen backbone with trained heads on that log (grey line at 1.0). (c) Wall-clock
minutes to adapt to a log and score its test partition for the two tasks all methods support, one full NVIDIA H200 per job,
one job at a time, one seed; \emph{FM-v2} embeds once for both read-outs. Because the \PFM{} backbone is frozen its states do
not depend on the head weights, so they are encoded once per log and both heads are trained from the cached tensors; this
reaches the same test metrics as re-running the backbone every epoch (max difference $0.009$ accuracy, $3.6\%$ MAE) and is
what the \PFM{} bar reports. \PFM-Scratch is the same architecture trained end to end from random
init on the target log. \emph{\PFM}: frozen backbone with trained heads;
\emph{\PFM-Scratch}: the same architecture trained end to end from random initialisation on the
target log; \emph{\PFM-FT}: \PFM{} fine-tuned end to end; \emph{SuTraN}: SuTraN-EW-NDA, official recipe, three seeds; \emph{FM-v2
Proto/kNN}: retrieval read-outs of the events-transf foundation model with $k$ selected on validation (next activity and
remaining time only).}
\label{fig:sota-wall}
\end{figure*}
"""
open(os.path.join(HERE, "sota_wall.tex"), "w").write(tex)
print("wrote sota_wall.tex")
