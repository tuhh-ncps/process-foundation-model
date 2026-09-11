"""One figure with all seven label-efficiency panels (2 x 4 grid, legend in the 8th slot).

Same data and aggregation as frozen_aggregate_panels.py (v2_all.csv; mean over the five held-out logs at each
label budget; MAE tasks divided by the Random-role MAE at full budget; bands = +/-1 s.d. across logs).
Colours match sota_wall: Rand. grey, PFM light orange, PFM-FT dark orange. Outputs frozen_agg_all.pdf/.png/.tex.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import frozen_aggregate_panels as fap

HERE = os.path.dirname(os.path.abspath(__file__))
# Four arms, ordered by how much of the model adapts to the new log: nothing pretrained ->
# frozen features -> role encoder only -> everything. Four maximally distinct hues (Okabe-Ito,
# colour-blind safe) rather than one colour family: this figure compares the arms against each
# other, so telling the three PFM curves apart matters more than grouping them.
fap.SER = [("Frozen Random", "random_role", "#6e6e6e", "solid", "^"),
           ("PFM", "pfm", "#0072b2", "solid", "o"),
           ("PFM-FT", "pfm_ft", "#d55e00", "solid", "s")]

per = fap.load()
plt.rcParams.update({"font.size": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6})
fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6))
for ax, (task, cap, ylab) in zip(axes.flat, fap.TASKS):
    fap.draw(ax, per, task, ylab, legend=False)
    tag = "pretext head" if task in fap.PRETEXT else "unseen task"
    ax.set_title("%s (%s)" % (cap, tag), fontsize=10)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_xlabel("labelled cases (log scale)", fontsize=9.5)
# legend in the 8th slot
lax = axes.flat[-1]
lax.axis("off")
handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=6, lw=1.9, markeredgecolor="white",
                  markeredgewidth=0.7, label=n) for n, _a, c, ls, mk in fap.SER]
lax.legend(handles=handles, loc="center", fontsize=12, frameon=True, framealpha=0.0,
           handlelength=2.6, borderpad=1.0, labelspacing=0.8)
fig.tight_layout(w_pad=1.0, h_pad=1.4)
# transparent page + panel backgrounds (the figure drops onto any slide/page colour)
fig.patch.set_alpha(0.0)
for a in axes.flat:
    a.patch.set_alpha(0.0)
fig.savefig(os.path.join(HERE, "frozen_agg_all.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "frozen_agg_all.png"), dpi=200, bbox_inches="tight", transparent=True)
print("wrote frozen_agg_all.pdf / .png")

tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/frozen_agg_all.pdf}
\caption{Label-efficiency curves aggregated over the five held-out logs (BPI13, BPI17, BPI20ID, Helpdesk, MIMIC):
mean of the task metric over logs at each label budget, probes trained for up to 100 epochs with early stopping on the
validation partition, three seeds; bands are $\pm 1$ s.d.\ across logs. The arms differ in how much of the model adapts
to the new log. \emph{Frozen Random}: randomly initialised role encoder and frozen backbone with trained heads;
\emph{\PFM}: pretrained frozen backbone with trained heads; \emph{\PFM-FT}: \PFM{} fine-tuned end to end
(role encoder, backbone and head). For the three MAE tasks each log's error is divided by that log's Frozen Random MAE
at full supervision, so the curves are unitless and comparable across logs. The budget axis is
logarithmic; \emph{all} is the full training partition (2.7k--20k cases depending on the log), which does not lie on
that scale, so the last hop is drawn as a dashed connector. \emph{Pretext head} marks tasks with a
matching pretraining objective; \emph{unseen task} marks tasks the backbone was never trained for.}
\label{fig:agg-all}
\end{figure*}
"""
open(os.path.join(HERE, "frozen_agg_all.tex"), "w").write(tex)
print("wrote frozen_agg_all.tex")
