"""One figure with all seven label-efficiency panels (2 x 4 grid, legend in the 8th slot).

Same data and aggregation as frozen_aggregate_panels.py (v2_all.csv; mean over the five held-out logs at each
label budget; MAE tasks divided by the Random-role MAE at full budget; bands = +/-1 s.d. across logs).

Which PFM backbone the curves come from is a command-line choice:

    python figures/frozen_agg_merged.py            # seed-0 backbone (arms pfm, pfm_ft)    -> frozen_agg_all.*
    python figures/frozen_agg_merged.py --seed2    # seed-2 backbone (arms pfm_s2, pfm_ft_s2) -> frozen_agg_all_seed2.*

Frozen Random and PFM-Scratch load no pretrained weights, so the same random_role and pfm_scratch
curves serve both variants.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import frozen_aggregate_panels as fap

HERE = os.path.dirname(os.path.abspath(__file__))
SEED2 = "--seed2" in sys.argv
SUFFIX = "_seed2" if SEED2 else ""
PFM_ARM, FT_ARM = ("pfm_s2", "pfm_ft_s2") if SEED2 else ("pfm", "pfm_ft")
BACKBONE = ("backbone-20260907-103600-multi-none-v2-gin15-s2-bfb92b" if SEED2
            else "backbone-20260906-153102-multi-none-v2-gin15-17fc3c")

# Arms ordered by pretraining and how much of the model adapts to the new log. Four maximally distinct hues
# (Okabe-Ito, colour-blind safe): this figure compares the arms against each other.
fap.SER = [("Frozen Random", "random_role", "#6e6e6e", "dashed", "^"),
           ("PFM-Scratch", "pfm_scratch", "#009e73", "dashed", "D"),
           ("PFM", PFM_ARM, "#0072b2", "solid", "o"),          # only the proposed frozen model is solid
           ("PFM-FT", FT_ARM, "#d55e00", "dashed", "s")]

per = fap.load()
missing = [a for _n, a, *_ in fap.SER if a not in set(per.backbone_alias)]
if missing:
    sys.exit("no rows for arm(s) %s in results/v2_all.csv; collect the grid first" % missing)
print("PFM backbone:", BACKBONE)

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
handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=6.5, lw=2.6, markeredgecolor="white",
                  markeredgewidth=0.7, label=n) for n, _a, c, ls, mk in fap.SER]
lax.legend(handles=handles, loc="center", fontsize=12, frameon=True, framealpha=0.0,
           handlelength=3.2, borderpad=1.0, labelspacing=0.8)
fig.tight_layout(w_pad=1.0, h_pad=1.4)
# transparent page + panel backgrounds (the figure drops onto any slide/page colour)
fig.patch.set_alpha(0.0)
for a in axes.flat:
    a.patch.set_alpha(0.0)
fig.savefig(os.path.join(HERE, "frozen_agg_all%s.pdf" % SUFFIX), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "frozen_agg_all%s.png" % SUFFIX), dpi=200, bbox_inches="tight", transparent=True)
print("wrote frozen_agg_all%s.pdf / .png" % SUFFIX)

seed_note = ("pretraining seed 2" if SEED2 else "pretraining seed 0")
tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/frozen_agg_all%(suffix)s.pdf}
\caption{Label-efficiency curves aggregated over the five held-out logs (BPI13, BPI17, BPI20ID, Helpdesk, MIMIC):
mean of the task metric over logs at each label budget, probes trained for up to 100 epochs with early stopping on the
validation partition, three seeds; bands are $\pm 1$ s.d.\ across logs. \PFM{} and \PFM-FT use the GIN-15 backbone
with %(seed)s. The arms differ in how much of the model adapts
to the new log. \emph{Frozen Random}: randomly initialised role encoder and frozen backbone with trained heads;
\emph{\PFM-Scratch}: the same architecture, randomly initialised and trained end to end on the target log;
\emph{\PFM}: pretrained frozen backbone with trained heads; \emph{\PFM-FT}: \PFM{} fine-tuned end to end
(role encoder, backbone and head). For the three MAE tasks each log's error is divided by that log's Frozen Random MAE
at full supervision, so the curves are unitless and comparable across logs. The budget axis is
logarithmic; \emph{all} is the full training partition (2.7k--20k cases depending on the log), which does not lie on
that scale, so the last hop is marked with a break symbol (//). \emph{Pretext head} marks tasks with a
matching pretraining objective; \emph{unseen task} marks tasks the backbone was never trained for.}
\label{fig:agg-all%(suffix)s}
\end{figure*}
""" % {"suffix": SUFFIX, "seed": seed_note}
open(os.path.join(HERE, "frozen_agg_all%s.tex" % SUFFIX), "w").write(tex)
print("wrote frozen_agg_all%s.tex" % SUFFIX)
