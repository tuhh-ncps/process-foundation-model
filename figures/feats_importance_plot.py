#!/usr/bin/env python3
r"""Feature-importance plot for the 15-dim role fingerprint, derived from feats_pca_corr.

The correlation matrix behind feats_pca_corr answers one question directly: how much of each
feature is information the OTHER fourteen cannot reconstruct. For feature j that is

    unique_j = 1 - R^2_j  =  1 / VIF_j  =  1 / (C^-1)_{jj}

with C the Pearson correlation matrix of the standardised fingerprints (298 activities pooled over
ten logs, cached in feats_pca_X.npy). A value of 1.0 means the feature is orthogonal to the rest;
0.0 means it is a linear combination of them and carries nothing of its own.

This is REDUNDANCY-based importance, not downstream predictive importance: it says what each
feature adds to the fingerprint, not how much any task needs it. The companion measure drawn as
grey markers is the mean |r| against the other fourteen, which is the same story read the other way.
A PCA-share measure was also computed and is deliberately NOT plotted: with 11 of 15 components
needed for 90% of the variance, every feature contributes 0.060-0.071 of the retained structure,
i.e. a flat 1/15, so it separates nothing.

Outputs feats_importance.pdf/.png.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NAMES = ["pagerank", "betweenness", "self_loop_p", "in_gap_median", "in_gap_std", "out_gap_median",
         "out_gap_std", "p_start", "p_terminal", "support", "pred_entropy", "succ_entropy",
         "mean_pos", "std_pos", "rework_p"]
PRETTY = {"pagerank": "PageRank", "betweenness": "Betweenness", "self_loop_p": "Self-loop prob.",
          "p_start": "Case-start prob.", "p_terminal": "Case-end prob.", "support": "Case coverage",
          "rework_p": "Repetition rate", "pred_entropy": "Predecessor entropy",
          "succ_entropy": "Successor entropy", "in_gap_median": "In-gap median",
          "in_gap_std": "In-gap spread", "out_gap_median": "Out-gap median",
          "out_gap_std": "Out-gap spread", "mean_pos": "Mean position", "std_pos": "Position spread"}
FAMILIES = [("Graph centrality", ["pagerank", "betweenness"], "#6e6e6e"),
            ("Control-flow structure", ["self_loop_p", "p_start", "p_terminal", "support", "rework_p"], "#0072b2"),
            ("Branching entropy", ["pred_entropy", "succ_entropy"], "#009e73"),
            ("Temporal performance", ["in_gap_median", "in_gap_std", "out_gap_median", "out_gap_std"], "#d55e00"),
            ("Position in case", ["mean_pos", "std_pos"], "#cc79a7")]
FAM_OF = {f: (n, c) for n, fs, c in FAMILIES for f in fs}

X = np.load(os.path.join(ROOT, "results", "feats_pca_X.npy"))
Xs = (X - X.mean(0)) / X.std(0)
C = np.corrcoef(Xs.T)
uniq = 1.0 / np.diag(np.linalg.inv(C))          # 1 - R^2 of feature j on the other 14
red = (np.abs(C).sum(1) - 1) / (len(NAMES) - 1)  # mean |r| against the other 14
order = np.argsort(uniq)                         # ascending -> most unique at the top of the bar chart

plt.rcParams.update({"font.size": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(6.4, 4.6))
y = np.arange(len(NAMES))
ax.barh(y, uniq[order], color=[FAM_OF[NAMES[j]][1] for j in order], edgecolor="white", linewidth=0.6, height=0.74)
ax.set_yticks(y)
ax.set_yticklabels([PRETTY[NAMES[j]] for j in order])
ax.set_xlabel(r"unique information  $1-R^2$  (share the other 14 features cannot explain)", fontsize=10)
ax.set_xlim(0, 1.0)
ax.axvline(uniq.mean(), color="#555", ls=(0, (4, 3)), lw=1.1, zorder=4)
ax.text(uniq.mean(), len(NAMES) - 0.35, "mean %.2f" % uniq.mean(), fontsize=9, color="#555",
        ha="center", va="bottom")
ax.set_ylim(-0.7, len(NAMES) - 0.1)
for i, j in enumerate(order):
    ax.text(uniq[j] + 0.012, i, "%.2f" % uniq[j], va="center", fontsize=9, color="#333")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
handles = [Patch(facecolor=c, edgecolor="white", label=n) for n, _fs, c in FAMILIES]
ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True, framealpha=0.0,
          handlelength=1.3, labelspacing=0.45, bbox_to_anchor=(1.0, 0.02))
fig.tight_layout()
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig("feats_importance.pdf", bbox_inches="tight", transparent=True)
fig.savefig("feats_importance.png", dpi=200, bbox_inches="tight", transparent=True)

print("%-20s %-24s %8s %8s" % ("feature", "family", "unique", "mean|r|"))
for j in order[::-1]:
    print("%-20s %-24s %8.3f %8.3f" % (PRETTY[NAMES[j]], FAM_OF[NAMES[j]][0], uniq[j], red[j]))
print("\nmean unique %.3f | min %.3f (%s) | max %.3f (%s)"
      % (uniq.mean(), uniq.min(), PRETTY[NAMES[int(np.argmin(uniq))]], uniq.max(), PRETTY[NAMES[int(np.argmax(uniq))]]))
print("wrote feats_importance.pdf / .png")
