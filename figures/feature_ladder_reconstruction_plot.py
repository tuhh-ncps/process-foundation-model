"""Fingerprint reconstruction vs feature budget k, with the next-activity curve on top.

Reads results/feature_ladder.json (Phase A) and results/feature_ladder_summary.csv (next activity, C3/D).
Writes feature_ladder_reconstruction.pdf / .png next to this file.

Left axis, pretraining logs only: the frozen order's cumulative reconstruction J(F_k)/15 and the gain
Delta J/15 contributed by the descriptor added at step k. Letters mark which descriptor enters (frozen order
CHFNJBMLDOKGIEA).

Right axis, five held-out logs: the five-log mean next-activity accuracy, with the D3 seed-wise SD (SD of the
three five-log means across evaluation seeds) as error bars -- the plateau wiggles are of that size, so the
bars are what keeps them from being read as structure. The axis is a different quantity on a different scale
from J, so where the accuracy curve sits relative to the J line carries no meaning; only its SHAPE does. Its
range is set from the accuracy data alone, padded so the k = 0 -> 3 rise and the plateau are both legible.

A1 caveat: the accuracies come from the cached evaluator, so budgets are compared with each other and not
with the main result tables.
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
ACC_COLOUR = "#d55e00"

art = json.load(open(os.path.join(RES, "feature_ladder.json")))
ks = list(range(N + 1))
J = [0.0] + [s["J_over_15"] for s in art["steps"]]                      # J(F_0) = 0 by construction
gain = [s["delta_J"] / N for s in art["steps"]]                         # bar at step k

summ = {int(r["k"]): r for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_summary.csv")))}
acc = [float(summ[k]["mu"]) for k in ks]
sd = [float(summ[k]["seedwise_sd"]) for k in ks]

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9.0, 5.8))

ax.bar(ks[1:], gain, color="#9ecae1", width=0.7, zorder=1, label="gain at step $k$, $\\Delta J/15$")
ax.plot(ks, J, "o-", color="#0072b2", lw=2.4, ms=5, zorder=4, label="frozen order, $J(F_k)/15$")
for k in range(1, N + 1):
    ax.annotate(art["order_letters"][k - 1], (k, J[k]), textcoords="offset points", xytext=(-3, 13),
                fontsize=8.5, color="#0072b2", zorder=5,
                # The flat accuracy curve must cross the rising J line exactly once, so the collision cannot
                # be relocated: lift the letters clear of it and give them a halo thick enough to mask a line
                # passing behind a glyph.
                path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
ax.set_xticks(ks)
ax.set_xlim(-0.4, N + 0.4)
ax.set_ylim(0, 1.08)
ax.set_xlabel("number of fingerprint descriptors $k$ (letters = descriptor added)")
ax.set_ylabel("fraction of standardised fingerprint variance reconstructed")
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

tx = ax.twinx()                                   # different quantity and scale: compare shapes, not heights
tx.errorbar(ks, acc, yerr=sd, color=ACC_COLOUR, lw=2.0, marker="s", ms=5, capsize=3, elinewidth=1.2,
            zorder=3, label="next-activity accuracy (mean $\\pm$ SD over eval seeds)")
# Scale set by the accuracy data, but deliberately loose. A tight range magnifies the plateau: the k = 7 to
# k = 10 dip is 0.010, about two seed SDs, and on a tight axis it reads as a collapse and rebound. Padding to
# ~2.5x the data range keeps the k = 0 -> 2 rise legible while letting the plateau look like one.
span = max(acc) - min(acc)
tx.set_ylim(min(acc) - 0.75 * span, max(acc) + 0.75 * span)
tx.set_ylabel("next-activity accuracy, five-log mean", color=ACC_COLOUR)
tx.tick_params(axis="y", colors=ACC_COLOUR)
tx.grid(False)

handles = ax.get_legend_handles_labels()[0] + tx.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + tx.get_legend_handles_labels()[1]
fig.legend(handles, labels, fontsize=9.5, frameon=False, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.015))          # outside: the interior has no gap that clears three entries
ax.set_title("Fingerprint reconstruction and next-activity accuracy vs $k$")
for sp in ("top",):
    ax.spines[sp].set_visible(False)
    tx.spines[sp].set_visible(False)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(HERE, "feature_ladder_reconstruction.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "feature_ladder_reconstruction.png"), dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote feature_ladder_reconstruction.pdf / .png")
print("J/15     :", " ".join(f"{v:.3f}" for v in J))
print("accuracy :", " ".join(f"{v:.3f}" for v in acc))
print("seed SD  :", " ".join(f"{v:.3f}" for v in sd))
print(f"right-axis range: {min(acc) - 0.75 * span:.4f} .. {max(acc) + 0.75 * span:.4f}"
      f"  (data range {span:.4f}, max error bar {max(sd):.4f})")
