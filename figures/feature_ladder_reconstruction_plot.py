"""Fingerprint reconstruction vs feature budget k, with the downstream curves on top.

Reads results/feature_ladder.json (Phase A), results/feature_ladder_summary.csv (next activity, C3/D) and
results/feature_ladder_tasks_summary.csv (amendment A2). Writes feature_ladder_reconstruction.pdf / .png
next to this file.

Left axis, pretraining logs only: the frozen order's cumulative reconstruction J(F_k)/15 and the gain
Delta J/15 contributed by the descriptor added at step k. Letters mark which descriptor enters (frozen order
CHFNJBMLDOKGIEA).

Right axis, five held-out logs: the five-log mean of the four classification-style downstream tasks. These are
a different quantity on a different scale from J, so where a downstream curve sits relative to the J line
carries no meaning -- only the SHAPE of each curve does. The MAE tasks are left out because their units do not
share an axis (see figures/feature_ladder_tasks_plot.py).

A1 caveat: the downstream numbers come from the cached evaluator, so budgets are compared with each other and
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

# (column task key, label, colour, marker) -- the four tasks whose metric is a fraction in [0, 1]
TASKS = [("next_activity", "next activity", "#d55e00", "s"),
         ("next_3_activities", "next 3 activities", "#009e73", "^"),
         ("next_5_activities", "next 5 activities", "#cc79a7", "v"),
         ("future_activity_set", "future activity set", "#762a83", "D")]

art = json.load(open(os.path.join(RES, "feature_ladder.json")))
ks = list(range(N + 1))
J = [0.0] + [s["J_over_15"] for s in art["steps"]]                      # J(F_0) = 0 by construction
gain = [s["delta_J"] / N for s in art["steps"]]                         # bar at step k

na = {int(r["k"]): float(r["mu"]) for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_summary.csv")))}
tasks = {}
for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_tasks_summary.csv"))):
    tasks.setdefault(r["task"], {})[int(r["k"])] = float(r["mu"])
tasks["next_activity"] = na
curves = {key: [tasks[key][k] for k in ks] for key, _, _, _ in TASKS}

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(9.0, 5.8))

ax.bar(ks[1:], gain, color="#9ecae1", width=0.7, zorder=1, label="gain at step $k$, $\\Delta J/15$")
ax.plot(ks, J, "o-", color="#0072b2", lw=2.4, ms=5, zorder=4, label="frozen order, $J(F_k)/15$")
for k in range(1, N + 1):
    ax.annotate(art["order_letters"][k - 1], (k, J[k]), textcoords="offset points", xytext=(-3, 13),
                fontsize=8.5, color="#0072b2", zorder=5,
                # The flat future-activity-set curve must cross the rising J line exactly once, so the
                # collision cannot be relocated: lift the letters clear of the crossing and give them a halo
                # thick enough to mask a line passing behind a glyph.
                path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])
ax.set_xticks(ks)
ax.set_xlim(-0.4, N + 0.4)
ax.set_ylim(0, 1.08)
ax.set_xlabel("number of fingerprint descriptors $k$ (letters = descriptor added)")
ax.set_ylabel("fraction of standardised fingerprint variance reconstructed")
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

tx = ax.twinx()                                   # different quantity and scale: compare shapes, not heights
for key, label, colour, marker in TASKS:
    tx.plot(ks, curves[key], color=colour, lw=1.8, marker=marker, ms=4.5, alpha=0.9, zorder=3, label=label)
lo = min(min(v) for v in curves.values())
hi = max(max(v) for v in curves.values())
pad = 0.06 * (hi - lo)
tx.set_ylim(lo - pad, hi + 2.5 * pad)             # headroom so the curves clear the J line's upper half
tx.set_ylabel("downstream metric, five-log mean\n(accuracy; micro-F1 for future activity set)")
tx.grid(False)

handles = ax.get_legend_handles_labels()[0] + tx.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + tx.get_legend_handles_labels()[1]
fig.legend(handles, labels, fontsize=9.5, frameon=False, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.015))          # outside: no interior gap clears six entries
ax.set_title("Fingerprint reconstruction and downstream accuracy vs $k$")
for sp in ("top",):
    ax.spines[sp].set_visible(False)
    tx.spines[sp].set_visible(False)

fig.tight_layout(rect=(0, 0.10, 1, 1))
fig.savefig(os.path.join(HERE, "feature_ladder_reconstruction.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "feature_ladder_reconstruction.png"), dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote feature_ladder_reconstruction.pdf / .png")
print("J/15 :", " ".join(f"{v:.3f}" for v in J))
for key, label, _, _ in TASKS:
    print(f"{label:22s}", " ".join(f"{v:.3f}" for v in curves[key]))
