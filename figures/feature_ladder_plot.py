"""Feature-budget ladder figure (protocols/feature_ladder.md, D7).

Reads results/feature_ladder_summary.csv and results/feature_ladder_summary.json
(scripts/feature_ladder_analysis.py report). Writes feature_ladder.pdf / .png next to this file.

(a) five-log mean next-activity accuracy vs number of fingerprint descriptors k, error bars = SD of the three
    five-log means across downstream evaluation seeds (downstream-head stochasticity only); horizontal line at
    mu_15 - sigma_15 (the pre-registered near-full region); Frozen Random reference; markers at k_near and at the
    first budget containing case-start or case-end probability; J(S)/15 on the right axis.
(b) paired difference to k = 15 per budget: five-log mean with a t(4) 95% interval and the five per-log values.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]

rows = [{k: (float(v) if k not in ("k",) else int(v)) for k, v in r.items()}
        for r in csv.DictReader(open(os.path.join(RES, "feature_ladder_summary.csv")))]
summ = json.load(open(os.path.join(RES, "feature_ladder_summary.json")))
ks = [r["k"] for r in rows]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.0))

ax.errorbar(ks, [r["mu"] for r in rows], yerr=[r["seedwise_sd"] for r in rows], color="#0072b2", marker="o",
            ms=5, lw=1.8, capsize=3, label="GIN-$k$ (mean $\\pm$ SD over eval seeds)")
ax.axhline(summ["threshold_primary"], color="#0072b2", ls="--", lw=1.1, label="$\\mu_{15}-\\sigma_{15}$")
ax.axhline(summ["frozen_random_reference"], color="#6e6e6e", ls=":", lw=1.4, label="Frozen Random")
ax.axvline(summ["k_near_primary"], color="#d55e00", lw=1.0, alpha=0.8, label=f"$k_\\mathrm{{near}}={summ['k_near_primary']}$")
ax.axvline(summ["case_start_or_end_enters_at_k"], color="#999999", lw=0.9, ls="-.",
           label=f"case-start/end enters ($k={summ['case_start_or_end_enters_at_k']}$)")
ax.set_xlabel("number of fingerprint descriptors $k$")
ax.set_ylabel("next-activity accuracy (5-log mean)")
ax.set_xticks(ks)
jx = ax.twinx()
jx.plot(ks, [r["J_over_15"] for r in rows], color="#009e73", lw=1.2, alpha=0.7)
jx.set_ylabel("$J(S)/15$", color="#009e73")
jx.tick_params(axis="y", colors="#009e73")
jx.set_ylim(0, 1.02)
jx.grid(False)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.0)
ax.set_title("(a) accuracy vs feature budget", fontsize=11)

bx.axhline(0, color="#444444", lw=0.9)
for r in rows:
    if r["k"] == 15:
        continue
    bx.plot([r["k"], r["k"]], [r["delta_ci_low"], r["delta_ci_high"]], color="#0072b2", lw=2.0, alpha=0.6)
    bx.plot(r["k"], r["delta_mean"], "o", color="#0072b2", ms=5)
    bx.scatter([r["k"] + 0.18] * len(LOGS), [r[f"delta_{log}"] for log in LOGS], s=9, color="#6e6e6e", zorder=3)
bx.set_xlabel("number of fingerprint descriptors $k$")
bx.set_ylabel("accuracy difference to $k=15$")
bx.set_xticks(ks[:-1])
bx.set_title("(b) paired difference to $k=15$ (95% $t$-interval; dots = logs)", fontsize=11)

for a in (ax, bx):
    for sp in ("top",):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "feature_ladder.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "feature_ladder.png"), dpi=200, bbox_inches="tight", transparent=True)
print("wrote feature_ladder.pdf / .png")
