"""Per-descriptor variance reconstructed vs feature budget k (Phase A, pretraining logs only).

Reads results/feature_ladder.json and results/feature_ladder_fingerprints.npz, and reuses the estimators in
scripts/feature_ladder.py so the heatmap cannot drift from the frozen order. Writes
feature_ladder_descriptor_heatmap.pdf / .png next to this file.

Cell (descriptor f, budget k) = R^2 of f from the first k descriptors of the frozen order, under the
log-balanced correlation matrix R_bar. Rows are listed in the order descriptors are added; the orange box marks
the budget at which each descriptor enters (R^2 = 1 from then on). Rows are labelled with the Table-1 letters
only; no y-axis title, since the letters are self-explanatory with the caption key.
"""
import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("fl", os.path.join(REPO, "scripts", "feature_ladder.py"))
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

art = json.load(open(os.path.join(REPO, "results", "feature_ladder.json")))
z = np.load(os.path.join(REPO, "results", "feature_ladder_fingerprints.npz"), allow_pickle=False)
logs = [log for log, _ in fl.PRETRAIN]
R_bar = sum(fl.log_correlation(z[f"{log}__X"])[0] for log in logs) / len(logs)
order = art["order_code_indices"]
N = fl.N

M = np.array([fl.recovered(R_bar, order[:k], fl.RCOND) for k in range(N + 1)])   # (16 budgets, 15), code order
assert np.allclose(M.sum(axis=1)[1:], [s["J"] for s in art["steps"]])           # same J as the frozen artifact
H = M[:, order].T                                                                # rows in the order added

TXT = 16                                      # ticks and titles share one size
plt.rcParams.update({"font.size": TXT})
fig, ax = plt.subplots(figsize=(4.6, 3.8))   # one third of a figure* row
im = ax.imshow(H, aspect="auto", cmap="Blues", vmin=0, vmax=1)
for i, f in enumerate(order):
    ax.add_patch(plt.Rectangle((i + 1 - 0.5, i - 0.5), 1, 1, fill=False, ec="#d55e00", lw=1.4))
ax.set_yticks(range(N))
ax.set_yticklabels([fl.LETTER[fl.NAMES[f]] for f in order], fontsize=12)
ax.set_xticks(range(0, N + 1, 3))
ax.set_xlabel("# descriptors $k$", fontsize=TXT)
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
cb.set_label("$R^2$", fontsize=TXT)
cb.ax.tick_params(labelsize=TXT)
ax.tick_params(axis="x", labelsize=TXT)

fig.tight_layout()
stem = os.path.join(HERE, "feature_ladder_descriptor_heatmap")
fig.savefig(stem + ".pdf", bbox_inches="tight", transparent=True)
fig.savefig(stem + ".png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", stem + ".pdf / .png | row letters:", "".join(fl.LETTER[fl.NAMES[f]] for f in order))
