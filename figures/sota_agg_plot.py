"""Full-supervision comparison with SOTA baselines, aggregated over the five held-out logs.

Methods: Rand. (random role encoder + random frozen backbone, trained heads), PFM (frozen), PFM-FT,
SuTraN (official recipe, NDA, equal weighting, 3 seeds), FM-v2 Proto / kNN (events-transf retrieval
read-outs, k selected on validation, one deterministic run). All at full budget on identical test
queries.  Accuracy-type tasks: raw metric.  MAE tasks: each log's MAE divided by that log's Rand. MAE
at full budget (unitless; lower is better), same convention as frozen_agg_*.
Bars = mean over logs.  Inputs: results/v2_all.csv,
results/baselines/sutran_v2 (preferred) or sutran, results/baselines/fmv2_v2 (preferred) or fmv2.
Outputs: sota_agg.pdf/.png, sota_agg_data.csv, sota_agg.tex (figure block).
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root
D = os.path.join(ROOT, "results")
BASE = os.path.join(D, "baselines")   # SuTraN / FM-v2 replays
LOGS = [("helpdesk", "Helpdesk"), ("bpi13_incidents", "BPI13"), ("mimic_transfer", "MIMIC"),
        ("BPI20ID", "BPI20ID"), ("BPI17", "BPI17")]
ACC = [("next_activity", "Next\nactivity"), ("next_3_activities", "Next 3"), ("next_5_activities", "Next 5"),
       ("future_activity_set", "Future\nset")]
MAE = [("next_time", "Next\ntime"), ("remaining_time", "Remaining\ntime"), ("remaining_count", "Remaining\ncount")]
# Our family = turquoise, deepening with how much of the model adapts (PFM -> PFM-RFT -> PFM-FT);
# SuTraN = red; FM-v2 = yellow-green (kept clear of the turquoise ramp).
METHODS = [("Rand.", "#9a9a9a"), ("PFM", "#7ecfcf"), ("PFM-RFT", "#2ba3a8"), ("PFM-FT", "#0f6470"),
           ("SuTraN", "#d62728"), ("FM-v2 Proto", "#4d8b1f"), ("FM-v2 kNN", "#a8cf5c")]
PLOT = [m for m in METHODS if m[0] != "Rand."]  # Rand. stays the MAE normaliser but is not drawn
LOG_MARK = {"Helpdesk": "o", "BPI13": "s", "MIMIC": "^", "BPI20ID": "D", "BPI17": "v"}


def ours(df, lg, arm, task):
    v = df[(df.log == lg) & (df.arm == arm) & (df.task == task) & (df.n_labels == "all")].value
    return float(v.mean()) if len(v) else np.nan


def sutran(lg, task):
    fs = glob.glob(os.path.join(BASE, "sutran_v2", "%s_s*.csv" % lg)) or glob.glob(os.path.join(BASE, "sutran", "%s_s*.csv" % lg))
    if not fs:
        return np.nan, ""
    s = pd.concat([pd.read_csv(f) for f in fs])
    src = "v2" if "sutran_v2" in fs[0] else "old"
    v = s[s.task == task].value
    return (float(v.mean()) if len(v) else np.nan), src


def fmv2(lg, task):
    f = os.path.join(BASE, "fmv2_v2", "%s_val_%s.csv" % (lg, task))
    if not os.path.exists(f):
        f = os.path.join(BASE, "fmv2", "%s_val_%s.csv" % (lg, task))
    if not os.path.exists(f):
        return np.nan, np.nan
    d = pd.read_csv(f)
    d = d[d.n_labels.astype(str) == "all"]
    hib = task == "next_activity"
    out = []
    for p in ("proto_head", "foundation_knn"):
        v = d[(d.predictor == p) & (d.split == "val")].groupby("k").value.mean()
        k = v.idxmax() if hib else v.idxmin()
        out.append(float(d[(d.predictor == p) & (d.split == "test") & (d.k == k)].value.mean()))
    return out[0], out[1]


new = pd.read_csv(os.path.join(D, "v2_all.csv"))
new["n_labels"] = new.n_labels.astype(str)
rows, sources = [], {}
for lg, name in LOGS:
    for task, _ in ACC + MAE:
        r = {"log": name, "task": task,
             "Rand.": ours(new, lg, "random_role", task), "PFM": ours(new, lg, "pfm", task),
             "PFM-RFT": ours(new, lg, "pfm_rft", task), "PFM-FT": ours(new, lg, "pfm_ft", task)}
        r["SuTraN"], sources[name] = sutran(lg, task)
        r["FM-v2 Proto"], r["FM-v2 kNN"] = fmv2(lg, task) if task in ("next_activity", "remaining_time") else (np.nan, np.nan)
        rows.append(r)
raw = pd.DataFrame(rows)
norm = raw.copy()
for task, _ in MAE:
    m = norm.task == task
    ref = norm.loc[m].set_index("log")["Rand."]
    for meth, _ in METHODS:
        norm.loc[m, meth] = norm.loc[m, meth].values / norm.loc[m, "log"].map(ref).values
raw.to_csv(os.path.join(ROOT, "results", "sota_agg_data_raw.csv"), index=False)
norm.to_csv(os.path.join(ROOT, "results", "sota_agg_data.csv"), index=False)
print("SuTraN sources per log:", sources)
agg = norm.groupby("task")[[m for m, _ in METHODS]].mean()
print(agg.loc[[t for t, _ in ACC + MAE]].round(3).to_string())

# ---- figure ----
plt.rcParams.update({"font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [4, 3]})
w = 0.13
for ax, tasks, ylab in ((axes[0], ACC, r"Accuracy / micro-F1 $\uparrow$"),
                        (axes[1], MAE, r"normalised MAE $\downarrow$")):
    x = np.arange(len(tasks))
    for j, (meth, color) in enumerate(PLOT):
        xs = x + (j - (len(PLOT) - 1) / 2) * w
        means = [agg.loc[t, meth] for t, _ in tasks]
        ax.bar(xs, [0 if np.isnan(v) else v for v in means], w, color=color, edgecolor="white", linewidth=0.5,
               label=meth, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab in tasks])
    ax.set_ylabel(ylab, fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylim(0.4, 1.0)
axes[1].set_yscale("log")
axes[1].set_ylim(0.5, 3.2)
axes[1].set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
axes[1].set_yticklabels(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"])
axes[1].axhline(1.0, color="#bbb", lw=0.8, zorder=1)
from matplotlib.ticker import NullFormatter
axes[1].yaxis.set_minor_formatter(NullFormatter())
h, l = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06),
           handlelength=1.4, columnspacing=1.2)
fig.tight_layout()
fig.patch.set_alpha(0.0)
for a in axes:
    a.patch.set_alpha(0.0)
fig.savefig(os.path.join(HERE, "sota_agg.pdf"), bbox_inches="tight", transparent=True)
fig.savefig(os.path.join(HERE, "sota_agg.png"), dpi=200, bbox_inches="tight", transparent=True)
print("wrote sota_agg.pdf / .png")

tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/sota_agg.pdf}
\caption{Full-supervision comparison with the baselines, aggregated over the five held-out logs
(Helpdesk, BPI13, MIMIC, BPI20ID, BPI17), identical test queries; bars are the mean over logs. \emph{\PFM}:
pretrained frozen backbone with trained heads; \emph{\PFM-RFT}: frozen backbone with the role encoder
(9.5k parameters, $0.19\%$ of the backbone) trained alongside the head; \emph{\PFM-FT}: \PFM{} fine-tuned end to end;
\emph{SuTraN}: SuTraN-EW-NDA, the official recipe with equal loss weighting and non-data-aware inputs, three seeds; \emph{FM-v2 Proto/kNN}:
retrieval read-outs of the events-transf foundation model with $k$ selected on validation (next activity and
remaining time only). Left: accuracy-type tasks (raw metric). Right: MAE tasks, each log's error divided by
the full-budget MAE of a randomly initialised frozen backbone with trained heads on that log (1.0, grey line), so
that logs of different time scale are comparable.}
\label{fig:sota-agg}
\end{figure*}
"""
open(os.path.join(HERE, "sota_agg.tex"), "w").write(tex)
print("wrote sota_agg.tex")
