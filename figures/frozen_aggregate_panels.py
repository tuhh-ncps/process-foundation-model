"""One PDF per task for the raw-metric aggregate, plus a LaTeX subfigure block.

v2 protocol (leak-free: position_ratio removed, 64-event exclusion, ID-free backbone, budget
corpus). Data: results/v2_all.csv (collect_v2.py). Arms: random_role -> Frozen Random, pfm_scratch -> PFM-Scratch,
pfm_ft -> PFM-FT (PFM fine-tuned end to end), pfm -> PFM (frozen). Mean over the 5 held-out logs,
raw metric, +/-1 s.d. across logs. Each panel is emitted as its own
PDF with NO title and NO legend -- the task name lives in the LaTeX subcaption. The legend
is a standalone frozen_agg_legend.pdf that fills the 8th slot of the 2x4 grid.

Outputs: frozen_agg_<task>.pdf x7, frozen_agg_legend.pdf, and frozen_agg_panels.tex.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib import patheffects
from matplotlib.transforms import blended_transform_factory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root
DATA = os.path.join(ROOT, "results")

LOGS = ["bpi13_incidents", "BPI17", "BPI20ID", "helpdesk", "mimic_transfer"]
B = ["10", "30", "100", "300", "1000", "all"]
XT = ["10", "30", "100", "300", "1k", "all"]

PRETEXT = {"next_activity", "next_time", "remaining_time"}
# key, subcaption text, y-label
TASKS = [
    ("next_activity", "Next activity", r"Accuracy $\uparrow$"),
    ("next_3_activities", "Next 3 activities", r"Accuracy $\uparrow$"),
    ("next_5_activities", "Next 5 activities", r"Accuracy $\uparrow$"),
    ("future_activity_set", "Future activity set", r"micro-F1 $\uparrow$"),
    ("next_time", "Next event time", r"normalised MAE $\downarrow$"),
    ("remaining_time", "Remaining time", r"normalised MAE $\downarrow$"),
    ("remaining_count", "Remaining event count", r"normalised MAE $\downarrow$"),
]
# The budgets 10/30/100/300/1000 form a geometric ladder (alternating x3 and x3.33), so drawing
# them at equal spacing IS a log axis to within ~1% of the axis width. "all" is NOT on that scale:
# it is the full training partition, a different number of cases in every log (2.7k-20k), so the
# final hop is drawn solid but marked with a // break on every line and on the x-axis.
BREAK_KW = dict(ha="center", va="center", fontsize=13, fontweight="bold", zorder=7)  # no background box
BREAK_STROKE = 1.1   # extra outline (in the mark's own colour) so the // reads heavier than plain bold
# PFM_SEED selects the pre-trained GIN-15 backbone behind PFM and PFM-FT: 0 = 17fc3c (the manuscript figure),
# 2 = the pre-training seed-2 backbone (arms pfm_s2 / pfm_ft_s2). Non-zero seeds write *_s<seed> files, so the
# manuscript's seed-0 PDFs are never overwritten. Frozen Random and PFM-Scratch use no pre-trained backbone.
PFM_SEED = int(os.environ.get("PFM_SEED", "0"))
_SFX = "" if PFM_SEED == 0 else "_s%d" % PFM_SEED
SER = [
    ("Frozen Random", "random_role", "#9a9a9a", "dashed", "^"),
    ("PFM-Scratch", "pfm_scratch", "#009e73", "dashed", "D"),  # PFM architecture, random init, trained end to end
    ("PFM-FT", "pfm_ft" + _SFX, "#E1912F", "dashed", "s"),
    ("PFM", "pfm" + _SFX, "#2b6cb8", "solid", "o"),            # only the proposed frozen model is solid
]
LW = 2.4                   # line width of every curve
# Single panels are included at 0.24\textwidth: on this fixed canvas, PANEL_FS prints at the manuscript caption size
# (calibrated on the rendered page); tick labels slightly smaller so 100/300 do not touch. The legend PDF is
# included at scale=0.5, so LEGEND_FS prints at the same size.
PANEL_W, PANEL_H = 3.14, 2.58                      # inches, identical for all seven panels
PAD_L, PAD_B, PAD_R, PAD_T = 0.80, 0.64, 0.10, 0.12  # inches around the axes box
PANEL_FS, TICK_FS, LEGEND_FS = 14.5, 13, 14
REF_ALIAS = "random_role"  # MAE normaliser: this arm's full-budget MAE per log


def load():
    df = pd.read_csv(os.path.join(DATA, "v2_all.csv"))
    df["n_labels"] = df.n_labels.astype(str)
    df = df[df.log.isin(LOGS) & df.n_labels.isin(B) & df.seed.isin([0, 1, 2])]
    df = df.rename(columns={"arm": "backbone_alias"})
    cov = df[df.task == "next_activity"].groupby(["log", "backbone_alias"]).seed.nunique()
    print("seeds per log x arm:\n" + cov.unstack().to_string())
    per = df.groupby(["log", "task", "backbone_alias", "n_labels"]).value.mean().reset_index()
    # MAE tasks: divide by that log's Baseline MAE at full budget -> unitless, 1.0 = random
    # baseline's final error, so logs of very different scale become comparable.
    mae_tasks = {"next_time", "remaining_time", "remaining_count"}
    ref = per[(per.task.isin(mae_tasks)) & (per.backbone_alias == REF_ALIAS) & (per.n_labels == "all")]
    ref = ref.set_index(["log", "task"]).value
    def norm(r):
        if r.task in mae_tasks:
            return r.value / ref.get((r.log, r.task), np.nan)
        return r.value
    per["value"] = per.apply(norm, axis=1)
    return per


def draw(ax, per, task, ylab, legend):
    d = per[per.task == task]
    x = np.arange(len(B))
    for name, alias, c, ls, mk in SER:
        a = d[d.backbone_alias == alias].pivot_table(
            index="log", columns="n_labels", values="value").reindex(columns=B)
        mean, sd = a.mean(axis=0).values, a.std(axis=0, ddof=0).values
        ok = ~np.isnan(mean)
        ax.fill_between(x[ok], (mean - sd)[ok], (mean + sd)[ok], color=c, alpha=0.15, lw=0)
        n = len(B)
        main = ok.copy()
        main[n - 1] = False          # the log-spaced budgets, in the series' own line style
        ax.plot(x[main], mean[main], color=c, ls=ls, marker=mk, ms=5.5, lw=LW,
                markeredgecolor="white", markeredgewidth=0.7, label=name)
        if ok[n - 2] and ok[n - 1]:  # 1k -> all: off the log scale, so a solid step with a // break mark
            ax.plot(x[n - 2:], mean[n - 2:], color=c, ls=ls, lw=LW, zorder=2)
            ax.text((x[n - 2] + x[n - 1]) / 2, (mean[n - 2] + mean[n - 1]) / 2, "//", color=c,
                    path_effects=[patheffects.withStroke(linewidth=BREAK_STROKE, foreground=c)], **BREAK_KW)
        if ok[n - 1]:
            ax.plot(x[n - 1], mean[n - 1], color=c, marker=mk, ms=5.5, lw=0,
                    markeredgecolor="white", markeredgewidth=0.7)
    ax.text(len(B) - 1.5, 0, "//", color="black", transform=blended_transform_factory(ax.transData, ax.transAxes),
            clip_on=False, path_effects=[patheffects.withStroke(linewidth=BREAK_STROKE, foreground="black")],
            **BREAK_KW)   # break on the x-axis between 1k and all
    ax.set_ylabel(ylab)
    ax.set_xlabel("# cases (log)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(XT)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(loc="best", fontsize=10.5, frameon=True, framealpha=0.95,
                  handlelength=2.2, borderpad=0.6, labelspacing=0.5)


def main():
    per = load()
    missing = [a for _n, a, *_ in SER if a not in set(per.backbone_alias)]
    if missing:
        sys.exit("no rows for arm(s) %s in results/v2_all.csv; collect the grid first" % missing)
    plt.rcParams.update({
        "font.size": PANEL_FS, "axes.labelsize": PANEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    })
    names = []
    for i, (task, cap, ylab) in enumerate(TASKS):
        # fixed canvas and axes box (no tight cropping): every panel prints at exactly the same scale
        fig = plt.figure(figsize=(PANEL_W, PANEL_H))
        ax = fig.add_axes([PAD_L / PANEL_W, PAD_B / PANEL_H, (PANEL_W - PAD_L - PAD_R) / PANEL_W,
                           (PANEL_H - PAD_B - PAD_T) / PANEL_H])
        draw(ax, per, task, ylab, legend=False)
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        over = max(0.0, -tb.x0, -tb.y0, tb.x1 - PANEL_W, tb.y1 - PANEL_H)
        labs = [t.get_window_extent() for t in ax.get_xticklabels() if t.get_text()]
        gap = min(b.x0 - a.x1 for a, b in zip(labs, labs[1:])) / fig.dpi
        print(f"  {task}: text outside canvas {over:.3f} in, smallest x-tick gap {gap:.3f} in")
        fn = "frozen_agg_%s%s" % (task, _SFX)
        fig.savefig(os.path.join(HERE, fn + ".pdf"))
        fig.savefig(os.path.join(HERE, fn + ".png"), dpi=200)
        plt.close(fig)
        names.append((fn, cap, "pretext head" if task in PRETEXT else "unseen task"))
        print("wrote", fn + ".pdf")

    # ---- standalone legend: fills the empty 8th grid slot ----
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=6.5, lw=2.6,
                      markeredgecolor="white", markeredgewidth=0.7, label=n)
               for n, _a, c, ls, mk in SER]
    lf = plt.figure(figsize=(3.3, 2.75))
    lf.legend(handles=handles, loc="center", fontsize=LEGEND_FS, frameon=True, framealpha=0.95,
              handlelength=3.2, borderpad=1.0, labelspacing=0.9)
    lf.savefig(os.path.join(HERE, "frozen_agg_legend%s.pdf" % _SFX), bbox_inches="tight")
    lf.savefig(os.path.join(HERE, "frozen_agg_legend%s.png" % _SFX), dpi=200, bbox_inches="tight")
    plt.close(lf)
    print("wrote frozen_agg_legend%s.pdf" % _SFX)

    # ---- LaTeX subfigure block: 4 + 4 (7 panels, legend in the 8th slot) ----
    L = [r"\begin{figure*}[t]", r"\centering"]
    slots = names + [("frozen_agg_legend" + _SFX, None, None)]   # 8th = legend, no caption
    for i, (fn, cap, tag) in enumerate(slots):
        L.append(r"\begin{subfigure}[t]{0.24\textwidth}")
        L.append(r"  \includegraphics[width=\linewidth]{figures/%s.pdf}" % fn)
        if cap:
            L.append(r"  \caption{%s (%s)}" % (cap, tag))
        L.append(r"\end{subfigure}\hfill" if (i + 1) % 4 != 0 else r"\end{subfigure}")
        if (i + 1) % 4 == 0 and i != len(slots) - 1:
            L.append(r"\\[4pt]")
    L += [
        r"\caption{Label-efficiency curves aggregated over the five held-out logs "
        r"(BPI13, BPI17, BPI20ID, Helpdesk, MIMIC): mean of the raw task metric over logs "
        r"at each label budget, probes trained for up to 100 epochs with early stopping on the "
        r"validation partition, three seeds; bands are $\pm 1$ s.d.\ across logs. "
        r"\emph{Frozen Random}: randomly initialised role encoder and frozen backbone with trained heads; "
        r"\emph{\PFM-Scratch}: the same architecture, randomly initialised and trained end to end on the target log; "
        r"\emph{\PFM-FT}: \PFM{} fine-tuned end to end (role encoder, backbone and head); "
        r"\emph{\PFM}: pretrained frozen backbone with trained heads. For the three MAE tasks each "
        r"log's error is divided by that log's Frozen Random MAE at full supervision, so the curves are "
        r"unitless and comparable across logs. The budget axis is logarithmic; \emph{all} is the full "
        r"training partition (2.7k--20k cases depending on the log), which does not lie on that scale, "
        r"so the last hop is marked with a break symbol (//). "
        r"\emph{Pretext head} marks tasks with a "
        r"matching pretraining objective; \emph{unseen task} marks tasks the backbone was "
        r"never trained for." + (r" \PFM{} and \PFM-FT use the GIN-15 backbone of pre-training seed %d." % PFM_SEED
                                  if PFM_SEED else "") + "}",
        r"\label{fig:agg-raw%s}" % _SFX.replace("_", "-"),
        r"\end{figure*}",
    ]
    tex = os.path.join(HERE, "frozen_agg_panels%s.tex" % _SFX)
    open(tex, "w").write("\n".join(L) + "\n")
    print("wrote", tex)


if __name__ == "__main__":
    main()
