"""One PDF per task for the raw-metric aggregate, plus a LaTeX subfigure block.

v2 protocol (leak-free: position_ratio removed, 64-event exclusion, ID-free backbone, budget
corpus). Data: labeleff_data/v2_all.csv (collect_v2.py). Arms: random_role -> Rand.,
pfm_ft -> PFM-FT (PFM fine-tuned end to end), pfm -> PFM (frozen). Mean over the 5 held-out logs,
raw metric, +/-1 s.d. across logs. Each panel is emitted as its own
PDF with NO title and NO legend -- the task name lives in the LaTeX subcaption. The legend
is a standalone frozen_agg_legend.pdf that fills the 8th slot of the 2x4 grid.

Outputs: frozen_agg_<task>.pdf x7, frozen_agg_legend.pdf, and frozen_agg_panels.tex.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, ScalarFormatter

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
# final hop is drawn as a dashed connector instead of a normal step.
JUMP_LS = (0, (2.2, 2.2))
SER = [
    ("Frozen Random", "random_role", "#9a9a9a", "solid", "^"),
    ("PFM-FT", "pfm_ft", "#E1912F", "solid", "s"),
    ("PFM", "pfm", "#2b6cb8", "solid", "o"),
]
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
        ax.plot(x[main], mean[main], color=c, ls=ls, marker=mk, ms=5, lw=1.7,
                markeredgecolor="white", markeredgewidth=0.7, label=name)
        if ok[n - 2] and ok[n - 1]:  # 1k -> all: off the log scale, so a dashed connector
            ax.plot(x[n - 2:], mean[n - 2:], color=c, ls=JUMP_LS, lw=1.7, zorder=2)
        if ok[n - 1]:
            ax.plot(x[n - 1], mean[n - 1], color=c, marker=mk, ms=5, lw=0,
                    markeredgecolor="white", markeredgewidth=0.7)
    ax.axvline(len(B) - 1.5, color="#c4c4c4", lw=0.8, ls=(0, (1, 3)), zorder=1)
    ax.set_ylabel(ylab)
    ax.set_xlabel("labelled cases (log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(XT)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(loc="best", fontsize=10.5, frameon=True, framealpha=0.95,
                  handlelength=2.2, borderpad=0.6, labelspacing=0.5)


def main():
    per = load()
    plt.rcParams.update({
        "font.size": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    })
    names = []
    for i, (task, cap, ylab) in enumerate(TASKS):
        fig, ax = plt.subplots(figsize=(3.3, 2.75))
        draw(ax, per, task, ylab, legend=False)
        fig.tight_layout()
        fn = "frozen_agg_%s" % task
        fig.savefig(os.path.join(HERE, fn + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(HERE, fn + ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        names.append((fn, cap, "pretext head" if task in PRETEXT else "unseen task"))
        print("wrote", fn + ".pdf")

    # ---- standalone legend: fills the empty 8th grid slot ----
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=6, lw=1.9,
                      markeredgecolor="white", markeredgewidth=0.7, label=n)
               for n, _a, c, ls, mk in SER]
    lf = plt.figure(figsize=(3.3, 2.75))
    lf.legend(handles=handles, loc="center", fontsize=13, frameon=True, framealpha=0.95,
              handlelength=2.6, borderpad=1.0, labelspacing=0.9)
    lf.savefig(os.path.join(HERE, "frozen_agg_legend.pdf"), bbox_inches="tight")
    lf.savefig(os.path.join(HERE, "frozen_agg_legend.png"), dpi=200, bbox_inches="tight")
    plt.close(lf)
    print("wrote frozen_agg_legend.pdf")

    # ---- LaTeX subfigure block: 4 + 4 (7 panels, legend in the 8th slot) ----
    L = [r"\begin{figure*}[t]", r"\centering"]
    slots = names + [("frozen_agg_legend", None, None)]   # 8th = legend, no caption
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
        r"\emph{\PFM-FT}: \PFM{} fine-tuned end to end (role encoder, backbone and head); "
        r"\emph{\PFM}: pretrained frozen backbone with trained heads. For the three MAE tasks each "
        r"log's error is divided by that log's Frozen Random MAE at full supervision, so the curves are "
        r"unitless and comparable across logs. The budget axis is logarithmic; \emph{all} is the full "
        r"training partition (2.7k--20k cases depending on the log), which does not lie on that scale, "
        r"so the last hop is drawn as a dashed connector. "
        r"\emph{Pretext head} marks tasks with a "
        r"matching pretraining objective; \emph{unseen task} marks tasks the backbone was "
        r"never trained for.}",
        r"\label{fig:agg-raw}",
        r"\end{figure*}",
    ]
    tex = os.path.join(HERE, "frozen_agg_panels.tex")
    open(tex, "w").write("\n".join(L) + "\n")
    print("wrote", tex)


if __name__ == "__main__":
    main()
