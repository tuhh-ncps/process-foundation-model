"""Table 5 of the manuscript: full training budget on the five held-out logs, seven tasks.

Reads results/v2_all.csv (our arms) and results/sota_agg_data_raw.csv (the baseline replays, written by
figures/sota_agg_plot.py). Writes table5.tex / table5.md next to this file.

Arms as PUBLISHED. PFM is the frozen probe on the seed-2 pretrained backbone (arm `pfm_s2`); PFM-FT is the
end-to-end finetune of the seed-0 backbone (arm `pfm_ft`). The two columns therefore come from different
pretraining seeds - see REPRODUCE.md, "Differences from the submitted manuscript". Every cell is a mean over
three evaluation seeds; accuracy and F1 in %, MAE in days (remaining count in events), rounded as printed.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PFM_ARM, FT_ARM = "pfm_s2", "pfm_ft"
LOGS = [("bpi13_incidents", "BPI13"), ("BPI17", "BPI17"), ("BPI20ID", "BPI20ID"),
        ("helpdesk", "Helpdesk"), ("mimic_transfer", "MIMIC-5k")]
ACC = ["next_activity", "next_3_activities", "next_5_activities", "future_activity_set"]
MAE = ["next_time", "remaining_time", "remaining_count"]
TASKS = ACC + MAE
HEAD = {"next_activity": "Next act. (Acc, %)", "next_3_activities": "Next-3 (Acc, %)",
        "next_5_activities": "Next-5 (Acc, %)", "future_activity_set": "Future set (F1, %)",
        "next_time": "Next time (MAE, d)", "remaining_time": "Rem. time (MAE, d)",
        "remaining_count": "Rem. count (MAE, ev.)"}
# FM-v2 supports only its two native tasks; the baseline replays live in the aggregated raw CSV.
FMV2_TASKS = {"next_activity", "remaining_time"}
BASE_NAME = {"MIMIC-5k": "MIMIC"}

d = pd.read_csv(os.path.join(ROOT, "results", "v2_all.csv"))
d = d[(d.n_labels.astype(str) == "all") & d.seed.isin([0, 1, 2])]
ours = d.groupby(["arm", "log", "task"]).value.mean()
base = pd.read_csv(os.path.join(ROOT, "results", "sota_agg_data_raw.csv")).set_index(["log", "task"])


def cell(task, value):
    if value is None or pd.isna(value):
        return "--"
    return f"{100 * value:.0f}" if task in ACC else f"{value:.1f}"


rows = []
for key, name in LOGS:
    row = {"log": name}
    for t in TASKS:
        row[f"{t}/PFM"] = cell(t, ours[(PFM_ARM, key, t)])
        row[f"{t}/PFM-FT"] = cell(t, ours[(FT_ARM, key, t)])
        bname = BASE_NAME.get(name, name)   # the aggregate CSV labels the subset "MIMIC"
        b = base.loc[(bname, t)] if (bname, t) in base.index else None
        row[f"{t}/SuTraN"] = cell(t, None if b is None else b["SuTraN"])
        if t in FMV2_TASKS:
            row[f"{t}/FM-v2 Proto"] = cell(t, None if b is None else b["FM-v2 Proto"])
            row[f"{t}/FM-v2 kNN"] = cell(t, None if b is None else b["FM-v2 kNN"])
    rows.append(row)

cols = [c for t in TASKS for c in
        [f"{t}/PFM", f"{t}/PFM-FT"] + ([f"{t}/FM-v2 Proto", f"{t}/FM-v2 kNN"] if t in FMV2_TASKS else [])
        + [f"{t}/SuTraN"]]
md = ["| Log | " + " | ".join(f"{HEAD[c.split('/')[0]].split(' (')[0]} {c.split('/')[1]}" for c in cols) + " |",
      "|" + "---|" * (len(cols) + 1)]
for r in rows:
    md.append("| " + r["log"] + " | " + " | ".join(r[c] for c in cols) + " |")
open(os.path.join(HERE, "table5.md"), "w").write("\n".join(md) + "\n")

tex = [r"% Table 5: full training budget, five held-out logs x seven tasks.",
       "%% PFM = arm %s (seed-2 backbone, frozen); PFM-FT = arm %s (seed-0 backbone, finetuned)." % (PFM_ARM, FT_ARM),
       r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
       "Log & " + " & ".join(c.replace("/", " ") for c in cols) + r" \\", r"\midrule"]
tex += ["%s & %s \\\\" % (r["log"], " & ".join(r[c] for c in cols)) for r in rows]
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "table5.tex"), "w").write("\n".join(tex) + "\n")

print("\n".join(md))
print("\nwrote table5.md / table5.tex")
