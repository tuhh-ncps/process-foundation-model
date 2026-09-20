"""v2 ablation table: 7 backbone variants, full budget, 5 held-out logs x 3 seeds, 7 tasks.

Variants (all re-pretrained on the leak-free v2 protocol, id_dropout 1.0, unit loss weights, frozen probes,
budget corpus): pfm = GIN role encoder on the 15-feature fingerprint (the model); mlp15 = per-activity MLP on the
fingerprint, no message passing; raw15 = the raw 15-d fingerprint as e(a), no encoder; gin11 = GIN on the
11-feature subset; gin0 = GIN on a constant input (DFG topology only); latent0 = no latent (JEPA) pretraining loss;
norole = no role channel (role_dim 0).
Accuracy-type tasks: mean over logs of the raw metric.  MAE tasks: each log's MAE divided by that log's Random-role
MAE at full budget (as in the figures), then mean over logs.  Prints a markdown table and writes table_ablation_v2.tex.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root
d = pd.read_csv(os.path.join(ROOT, "results", "v2_all.csv"))
d["n_labels"] = d.n_labels.astype(str)
d = d[d.n_labels == "all"]
LOGS = ["helpdesk", "bpi13_incidents", "mimic_transfer", "BPI20ID", "BPI17"]
# Arms as PUBLISHED in Table 6 of the manuscript. The role variants (mlp15/raw15/gin11/gin0/norole) were
# pretrained once, with seed 0. The two arms that were replicated across pretraining seeds appear in the
# submitted table with a specific replica: GIN-15 with the seed-2 backbone (gin15_s2) and the no-latent
# variant with the seed-1 backbone (latent0_s1). Those two rows are therefore NOT seed-matched with each
# other; the seed replication printed below this table gives the matched comparison, and REPRODUCE.md
# ("Mapping the artifact onto the published tables") says what changes under it.
VAR = [("gin15_s2", "GIN-15 (full)"), ("mlp15", "MLP-15"), ("raw15", "raw-15"), ("gin11", "GIN-11"), ("gin0", "GIN-0"),
       ("latent0_s1", "no latent"), ("norole", "no role")]
ACC = [("next_activity", "Next act. (acc)"), ("next_3_activities", "Next-3 (acc)"), ("next_5_activities", "Next-5 (acc)"),
       ("future_activity_set", "Future set (F1)")]
MAE = [("next_time", "Next time (norm. MAE)"), ("remaining_time", "Rem. time (norm. MAE)"), ("remaining_count", "Rem. count (norm. MAE)")]

per = d.groupby(["log", "task", "arm"]).value.mean().unstack("arm")  # per-log seed means
rows = []
for task, name in ACC + MAE:
    vals = {}
    for arm, _ in VAR:
        xs = []
        for lg in LOGS:
            v = per.loc[(lg, task), arm]
            if task in dict(MAE):
                v = v / per.loc[(lg, task), "random_role"]
            xs.append(v)
        vals[arm] = float(np.mean(xs))
    rows.append((name, vals))

hib = {t for t, _ in ACC}
md = ["| Task | " + " | ".join(n for _, n in VAR) + " |", "|---|" + "---|" * len(VAR)]
tex = [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{Backbone ablation on the v2 protocol: frozen probes at full budget, mean over the five held-out logs and three seeds. "
       r"Accuracy-type tasks report the raw metric; MAE tasks report each log's MAE divided by the Random-role MAE on that log (lower is better). "
       r"Variants: GIN-15 is the model; MLP-15 replaces message passing by a per-activity MLP; raw-15 uses the fingerprint itself as $e(a)$; "
       r"GIN-11 uses the 11-feature subset; GIN-0 feeds the GIN a constant input (DFG topology only); no latent drops the latent pretraining loss; "
       r"no role removes the role channel. Best per row in bold.}",
       r"\label{tab:ablation-v2}",
       r"\begin{tabular}{l" + "c" * len(VAR) + "}", r"\toprule",
       "Task & " + " & ".join(n for _, n in VAR) + r" \\", r"\midrule"]
for (task, name), (_, vals) in zip(ACC + MAE, rows):
    best = max(vals.values()) if task in hib else min(vals.values())
    dp = 3 if task in hib else 2
    cells = []
    for arm, _ in VAR:
        s = f"{vals[arm]:.{dp}f}"
        cells.append(("**%s**" % s) if abs(vals[arm] - best) < 1e-9 else s)
    md.append("| " + name + " | " + " | ".join(cells) + " |")
    cells_tex = [(r"\textbf{%s}" % c.strip("*")) if c.startswith("**") else c for c in cells]
    tex.append(name.replace("&", r"\&") + " & " + " & ".join(cells_tex) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print("\n".join(md))
open(os.path.join(HERE, "table_ablation_v2.tex"), "w").write("\n".join(tex) + "\n")
print("\nwrote table_ablation_v2.tex")

# ---------------------------------------------------------------------------------------------
# Same table with raw MAE (days for the time tasks, events for remaining count) instead of the
# ratio to the Random-role MAE: mean over the five logs of each log's raw MAE.  Note that the raw
# mean is dominated by the logs with long durations (BPI20ID, BPI17).  Writes table_ablation_v2_rawmae.tex.
# ---------------------------------------------------------------------------------------------
MAE_RAW = [("next_time", "Next time (MAE, days)"), ("remaining_time", "Rem. time (MAE, days)"), ("remaining_count", "Rem. count (MAE, events)")]
rows_raw = []
for task, name in ACC + MAE_RAW:
    vals = {arm: float(np.mean([per.loc[(lg, task), arm] for lg in LOGS])) for arm, _ in VAR}
    rows_raw.append((name, vals))
md_raw = ["", "Same table, raw MAE (mean over logs of days / events):",
          "| Task | " + " | ".join(n for _, n in VAR) + " |", "|---|" + "---|" * len(VAR)]
tex_raw = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Backbone ablation on the v2 protocol: frozen probes at full budget, mean over the five held-out logs and three seeds. "
           r"Accuracy-type tasks report the raw metric; MAE tasks report the mean over logs of the raw MAE (days for the time tasks, events for the remaining count; lower is better). "
           r"Variants: GIN-15 is the model; MLP-15 replaces message passing by a per-activity MLP; raw-15 uses the fingerprint itself as $e(a)$; "
           r"GIN-11 uses the 11-feature subset; GIN-0 feeds the GIN a constant input (DFG topology only); no latent drops the latent pretraining loss; "
           r"no role removes the role channel. Best per row in bold.}",
           r"\label{tab:ablation-v2-rawmae}",
           r"\begin{tabular}{l" + "c" * len(VAR) + "}", r"\toprule",
           "Task & " + " & ".join(n for _, n in VAR) + r" \\", r"\midrule"]
for (task, name), (_, vals) in zip(ACC + MAE_RAW, rows_raw):
    best = max(vals.values()) if task in hib else min(vals.values())
    dp = 3 if task in hib else 2
    cells = []
    for arm, _ in VAR:
        s_ = f"{vals[arm]:.{dp}f}"
        cells.append(("**%s**" % s_) if abs(vals[arm] - best) < 1e-9 else s_)
    md_raw.append("| " + name + " | " + " | ".join(cells) + " |")
    cells_tex = [(r"\textbf{%s}" % c.strip("*")) if c.startswith("**") else c for c in cells]
    tex_raw.append(name.replace("&", r"\&") + " & " + " & ".join(cells_tex) + r" \\")
tex_raw += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print("\n".join(md_raw))
open(os.path.join(HERE, "table_ablation_v2_rawmae.tex"), "w").write("\n".join(tex_raw) + "\n")
print("\nwrote table_ablation_v2_rawmae.tex")

# per-log raw MAE detail (for the appendix / text)
print("\nPer-log raw MAE (full budget):")
for task, name in MAE_RAW:
    sub_ = per.xs(task, level="task")[[a for a, _ in VAR]].reindex(LOGS).round(2)
    sub_.columns = [n for _, n in VAR]
    print("\n" + name); print(sub_.to_string())

# per-log next-activity detail, for the text
print("\nPer-log next-activity accuracy (full budget):")
sub = per.xs("next_activity", level="task")[[a for a, _ in VAR]].reindex(LOGS).round(3)
sub.columns = [n for _, n in VAR]
print(sub.to_string())


# ---------------------------------------------------------------------------------------------
# Pretraining-seed replication: the full model and the no-latent variant were each pretrained with
# three seeds (arms pfm/gin15_s1/gin15_s2 and latent0/latent0_s1/latent0_s2) and probed on all
# five held-out logs (three evaluation seeds per cell).  Report the 5-log mean per pretraining seed
# and mean +/- sample s.d. (ddof=1) over the three pretraining seeds; same normalisation as above.
# ---------------------------------------------------------------------------------------------
SEED_LOGS = LOGS
PAIRS = [("GIN-15 (full)", ["pfm", "gin15_s1", "gin15_s2"]), ("no latent", ["latent0", "latent0_s1", "latent0_s2"])]
NS = len(PAIRS[0][1])


def log_mean(arm, task):
    xs = []
    for lg in SEED_LOGS:
        v = per.loc[(lg, task), arm]
        if task in dict(MAE):
            v = v / per.loc[(lg, task), "random_role"]
        xs.append(v)
    return float(np.mean(xs))


md2 = ["", "Pretraining-seed replication, five logs, full budget (three pretraining seeds x three evaluation seeds):",
       "| Task | " + " | ".join(f"GIN-15 s{i}" for i in range(NS)) + " | GIN-15 mean ± s.d. | "
       + " | ".join(f"no latent s{i}" for i in range(NS)) + " | no latent mean ± s.d. | Δ (no latent − full) |",
       "|---|" + "---|" * (2 * NS + 3)]
tex2 = [r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Pretraining-seed replication of the latent ablation: the full model and the no-latent variant were each "
        r"pretrained three times with different seeds and probed frozen at full budget on the five held-out logs, three evaluation "
        r"seeds per cell. Entries are the mean over logs, reported as mean $\pm$ s.d.\ over the three pretraining seeds. "
        r"Accuracy-type tasks report the raw metric; MAE tasks the ratio to the Random-role MAE per log (lower is better). "
        r"$\Delta$ is no latent minus full. On every task $|\Delta|$ is at most about two pretraining-seed standard deviations, and its sign is not consistent: dropping the latent loss is marginally better on the activity tasks and marginally worse on the time tasks.}",
        r"\label{tab:ablation-latent-seeds}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Task & GIN-15 (full) & no latent & $\Delta$ \\", r"\midrule"]
for task, name in ACC + MAE:
    dp = 3 if task in hib else 2
    vals = {}
    for label, arms_ in PAIRS:
        xs = [log_mean(a, task) for a in arms_]
        vals[label] = (xs, float(np.mean(xs)), float(np.std(xs, ddof=1)))
    f, l = vals["GIN-15 (full)"], vals["no latent"]
    delta = l[1] - f[1]
    md2.append("| " + name + " | " + " | ".join(f"{x:.{dp}f}" for x in f[0]) + f" | {f[1]:.{dp}f} ± {f[2]:.{dp}f} | "
               + " | ".join(f"{x:.{dp}f}" for x in l[0]) + f" | {l[1]:.{dp}f} ± {l[2]:.{dp}f} | {delta:+.{dp}f} |")
    tex2.append(name.replace("&", r"\&") + f" & {f[1]:.{dp}f} $\\pm$ {f[2]:.{dp}f} & {l[1]:.{dp}f} $\\pm$ {l[2]:.{dp}f} & {delta:+.{dp}f} \\\\")
tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
print("\n".join(md2))
open(os.path.join(HERE, "table_ablation_latent_seeds.tex"), "w").write("\n".join(tex2) + "\n")
print("\nwrote table_ablation_latent_seeds.tex")

# per-log next-activity detail for the replication arms (for the text)
print("\nPer-log next-activity accuracy, replication arms (full budget):")
arms_all = [a for _, arms_ in PAIRS for a in arms_]
sub2 = per.xs("next_activity", level="task")[arms_all].reindex(LOGS).round(3)
print(sub2.to_string())
