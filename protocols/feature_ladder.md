# Protocol: fingerprint feature-budget ladder (GIN-k, k = 0…15)

Status: **FROZEN** at the commit that first adds this file. Any later change is recorded as a dated entry
under *Amendments* at the end; the body is never edited in place.

**Question.** How does the frozen PFM representation change as more fingerprint descriptors are
available, when descriptors are added in a fixed order derived from pretraining data only?

**Scope.** One pretraining seed (seed 0); next-activity prediction; full labelled budget; five held-out
logs; three downstream evaluation seeds. Run IDs: r31 (pretraining), r32 (evaluation).

Descriptor letters (Table 1 order, used for tie-breaking):
A PageRank · B Betweenness · C Self-loop prob. · D Case-start prob. · E Case-end prob. ·
F Case coverage · G Repetition rate · H Predecessor entropy · I Successor entropy ·
J In-gap median · K In-gap spread · L Out-gap median · M Out-gap spread · N Mean position · O Position spread.

---

## Phase A — frozen feature order (pretraining data only)

**A1. Fingerprints.** Six pretraining logs only: BPI12, BPI19, BPI18, Road Traffic, BPI11, Hospital
Billing. For each log, use exactly the Phase-1b training traces: the log read as in its
`configs/data/<name>.yaml` (bpi12_solo, bpi19, bpi18, road_traffic, hospital_billing, bpi11), stripped to
control flow, cases with ≥2 events, 70/15/15 chronological split, 70% training partition, no trace caps.
Fingerprints are computed **per log** (each log's own vocabulary and directly-follows graph), one
15-column matrix over that log's activities. No held-out evaluation log is read. Phase A runs as a
CPU job in the project container on the cluster; the artifact is copied back and committed.

**A2. Per-log correlation.** Standardise each descriptor within each log; compute R_ℓ (15×15).
A descriptor that is constant within a log gets correlation 0 with every other descriptor and 1 on the
diagonal for that log.

**A3. Log balancing.** R̄ = (1/6) Σ_ℓ R_ℓ. Record the condition number of R̄.

**A4. Primary greedy ordering.** Start with S = ∅; at each step add the f ∉ S maximising
J(S ∪ {f}) = tr(R̄_{·S} R̄_{SS}^+ R̄_{S·}), with R̄_{SS}^+ the Moore–Penrose pseudo-inverse at
rcond = 1e-10. Exact ties are resolved by the fixed A–O order above. J(S)/15 is the fraction of
standardised descriptor variance recovered by one linear reconstruction shared across the six
pretraining logs, logs weighted equally.

**A4a. Invariants of R̄ and J(S) (asserted; any violation stops Phase A and nothing is frozen).**
With ε = 1e-8:
1. R̄ is symmetric (max |R̄ − R̄ᵀ| ≤ 1e-12), has unit diagonal (max |diag R̄ − 1| ≤ 1e-12) and is
   positive semidefinite (smallest eigenvalue ≥ −1e-10).
2. J(∅) = 0 and J(F_15) = tr(R̄) = 15 (within ε).
3. For every evaluated S: |S| − ε ≤ J(S) ≤ 15 + ε.
4. Per-descriptor recovered variance lies in [−ε, 1 + ε], and equals 1 (within ε) for every selected
   descriptor.
5. Monotonicity along the greedy path: ΔJ_k = J(F_k) − J(F_{k−1}) ≥ −ε for k = 1…15.
6. Order invariance: J(S) is unchanged (within ε) under a random permutation of the indices in S,
   checked at every greedy step.
No diminishing-returns property is assumed or asserted.

**A4b. Pseudo-inverse robustness (pretraining data only, before freezing).** Recompute the ordering
with rcond ∈ {1e-8, 1e-10, 1e-12}. Report whether each ordering is identical to the primary; if not,
the first differing position, Kendall's τ, and J(S)/15. The primary ordering always uses 1e-10;
differences are reported as a conditioning note and never change the primary.

**A5. Leave-one-log-out sensitivity.** For each pretraining log ℓ, R̄_{−ℓ} = (1/5) Σ_{ℓ′≠ℓ} R_ℓ′,
with the same standardisation, constant-descriptor rule, pseudo-inverse and tie-break; derive a greedy
ordering from each. Compare each with the primary using Kendall's τ over the 15 ranks (all six values,
minimum, and the log giving it), top-k overlap |F_k ∩ F_k^{(−ℓ)}|/k for k = 3, 5, 10 (chance k/15),
the relative variance retained J_R̄(F_k^{(−ℓ)})/J_R̄(F_k) on the six-log R̄, and the step at which
Case-start or Case-end probability enters. This analysis tests whether the ordering depends strongly on
any single pretraining process and is never used to select or modify the primary ordering.

**A6. Frozen artifact** `results/feature_ladder.json`: the order f_1…f_15 (name, code index, letter);
ΔJ, J(S), J(S)/15 per step; all settings (logs, trace settings, constant-descriptor rule,
pseudo-inverse and rcond, tie-break, cond(R̄)); the A4a invariant values; A4b and A5 results; the
primary-order step where Case-start/Case-end probability enters; script git hash.

**A7. Freeze gate.** `scripts/feature_ladder.py` and `results/feature_ladder.json` are committed before
any GPU job. Nothing downstream may reorder, drop or tune descriptors.

## Phase A′ — zero-masking (fixed model size)

**A′1.** Every k uses the full `Linear(15, 64)` role-encoder input. Selected descriptors keep their
original values; unselected descriptors are set to 0.

**A′2.** The mask is applied inside the role embedder, before feature dropout — never on the graph
fingerprints, which also supply the Phase-1a start/end role labels.

**A′3.** Config `model.role_feature_mask` (kept code indices; default all 15), rebuilt from the config
as a non-persistent buffer so existing checkpoints load strictly.

**A′4. Unit tests (must pass before Phase B).** (1) all-ones mask with the same weights is bit-identical
to GIN-15; (2) masked input columns receive exactly zero gradient; (3) changing values in masked
columns leaves the embedding unchanged.

**A′5.** k = 15 is GIN-15 (backbone `backbone-20260906-153102-multi-none-v2-gin15-17fc3c`, reused);
k = 0 is trained new. Earlier shrunk-input ablations (GIN-11, GIN-0) are not directly comparable.

## Phase B — pretraining (r31)

**B1.** M_k = first k code indices of the frozen order, k = 0…14.

**B2. Phase 1a**, 15 role encoders: `role/frozen.yaml`, `model.role_feature_mask=M_k`; all else
identical to GIN-15 (GIN, role_dim 64, 120 epochs, same losses and weights, Sepsis + Receipt
validation, seed 0).

**B3. Phase 1b**, 15 backbones: recipe identical to the released GIN-15 (six pretraining logs,
`freeze_role=true`, `id_dropout=1.0`, act/time/remaining/latent weights 1, outcome weight 0.3,
15 epochs, seed 0), `role_init_from` the matching encoder; tag `-v2-fb{k:02d}`; pinned full H200.

**B4. Gate.** All manifests completed with checkpoints; each manifest records the expected mask; losses
finite and comparable to GIN-15.

## Phase C — evaluation (r32)

**C1.** Extend the cached-feature evaluator with `--backbone`, `--eval-seed`, `--tasks`; the mask is read
from each backbone manifest.

**C2. Validation gate (all must pass before any ladder evaluation; thresholds are not relaxed after
the fact).**
1. *Mask plumbing.* For GIN-15 on each of the five held-out logs, cached event states computed with an
   explicit all-ones `role_feature_mask` and with no mask key are bit-identical (max |diff| = 0).
2. *Agreement with the standard path.* GIN-15 (seed-0 backbone) through the extended cached evaluator,
   5 logs × eval seeds {0, 1, 2}, compared with the existing `pfm` next-activity rows in
   `results/v2_all.csv`: |Δ accuracy| ≤ 0.010 for every (log, seed), and |Δ| ≤ 0.005 for the five-log
   mean of each seed and for the overall mean.
3. *Repeatability.* One configuration (Helpdesk, eval seed 0) run twice through the cached evaluator:
   |Δ accuracy| ≤ 0.002.
Failure: stop, fix, and rerun C2 in full.

**C3.** Evaluate 16 backbones (k = 0…15), all through the same cached path: BPI17, BPI20ID, MIMIC,
BPI13, Helpdesk; v2 split; cases with 2–64 events; full labelled budget; role corpus = full training
partition; next activity; eval seeds {0, 1, 2} for every k; frozen role encoder and backbone; linear
head only (AdamW 1e-3, ≤100 epochs, patience 10, best-validation restore).

**C4.** Frozen Random reference: the existing full-budget `random_role` next-activity result, drawn as a
horizontal line and never called k = 0.

## Phase D — aggregation and reporting

**D1.** Per (k, log): ā_{k,ℓ} = mean over the three eval seeds; s²_{k,ℓ} = their variance.

**D2.** Per k: μ_k = (1/5) Σ_ℓ ā_{k,ℓ}.

**D3. Downstream-seed variation (primary uncertainty; downstream-head stochasticity only).** For each
evaluation seed, average accuracy across the five held-out logs; report the mean and SD of these three
five-log means. This seed-wise SD is the plotted error bar. The pooled estimate
√(Σ_ℓ s²_{k,ℓ})/5 and the between-log SD are reported in the summary table only.

**D3b. Pretraining reference scale.** σ_15 = 0.0058: SD of the five-log mean next-activity accuracy
across the three existing GIN-15 pretraining seeds (`17fc3c`, `ddeadf`, `bfb92b`). Used only as a
descriptive reference line and in the predefined k_near criterion; never propagated as an uncertainty
for other feature budgets.

**D4. Primary summary (fixed before observing results).** μ_15 is the ladder's own k = 15 point (same
pretraining seed, pipeline and evaluation seeds as every k).
k_near = min{ k : μ_j ≥ μ_15 − σ_15 for all j ≥ k }.
k_near is descriptive — the smallest budget from which all larger budgets lie in the near-full-performance
region — not a statistical equivalence or saturation test. μ_15 is a single draw (the lowest of the three
existing GIN-15 seeds), so the threshold is slightly lenient; σ_15 is estimated from three runs and serves
only as a scale. Pre-registered sensitivity: k_near with the three-seed GIN-15 mean (0.7303) as μ_15.
No other thresholds. The overall trend is described separately; differences between adjacent budgets are
not interpreted.

**D5. Paired differences.** For each k, Δ_{k,ℓ} = ā_{k,ℓ} − ā_{15,ℓ} on each held-out log after averaging
downstream seeds. Report the five-log mean Δ_k and the paired 95% t-interval across logs,
Δ_k ± t_{4,0.975}·SD_ℓ(Δ_{k,ℓ})/√5 with t_{4,0.975} = 2.776. The interval describes cross-log variation
conditional on the single pretrained backbone used at each budget. The five per-log differences are
shown beside each interval; the 15 intervals are unadjusted and descriptive; Δ_15 ≡ 0.

**D6. Files.** `results/feature_ladder_next_activity.csv` (one row per k, log, eval seed);
`results/feature_ladder_summary.csv` (per k: μ_k, seed-wise mean and SD, pooled SD, between-log SD, Δ_k,
t-interval, five per-log Δ, n logs, n eval seeds, k_near primary and sensitivity, σ_15 and source).

**D7. Figure** `figures/feature_ladder_plot.py`. Panel (a): μ_k ± seed-wise SD vs k; horizontal line at
μ_15 − σ_15; Frozen Random line; markers at k_near and where Case-start/Case-end probability enters.
Panel (b): Δ_k with paired 95% t-intervals, five per-log points, zero line. Secondary: J(S)/15.

## Phase E — documentation

`hpc/README.md` rows r31/r32; `REPRODUCE.md` commands per phase. Limitations stated: single pretraining
seed (exploratory, no propagated pretraining uncertainty); greedy ordering; Phase-1a labels derived from
Case-start/Case-end probability; next activity only; D5 intervals conditional on single backbones.

---

## Amendments

### A1 — 2026-09-15 — C2.2 waived; cached evaluation retained

Decision: T. Tran, after the C2 gate returned FAILED (commit d33be75, job 4546).

- C2.1 (mask plumbing: max |diff| = 0 on all five logs) and C2.3 (repeatability: identical accuracy) passed
  and remain required.
- C2.2 (agreement of the cached evaluator with the standard label-efficiency path on GIN-15) failed:
  per-run |Δ accuracy| up to 0.043; per-log three-seed mean differences −0.030 (Helpdesk), +0.010 (MIMIC),
  +0.008 (BPI17), −0.003 (BPI13), −0.002 (BPI20ID); overall −0.003. Attributed to the cached path's plain
  shuffled batches and single seeding versus the standard path's length-bucketed batches and per-probe
  reseeding.
- The feature-budget ladder is an ablation whose budgets are compared only with each other. C3 therefore
  proceeds with the cached evaluator for every k, as specified. The C2 thresholds are unchanged; the C2.2
  failure and its magnitude are reported with the results.
- Consequences for Phase D: μ_k, the D3 seed-wise SD, the D4 primary k_near (μ_15 is the ladder's own
  cached k = 15 point) and all D5 paired differences are within one evaluation pipeline. The Frozen Random
  reference (C4), σ_15 (D3b) and the D4 sensitivity value μ_15 = 0.7303 come from the standard path; they
  are labelled as cross-pipeline references, and absolute ladder accuracies are not compared numerically
  with the main result tables.
- Implementation: `protocols/feature_ladder_waivers.json` lists C2.2 under A1. The C2 verdict is
  `PASSED_WITH_WAIVER` only if every non-waived check passes; any other failure still blocks C3.

### A2 — 2026-09-16 — exploratory extension to six further tasks

Decision: T. Tran, while C3 (next activity) was finishing and before its results were aggregated.

- The same 16 frozen backbones (k = 0…15), five held-out logs, evaluation seeds {0, 1, 2} and full labelled
  budget are additionally evaluated on next 3 activities, next 5 activities, future activity set, next event
  time, remaining time and remaining event count (run r33).
- Evaluation uses the cached evaluator of C1, extended to cache the `next_time` target (NaN-padded, masked by
  its head). The other five tasks derive their targets from the already cached `next_activity` and
  `padding_mask`, exactly as their heads do on the standard path. The A1 pipeline caveat applies: numbers are
  compared across budgets within a task, not with the main result tables.
- A smoke run (Helpdesk, seed 0, GIN-15, all six tasks) must complete before the grouped jobs start.
- This extension is exploratory. It adds no primary claim and does not change A–D for next activity. Per task it
  reports the D2 mean, the D3 seed-wise SD and the D5 paired per-log differences to k = 15; no per-task
  k_near thresholds are defined, and MAE tasks are reported in raw units per log.
