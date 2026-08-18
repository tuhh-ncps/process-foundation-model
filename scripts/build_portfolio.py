"""Generate docs/dataset_portfolio.md from the profiler stats + curated domain/label metadata.

Reads data/very-raw/_portfolio_stats.json (produced by profile_datasets.py) and joins it with
a curated table (domain from the manifest, label availability, and a train/eval recommendation).
Emits a single markdown portfolio. Regenerate after any rescan:

  uv run python scripts/build_portfolio.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "data/very-raw/_portfolio_stats.json"
OUT = ROOT / "docs/dataset_portfolio.md"

# Curated metadata. Key = substring that uniquely matches a parent-dir name.
# role: PRETRAIN (SSL backbone corpus) | EVAL (downstream benchmark w/ clean labels)
#       | BOTH | HOLDOUT (cross-domain transfer eval) | CAUTION (special regime) | EXCLUDE
META: dict[str, dict] = {
    "BPI Challenge 2012": dict(
        short="BPI'12 Loan",
        domain="Banking / Finance",
        use="Consumer loan application",
        role="BOTH",
        labels="Outcome (A_APPROVED / A_DECLINED / A_CANCELLED); next-activity",
        note="Classic benchmark; already wired (bpi12.yaml). Strong for both pretrain volume and outcome eval.",
    ),
    "BPI Challenge 2013, closed": dict(
        short="BPI'13 Closed",
        domain="IT Service Mgmt",
        use="Closed problem management",
        role="EVAL",
        labels="Next-activity; problem resolution (few activities)",
        note="Only 3-4 activities — trivial control-flow; useful as a small eval, weak for pretraining.",
    ),
    "BPI Challenge 2013, incidents": dict(
        short="BPI'13 Incidents",
        domain="IT Service Mgmt",
        use="Incident management (VINST)",
        role="EVAL",
        labels="Next-activity; push-to-front / ping-pong behaviour",
        note="4 activities but rich lifecycle/resource. Good compact ITSM eval.",
    ),
    "BPI Challenge 2013, open": dict(
        short="BPI'13 Open",
        domain="IT Service Mgmt",
        use="Open problem management",
        role="EVAL",
        labels="Next-activity",
        note="Very small (819 cases, 3 acts). Minor eval only.",
    ),
    "BPI Challenge 2015 Municipality 1": dict(
        short="BPI'15-1",
        domain="Public Administration",
        use="Building permit applications",
        role="CAUTION",
        labels="Next-activity; 5 municipalities enable domain-shift transfer",
        note="~400 activities / ~1k cases: sparse, open-vocab regime. Excellent role-channel / open-vocab stress test; the 5 municipalities are a natural transfer suite.",
    ),
    "BPI Challenge 2015 Municipality 2": dict(
        short="BPI'15-2",
        domain="Public Administration",
        use="Building permit applications",
        role="CAUTION",
        labels="Next-activity",
        note="See BPI'15-1; municipality variant.",
    ),
    "BPI Challenge 2015 Municipality 3": dict(
        short="BPI'15-3",
        domain="Public Administration",
        use="Building permit applications",
        role="CAUTION",
        labels="Next-activity",
        note="See BPI'15-1; municipality variant.",
    ),
    "BPI Challenge 2015 Municipality 4": dict(
        short="BPI'15-4",
        domain="Public Administration",
        use="Building permit applications",
        role="CAUTION",
        labels="Next-activity",
        note="See BPI'15-1; municipality variant.",
    ),
    "BPI Challenge 2015 Municipality 5": dict(
        short="BPI'15-5",
        domain="Public Administration",
        use="Building permit applications",
        role="CAUTION",
        labels="Next-activity",
        note="See BPI'15-1; municipality variant.",
    ),
    "BPI Challenge 2018": dict(
        short="BPI'18 CAP",
        domain="Finance / Public Admin",
        use="Agricultural subsidy (CAP) payments",
        role="PRETRAIN",
        labels="Outcome (rejected/approved); year-based drift",
        note="Largest real log here (~2.5M events). Prime pretraining fuel; rich case attributes.",
    ),
    "BPI_Challenge_2019": dict(
        short="BPI'19 PO",
        domain="Procurement / Finance",
        use="Purchase-order handling",
        role="PRETRAIN",
        labels="Next-activity; PO-line item flow",
        note="Very large (~1.6M events, ~250k cases). Prime pretraining fuel; strong finance/procurement signal.",
    ),
    "Hospital Billing": dict(
        short="Hospital Billing",
        domain="Healthcare / Finance",
        use="Billing of hospital services",
        role="BOTH",
        labels="Outcome (billed/closed/reopened); next-activity",
        note="Large (~450k events) + clean healthcare-finance outcome. Good pretrain + eval.",
    ),
    "Real-life event logs - Hospital log": dict(
        short="BPI'11 Hospital",
        domain="Healthcare",
        use="Dutch academic hospital (gynaecology oncology)",
        role="BOTH",
        labels="Diagnosis/treatment codes; next-activity",
        note="BPI Challenge 2011. Very high activity count, rich medical attributes. Good healthcare pretrain + eval.",
    ),
    "Road Traffic Fine Management": dict(
        short="Road Traffic",
        domain="Public Administration",
        use="Traffic-fine collection",
        role="BOTH",
        labels="Outcome (fine paid vs sent-to-credit-collection) — canonical outcome benchmark",
        note="Large (~560k events), short well-structured cases, canonical outcome-prediction benchmark. Excellent eval AND pretrain.",
    ),
    "Sepsis Cases": dict(
        short="Sepsis",
        domain="Healthcare",
        use="ER pathway for sepsis patients",
        role="EVAL",
        labels="Outcome (ICU admission / return-to-ER / release type) — already wired (sepsis.yaml)",
        note="Small (1050 cases) but the flagship healthcare outcome benchmark. Keep as eval / cross-domain holdout.",
    ),
    "Activities of daily living": dict(
        short="ADL Smart-home",
        domain="Smart Home / Healthcare",
        use="Sensor-derived daily activities",
        role="EXCLUDE",
        labels="Activity recognition (small)",
        note="8 tiny logs (6-43 cases each). Too small to train or benchmark meaningfully; keep only as a curiosity / tiny transfer probe.",
    ),
    "Apache Commons Crypto": dict(
        short="Apache Crypto",
        domain="Software Engineering",
        use="Instrumented method-call execution",
        role="CAUTION",
        labels="Method-call next-event (deep recursion)",
        note="Software-execution regime: 1-3 cases but ~242k events EACH. Unusable case-level; needs windowing. OOD stress test only.",
    ),
    "JUnit 4.12 Software Event Log": dict(
        short="JUnit",
        domain="Software Engineering",
        use="Instrumented unit-test execution",
        role="CAUTION",
        labels="Method-call next-event",
        note="Software-execution regime; deep call traces. OOD stress / windowed pretraining only.",
    ),
    "NASA Crew Exploration Vehicle": dict(
        short="NASA CEV",
        domain="Software Eng / Aerospace",
        use="Instrumented statechart test execution",
        role="CAUTION",
        labels="Statechart transition next-event",
        note="Software-execution regime. OOD stress test only.",
    ),
    "Statechart Workbench": dict(
        short="Statechart WB",
        domain="Software Eng / Process Mining",
        use="Instrumented ProM alignment execution",
        role="CAUTION",
        labels="Method-call next-event",
        note="Software-execution regime. OOD stress test only.",
    ),
    "Process Discovery Contest": dict(
        short="PDC (synthetic)",
        domain="Synthetic Benchmark",
        use="Process-discovery competition logs",
        role="EXCLUDE",
        labels="Ground-truth classification (fitting vs non-fitting traces)",
        note="Synthetic, homogeneous, hundreds of generated variants per year. Exclude from real-data pretraining; optionally a controlled synthetic benchmark for discovery / classification with provided ground truth.",
    ),
}

ROLE_BADGE = {
    "PRETRAIN": "🟢 Pretrain",
    "EVAL": "🔵 Eval",
    "BOTH": "🟢🔵 Both",
    "HOLDOUT": "🟣 Holdout",
    "CAUTION": "🟡 Special",
    "EXCLUDE": "⚪ Exclude",
}


def meta_for(dirname: str) -> dict:
    for k, v in META.items():
        if k.lower() in dirname.lower():
            return v
    return dict(short=dirname, domain="?", use="?", role="?", labels="?", note="")


def fnum(n) -> str:
    return f"{n:,}" if isinstance(n, int | float) else str(n)


def main() -> None:
    stats = json.loads(STATS.read_text())
    rows = []
    for dirname, s in stats.items():
        m = meta_for(dirname)
        files = [f for f in s.get("files", []) if "n_cases" in f]
        # representative log = the one with the most cases (avoids pathological single-trace files)
        rep = max(files, key=lambda f: f.get("n_cases", 0)) if files else {}
        if s.get("sampled"):
            # PDC: only a few of N logs scanned -> report per-log typical + ×count, never a false sum
            n_cases = rep.get("n_cases", 0)
            n_events = rep.get("n_events", 0)
            acts = rep.get("n_activities", 0)
            scale_suffix = f" ×{s['n_log_files']}"
        elif s.get("is_collection"):
            n_cases = sum(f.get("n_cases", 0) for f in files)
            n_events = sum(f.get("n_events", 0) for f in files)
            acts = rep.get("n_activities", 0)
            scale_suffix = ""
        else:
            n_cases = rep.get("n_cases", 0)
            n_events = rep.get("n_events", 0)
            acts = rep.get("n_activities", 0)
            scale_suffix = ""
        rows.append(
            dict(
                dirname=dirname,
                m=m,
                s=s,
                f0=rep,
                n_cases=n_cases,
                n_events=n_events,
                acts=acts,
                scale_suffix=scale_suffix,
            )
        )

    # collapse the 10 yearly Process-Discovery-Contest collections into one family row
    import re

    pdc = [r for r in rows if "process discovery" in r["dirname"].lower()]
    if pdc:
        rows = [r for r in rows if "process discovery" not in r["dirname"].lower()]
        tot_logs = sum(r["s"].get("n_log_files", 0) for r in pdc)
        tot_mb = round(sum(r["s"].get("total_mb", 0) for r in pdc), 1)
        amin, amax = min(r["acts"] for r in pdc), max(r["acts"] for r in pdc)
        rep = max(pdc, key=lambda r: r["acts"])["f0"]
        pdc_years_tbl = sorted(
            (
                (mm.group(), r["s"].get("n_log_files", 0), r["acts"], r["n_cases"])
                for r in pdc
                if (mm := re.search(r"20\d\d", r["dirname"]))
            ),
            key=lambda t: t[0],
        )
        rows.append(
            dict(
                dirname=f"Process Discovery Contest 2016–2025 ({len(pdc)} yearly collections)",
                m=META["Process Discovery Contest"],
                f0=rep,
                s={
                    "is_collection": True,
                    "sampled": True,
                    "n_log_files": tot_logs,
                    "total_mb": tot_mb,
                },
                n_cases="~700–1,000",
                n_events=rep.get("n_events", 0),
                acts=f"{amin}–{amax}",
                scale_suffix=f"/log ×{tot_logs}",
                pdc_years=pdc_years_tbl,
            )
        )

    # order: pretrain-worthy real logs first (by events desc), then eval, then special/exclude
    order = {"PRETRAIN": 0, "BOTH": 1, "EVAL": 2, "HOLDOUT": 3, "CAUTION": 4, "EXCLUDE": 5, "?": 6}
    rows.sort(
        key=lambda r: (
            order.get(r["m"]["role"], 9),
            -(r["n_events"] if isinstance(r["n_events"], int) else 0),
        )
    )

    L: list[str] = []
    L.append("# Dataset Portfolio — `data/very-raw/`")
    L.append("")
    L.append(
        "Auto-generated by `scripts/build_portfolio.py` from `scripts/profile_datasets.py` output."
    )
    L.append("Stats are computed over the extracted `.xes` logs (streaming scan). Domains from the")
    L.append("`process_mining_ready_datasets` manifest. **Role** is our recommended use in the")
    L.append(
        "pipeline: 🟢 Pretrain backbone · 🔵 Eval downstream · 🟣 Holdout (cross-domain) · "
        "🟡 Special regime · ⚪ Exclude."
    )
    L.append("")
    L.append("## Summary table")
    L.append("")
    L.append(
        "| Dataset | Domain | Role | Cases | Events | Acts | Med len | Variants | Timespan | Note |"
    )
    L.append("|---|---|---|--:|--:|--:|--:|--:|---|---|")
    for r in rows:
        m, s, f0 = r["m"], r["s"], r["f0"]
        span = (
            f"{f0.get('time_start', '?')}→{f0.get('time_end', '?')}"
            if f0.get("has_timestamps")
            else "—"
        )
        vr = f0.get("n_variants", "?")
        medlen = f0.get("case_len", {}).get("median", "?")
        coll = " *(collection)*" if s.get("is_collection") else ""
        suf = r["scale_suffix"]
        L.append(
            f"| **{m['short']}**{coll} | {m['domain']} | {ROLE_BADGE.get(m['role'], m['role'])} "
            f"| {fnum(r['n_cases'])}{suf} | {fnum(r['n_events'])}{suf} | {fnum(r['acts'])} "
            f"| {medlen} | {fnum(vr)} | {span} | {m['note'].split('. ')[0].rstrip('.')}. |"
        )
    L.append("")

    pretrain_events = sum(
        r["n_events"]
        for r in rows
        if r["m"]["role"] in ("PRETRAIN", "BOTH") and isinstance(r["n_events"], int)
    )

    # strategy section
    L.append("## Recommended use: training vs evaluation")
    L.append("")
    L.append(
        "The guiding split is **learn general control-flow/time structure from large, real, "
        "timestamped, multi-domain logs; measure downstream transfer on clean-label benchmarks "
        "that are held out of pretraining.**"
    )
    L.append("")
    L.append("### 🟢 Pretraining corpus (SSL backbone)")
    L.append(
        f"Real, large, timestamped, and domain-diverse — together ~{pretrain_events / 1e6:.1f}M events:"
    )
    L.append("")
    L.append(
        "- **BPI'18 CAP** (finance/subsidy, 2.5M events) — largest single log, the backbone's volume anchor."
    )
    L.append(
        "- **BPI'19 Purchase Orders** (procurement, 1.6M events, 252k cases) — short cases, huge case count."
    )
    L.append(
        "- **Road Traffic Fines** (public admin, 561k events) — short, crisp, high-signal control-flow."
    )
    L.append("- **Hospital Billing** (healthcare-finance, 451k events, 100k cases).")
    L.append("- **BPI'12 Loan** (banking, 262k events) — dense mid-length cases with lifecycle.")
    L.append(
        "- **BPI'11 Hospital** (healthcare, 150k events, **624 activities**) — the open-vocab / role-channel workout."
    )
    L.append("")
    L.append(
        "Optionally fold in the five **BPI'15** municipalities for extra open-vocab pressure (each ~50k events, "
        "~400 activities) — but see the leakage note if you also want them as a transfer test."
    )
    L.append("")
    L.append("### 🔵 Downstream evaluation suite (clean labels)")
    L.append(
        "Datasets with a well-defined outcome or a standard next-activity target, sized for fast iteration:"
    )
    L.append("")
    L.append(
        "- **Sepsis** — ICU-admission / return-to-ER / release-type outcome. *Flagship healthcare benchmark; already wired.*"
    )
    L.append(
        "- **Road Traffic** — fine paid vs sent-to-credit-collection. *Canonical binary outcome.*"
    )
    L.append("- **BPI'12 Loan** — application APPROVED / DECLINED / CANCELLED. *Already wired.*")
    L.append("- **Hospital Billing** — billing closed/reopened outcome.")
    L.append("- **BPI'13 Incidents** — ITSM next-activity + push-to-front behaviour.")
    L.append("")
    L.append(
        "Road Traffic, Hospital Billing and BPI'12 are marked **Both**: large enough to help pretraining *and* "
        "carry clean labels. Use a **case-level split** — pretrain on the train split, fit/evaluate the head on "
        "the held-out split — so the same cases never appear on both sides."
    )
    L.append("")
    L.append("### 🟣 Cross-domain holdout (transfer test — never pretrained on)")
    L.append(
        "To test the role-space transfer claim, keep at least one domain entirely out of pretraining and only "
        "evaluate on it zero-/few-shot:"
    )
    L.append("")
    L.append(
        "- **Sepsis** (healthcare) is the natural holdout if the backbone is trained on finance/admin logs."
    )
    L.append(
        "- The **BPI'15** municipalities form a built-in domain-shift suite (train on 1–4, test on 5)."
    )
    L.append(
        "- The **software-execution** logs (JUnit / NASA / Apache / Statechart) are the far-OOD extreme."
    )
    L.append("")
    L.append("### 🟡 Special regimes — handle with care, don't mix naively")
    L.append(
        "- **Software-execution logs** (JUnit, NASA CEV, Apache Crypto, Statechart WB): method-call traces with "
        "1–3 cases but tens/hundreds of thousands of events and up to 2,614 activities. Unusable at case level; "
        "need trace windowing. Good only as an OOD stress test, not corpus fuel."
    )
    L.append(
        "- **BPI'15**: ~400 activities over ~1k cases — sparse/open-vocab; ideal to exercise the candidate encoder."
    )
    L.append("")
    L.append("### ⚪ Exclude (from real-data pretraining)")
    L.append(
        "- **Process Discovery Contest 2016–2025**: synthetic, homogeneous, 3,930 generated logs (~4.6 GB). Dominates file "
        "count but carries little real-world signal. Optionally a *controlled* synthetic benchmark for discovery / "
        "trace-classification (ground-truth labels are provided), never mixed into the real corpus."
    )
    L.append(
        "- **Activities of Daily Living**: 8 logs of 6–43 cases — too small to train or benchmark."
    )
    L.append("")
    L.append("### Leakage & hygiene notes")
    L.append(
        "- A dataset used **Both** ways must be split at the **case** level before pretraining; never fit a head on "
        "cases the backbone already saw."
    )
    L.append(
        "- BPI'19's timestamps include artifactual early dates (min shown as 1948) — clip/winsorize time features."
    )
    L.append(
        "- Software logs need `position: rope` (uncapped length) or explicit windowing; their max case length is "
        "in the tens of thousands."
    )
    L.append(
        "- Already wired in `configs/data/`: `bpi12`, `bpi17`, `sepsis`, `helpdesk`, `bpi20id`, `mimic_transfers` "
        "(+ multi/solo mixes). New high-value logs to wire next: **road_traffic**, **hospital_billing**, "
        "**bpi18**, **bpi19**, **bpi11_hospital**."
    )
    L.append("")

    # detailed sections
    L.append("## Per-dataset detail")
    L.append("")
    for r in rows:
        m, s, f0 = r["m"], r["s"], r["f0"]
        L.append(f"### {m['short']} — {ROLE_BADGE.get(m['role'], m['role'])}")
        L.append(f"- **Dir:** `{r['dirname']}/`")
        L.append(f"- **Domain / use case:** {m['domain']} — {m['use']}")
        if s.get("is_collection"):
            L.append(
                f"- **Collection:** {s['n_log_files']} log files, {s['total_mb']} MB total"
                + ("; stats from a sampled subset" if s.get("sampled") else "")
            )
        per = " (per representative log)" if s.get("sampled") else ""
        L.append(
            f"- **Scale{per}:** {fnum(r['n_cases'])}{r['scale_suffix']} cases · "
            f"{fnum(r['n_events'])}{r['scale_suffix']} events · "
            f"{fnum(r['acts'])} activities · {fnum(f0.get('n_variants', '?'))} variants "
            f"(variant ratio {f0.get('variant_ratio', '?')})"
        )
        if r.get("pdc_years"):
            L.append("- **Per-year collections** (all synthetic, ~700–1,000 cases/log):")
            L.append("")
            L.append("  | Year | # logs | Activities | Cases/log |")
            L.append("  |---|--:|--:|--:|")
            for y, nlogs, acts, ncases in r["pdc_years"]:
                L.append(f"  | {y} | {nlogs} | {acts} | {fnum(ncases)} |")
            L.append("")
        cl = f0.get("case_len", {})
        L.append(
            f"- **Case length:** min {cl.get('min', '?')}, median {cl.get('median', '?')}, "
            f"mean {cl.get('mean', '?')}, p90 {cl.get('p90', '?')}, max {cl.get('max', '?')}"
        )
        if f0.get("has_timestamps"):
            L.append(
                f"- **Time:** {f0.get('time_start')} → {f0.get('time_end')} "
                f"({f0.get('span_days')} days); avg case duration "
                f"{f0.get('avg_case_duration_days')} d (median {f0.get('median_case_duration_days')} d)"
            )
        else:
            L.append("- **Time:** no usable timestamps")
        top = f0.get("top_activities", [])[:8]
        if top:
            L.append("- **Top activities:** " + ", ".join(f"`{a}` ({fnum(c)})" for a, c in top))
        eav = f0.get("event_attr_keys", [])
        cav = f0.get("case_attr_keys", [])
        lif = f0.get("lifecycles", [])
        L.append(
            f"- **Event attrs ({len(eav)}):** "
            + (
                ", ".join(f"`{k}`" for k in eav[:16]) + (" …" if len(eav) > 16 else "")
                if eav
                else "—"
            )
        )
        L.append(
            f"- **Case attrs ({len(cav)}):** "
            + (
                ", ".join(f"`{k}`" for k in cav[:16]) + (" …" if len(cav) > 16 else "")
                if cav
                else "—"
            )
        )
        L.append("- **Lifecycle:** " + (", ".join(f"`{x}`" for x in lif) if lif else "—"))
        L.append(f"- **Labels available:** {m['labels']}")
        L.append(f"- **Recommendation:** {m['note']}")
        L.append("")

    OUT.write_text("\n".join(L))
    print(f"WROTE {OUT}  ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
