"""Trace-level outcome labeling with leakage-safe prefix stripping.

An outcome label is derived from the **full** trace (which terminal activity it
reaches), but the model input is truncated to the prefix **before** the first
decision event, so the deciding activity never leaks in. See ``docs/heads.md`` and
the BPI'12 analysis in ``PROGRESS.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pm_foundation.data.schema import EventLog, Trace


@dataclass
class OutcomeLabeler:
    """Derives a trace-level class label and strips decision-leaking events.

    Args:
        terminal_to_class: maps a terminal activity to its class name. Checked in
            insertion order, so the first matching terminal wins (priority).
        deciding_activities: activities removed from the input — the truncation cut
            is at the first event whose activity is in this set.
        min_prefix_len: traces whose surviving prefix is shorter than this are dropped.
    """

    terminal_to_class: dict[str, str]
    deciding_activities: set[str]
    min_prefix_len: int = 2
    classes: list[str] = field(init=False)
    class_to_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        ordered: list[str] = []
        for cls in self.terminal_to_class.values():
            if cls not in ordered:
                ordered.append(cls)
        self.classes = ordered
        self.class_to_id = {cls: i for i, cls in enumerate(ordered)}

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def label(self, trace: Trace) -> str | None:
        """Return the class name for a trace, or ``None`` if no terminal is present."""
        present = {e.activity for e in trace.events}
        for terminal, cls in self.terminal_to_class.items():
            if terminal in present:
                return cls
        return None

    def strip(self, trace: Trace) -> Trace:
        """Return the prefix up to (excluding) the first decision event."""
        events = []
        for event in trace.events:
            if event.activity in self.deciding_activities:
                break
            events.append(event)
        return Trace(case_id=trace.case_id, events=events, case_attributes=trace.case_attributes)

    def apply(self, log: EventLog) -> tuple[list[Trace], list[int]]:
        """Label + strip a log, dropping undecided or too-short traces.

        Returns aligned ``(stripped_traces, label_ids)``.
        """
        traces: list[Trace] = []
        labels: list[int] = []
        for trace in log.traces:
            cls = self.label(trace)
            if cls is None:
                continue
            stripped = self.strip(trace)
            if len(stripped) < self.min_prefix_len:
                continue
            traces.append(stripped)
            labels.append(self.class_to_id[cls])
        return traces, labels


# Care-unit name fragments (lowercased) that identify an ICU-level unit in the MIMIC-IV
# transfers log. Covers MICU / SICU / TSICU / CCU / CVICU / Neuro-SICU / generic ICU, and
# excludes step-down / intermediate / PACU units.
_MIMIC_ICU_TOKENS = ("intensive care", "coronary care", "sicu", "micu", "ccu", "cvicu", "tsicu")
_MIMIC_TERMINALS = frozenset({"EXPIRED", "DISCHARGED", "READMIT_30D", "NO_READMIT_30D"})


def _is_mimic_icu(activity: str) -> bool:
    a = activity.lower()
    return any(tok in a for tok in _MIMIC_ICU_TOKENS)


def _mimic_care_events(trace: Trace) -> list:
    """Care-unit events (terminals dropped), in order."""
    return [e for e in trace.events if e.activity not in _MIMIC_TERMINALS]


def _mimic_stay_hours(trace: Trace, t0) -> float:
    """Hours from admission (first care event, ``t0``) to discharge/death (terminal, else last)."""
    term = [e for e in trace.events if e.activity in _MIMIC_TERMINALS]
    end = term[0].timestamp if term else trace.events[-1].timestamp
    return (end - t0).total_seconds() / 3600.0


def _hours_since(event, t0) -> float:
    return (event.timestamp - t0).total_seconds() / 3600.0


@dataclass
class WindowedTraceLabeler:
    """Case-level label predicted from a FIXED observation window at the start of the admission.

    The label is derived from the FULL trace (``label_of``), but the model input is ONLY the
    care-unit events in the first ``window_hours``. This standardized prediction point fixes two
    leaks of a "cut before the deciding event" scheme: (1) the observation time no longer varies
    per patient, and (2) the window's elapsed duration cannot reveal an end-of-stay label (e.g.
    LOS). Admissions whose stay is shorter than ``window_hours`` (insufficient observation) are
    dropped. Duck-types :class:`OutcomeLabeler` (``apply`` / ``n_classes`` / ``classes``).
    """

    window_hours: float
    label_of: Callable[[Trace], str | None]
    class_names: list[str]
    min_events: int = 2
    classes: list[str] = field(init=False)
    class_to_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.classes = list(self.class_names)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def apply(self, log: EventLog) -> tuple[list[Trace], list[int]]:
        traces: list[Trace] = []
        labels: list[int] = []
        for trace in log.traces:
            cls = self.label_of(trace)
            if cls is None:
                continue
            care = _mimic_care_events(trace)
            if not care:
                continue
            t0 = care[0].timestamp
            if _mimic_stay_hours(trace, t0) < self.window_hours:
                continue  # insufficient observation — stay ends before the prediction point
            obs = [e for e in care if _hours_since(e, t0) <= self.window_hours]
            if len(obs) < self.min_events:
                continue
            traces.append(
                Trace(case_id=trace.case_id, events=obs, case_attributes=trace.case_attributes)
            )
            labels.append(self.class_to_id[cls])
        return traces, labels


@dataclass
class WindowedICULabeler:
    """ICU-admission-within-horizon prediction from a COMMON prediction time.

    At ``obs_hours`` after admission, given the care-unit trajectory so far, predict whether the
    patient enters an ICU/CCU unit within the next ``horizon_hours`` (or EVER, if ``horizon_hours``
    is ``None`` — a less time-constrained, better-balanced target). BOTH classes share the same
    prediction time (unlike a "cut before first ICU" scheme, which hands positives a systematically
    shorter prefix — a giveaway). Excluded: patients already in ICU by ``obs_hours`` (not a future
    prediction) and stays that end before ``obs_hours`` (unobservable). Duck-types
    :class:`OutcomeLabeler`.
    """

    obs_hours: float
    horizon_hours: float | None = 24.0
    min_events: int = 1
    classes: list[str] = field(init=False)
    class_to_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.classes = ["no_icu", "icu"]  # id 0 = negative, 1 = positive
        self.class_to_id = {"no_icu": 0, "icu": 1}

    @property
    def n_classes(self) -> int:
        return 2

    def apply(self, log: EventLog) -> tuple[list[Trace], list[int]]:
        traces: list[Trace] = []
        labels: list[int] = []
        for trace in log.traces:
            care = _mimic_care_events(trace)
            if not care:
                continue
            t0 = care[0].timestamp
            if _mimic_stay_hours(trace, t0) < self.obs_hours:
                continue  # can't observe up to the prediction time
            icu_hours = [_hours_since(e, t0) for e in care if _is_mimic_icu(e.activity)]
            if any(h <= self.obs_hours for h in icu_hours):
                continue  # already in ICU by the prediction time — not a future prediction
            obs = [e for e in care if _hours_since(e, t0) <= self.obs_hours]
            if len(obs) < self.min_events:
                continue
            if self.horizon_hours is None:  # "ever" reaches ICU after the observation window
                hit = any(h > self.obs_hours for h in icu_hours)
            else:
                hit = any(
                    self.obs_hours < h <= self.obs_hours + self.horizon_hours for h in icu_hours
                )
            traces.append(
                Trace(case_id=trace.case_id, events=obs, case_attributes=trace.case_attributes)
            )
            labels.append(1 if hit else 0)
        return traces, labels


def bpi12_application_outcome() -> OutcomeLabeler:
    """BPI'12 application outcome: approved / declined / cancelled.

    The label comes from the terminal ``A_`` activity; the input prefix is cut before
    the first application *or* offer decision (offer-accept ≈ approval, so offer
    terminals are stripped too).
    """
    return OutcomeLabeler(
        terminal_to_class={
            "A_APPROVED": "approved",
            "A_DECLINED": "declined",
            "A_CANCELLED": "cancelled",
        },
        deciding_activities={
            "A_APPROVED",
            "A_DECLINED",
            "A_CANCELLED",
            "A_REGISTERED",
            "A_ACTIVATED",
            "O_ACCEPTED",
            "O_DECLINED",
            "O_CANCELLED",
        },
    )


def bpi17_application_outcome() -> OutcomeLabeler:
    """BPI'17 application outcome: approved / denied / cancelled.

    BPI'17 is the successor loan-application log to BPI'12, with different activity names.
    The outcome is the final application state — ``A_Pending`` (offer accepted, loan
    granted), ``A_Denied``, or ``A_Cancelled``. NOTE ``A_Accepted`` is an *intermediate*
    "accepted for processing" state present in nearly every trace, NOT the outcome.

    The input prefix is cut before the first application terminal *or* offer decision
    (offer-accept ⟺ ``A_Pending``), so the deciding events never leak in — the direct
    mirror of :func:`bpi12_application_outcome`.
    """
    return OutcomeLabeler(
        terminal_to_class={
            "A_Pending": "approved",
            "A_Denied": "denied",
            "A_Cancelled": "cancelled",
        },
        deciding_activities={
            "A_Pending",
            "A_Denied",
            "A_Cancelled",
            "O_Accepted",
            "O_Refused",
            "O_Cancelled",
        },
    )


def bpi20id_declaration_outcome() -> OutcomeLabeler:
    """BPI Challenge 2020 International Declarations outcome: approved / rejected.

    A travel-expense declaration is eventually **approved** (reaches ``Payment Handled`` or a final
    supervisor/director approval) or **rejected** by an approver. Approved terminals are checked
    first, so a declaration rejected once then resubmitted-and-approved counts as approved (its
    *final* state). The input prefix is cut before the first declaration approval/rejection/payment
    decision — the permit sub-flow and the trip stay in the input, the deciding events never leak in.
    """
    return OutcomeLabeler(
        terminal_to_class={
            "Payment Handled": "approved",
            "Declaration FINAL_APPROVED by SUPERVISOR": "approved",
            "Declaration FINAL_APPROVED by DIRECTOR": "approved",
            "Declaration REJECTED by ADMINISTRATION": "rejected",
            "Declaration REJECTED by SUPERVISOR": "rejected",
            "Declaration REJECTED by BUDGET OWNER": "rejected",
            "Declaration REJECTED by PRE_APPROVER": "rejected",
            "Declaration REJECTED by DIRECTOR": "rejected",
            "Declaration REJECTED by MISSING": "rejected",
            "Declaration REJECTED by EMPLOYEE": "rejected",
        },
        # Cut before ANY declaration approval/rejection or the payment (all reveal the outcome);
        # permit approvals / trip events are kept as legitimate pre-decision context.
        deciding_activities={
            "Payment Handled",
            "Request Payment",
            "Declaration FINAL_APPROVED by SUPERVISOR",
            "Declaration FINAL_APPROVED by DIRECTOR",
            "Declaration APPROVED by ADMINISTRATION",
            "Declaration APPROVED by BUDGET OWNER",
            "Declaration APPROVED by PRE_APPROVER",
            "Declaration APPROVED by SUPERVISOR",
            "Declaration REJECTED by ADMINISTRATION",
            "Declaration REJECTED by SUPERVISOR",
            "Declaration REJECTED by BUDGET OWNER",
            "Declaration REJECTED by PRE_APPROVER",
            "Declaration REJECTED by DIRECTOR",
            "Declaration REJECTED by MISSING",
            "Declaration REJECTED by EMPLOYEE",
        },
    )


def sepsis_admission_outcome() -> OutcomeLabeler:
    """Sepsis pathway outcome: ICU admission (icu) vs discharged without ICU (home).

    ``Admission IC`` (intensive care) marks a severe course; a ``Release`` without an ICU admission
    is a normal discharge. Checked ICU-first, so a patient who goes to IC then is released still
    counts as ``icu``. The input is cut before the first ICU-admission / release / ER-return, keeping
    the ER + lab (Leucocytes/CRP/LacticAcid) + IV trajectory as pre-outcome context.
    """
    return OutcomeLabeler(
        terminal_to_class={
            "Admission IC": "icu",
            "Release A": "home",
            "Release B": "home",
            "Release C": "home",
            "Release D": "home",
            "Release E": "home",
        },
        deciding_activities={
            "Admission IC",
            "Release A",
            "Release B",
            "Release C",
            "Release D",
            "Release E",
            "Return ER",
        },
    )


def mimic_mortality() -> OutcomeLabeler:
    """MIMIC-IV in-hospital mortality: expired / survived.

    The converter (``scripts/build_mimic_log.py``) appends a terminal ``EXPIRED`` or
    ``DISCHARGED`` event to each admission's care-unit sequence. The label reads off
    that terminal; the input prefix is cut before it, so the model sees only the
    care-unit trajectory and must predict the outcome from the flow itself.
    """
    return OutcomeLabeler(
        terminal_to_class={
            "EXPIRED": "expired",
            "DISCHARGED": "survived",
        },
        deciding_activities=set(_MIMIC_TERMINALS),
    )


def _mimic_mortality_label(trace: Trace) -> str | None:
    acts = {e.activity for e in trace.events}
    if "EXPIRED" in acts:
        return "expired"
    if "DISCHARGED" in acts:
        return "survived"
    return None


def _mimic_los_label(trace: Trace) -> str | None:
    """Total length-of-stay bucket: short (<3d) / medium (3-7d) / long (>7d)."""
    care = _mimic_care_events(trace)
    if not care:
        return None
    days = _mimic_stay_hours(trace, care[0].timestamp) / 24.0
    if days < 3.0:
        return "short"
    if days < 7.0:
        return "medium"
    return "long"


def _mimic_los_long_label(trace: Trace) -> str | None:
    """Prolonged-stay binary: long (total LOS >= 7d) vs short. ~30% positive — well balanced."""
    care = _mimic_care_events(trace)
    if not care:
        return None
    days = _mimic_stay_hours(trace, care[0].timestamp) / 24.0
    return "long" if days >= 7.0 else "short"


# In-hospital mortality predicted from a fixed observation window (first 24h / 48h). Standardized
# prediction point — no "just before death/discharge" leak. Cohort = admissions surviving >= window.
def mimic_mortality_24h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(24.0, _mimic_mortality_label, ["survived", "expired"])


def mimic_mortality_48h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(48.0, _mimic_mortality_label, ["survived", "expired"])


# Total length-of-stay bucket predicted from a fixed window — the window's elapsed duration is
# bounded (24h/48h) regardless of the true LOS, so it cannot reveal the bucket (the old leak).
def mimic_los_24h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(24.0, _mimic_los_label, ["short", "medium", "long"])


def mimic_los_48h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(48.0, _mimic_los_label, ["short", "medium", "long"])


# Prolonged-stay (LOS >= 7d) binary from a fixed window — a well-balanced (~30%) alternative to
# the 3-class LOS bucket.
def mimic_los_long_24h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(24.0, _mimic_los_long_label, ["short", "long"])


def mimic_los_long_48h() -> WindowedTraceLabeler:
    return WindowedTraceLabeler(48.0, _mimic_los_long_label, ["short", "long"])


# ICU admission within the next 24h, predicted from a COMMON prediction time (24h / 48h). Both
# classes share the prediction point; patients already in ICU by then are excluded.
def mimic_icu_24h() -> WindowedICULabeler:
    return WindowedICULabeler(obs_hours=24.0, horizon_hours=24.0)


def mimic_icu_48h() -> WindowedICULabeler:
    return WindowedICULabeler(obs_hours=48.0, horizon_hours=24.0)


# ICU EVER (any time after the observation window) — less time-constrained, ~3x more positives
# than icu-next-24h, so it clears the macro-F1 floor more easily while carrying the same signal.
def mimic_icu_ever_24h() -> WindowedICULabeler:
    return WindowedICULabeler(obs_hours=24.0, horizon_hours=None)


def mimic_icu_ever_48h() -> WindowedICULabeler:
    return WindowedICULabeler(obs_hours=48.0, horizon_hours=None)


def mimic_readmission_30d() -> OutcomeLabeler:
    """MIMIC-IV 30-day readmission: readmit / no_readmit (discharged-alive cohort).

    Requires the converter (``scripts/build_mimic_log.py``) to append a terminal
    ``READMIT_30D`` / ``NO_READMIT_30D`` per SURVIVING admission (computed from the
    same patient's next admission time). Expired admissions carry no readmission terminal
    and are dropped here (they cannot be readmitted). The input is cut before the discharge
    terminal, so neither the mortality nor readmission marker leaks in.
    """
    return OutcomeLabeler(
        terminal_to_class={
            "READMIT_30D": "readmit",
            "NO_READMIT_30D": "no_readmit",
        },
        deciding_activities=set(_MIMIC_TERMINALS),
    )
