"""Trace construction, feature-spec fitting, splitting, and trace encoding.

Turns a normalized :class:`EventLog` into model-ready tensors, as described in
``docs/features.md`` and ``docs/input_data.md``. The fitted :class:`FeatureSpec`
(vocabularies + normalization statistics) is the reusable encoding artifact shared
by pretraining, finetuning, and inference.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import torch

from pm_foundation.data.schema import Event, EventLog, Trace
from pm_foundation.data.vocabulary import UNK, Vocabulary

# Number of temporal features per event (see ``_time_features``).
N_TIME_FEATURES = 12

# Attribute values treated as "missing" regardless of attribute kind.
_MISSING_TOKENS = frozenset({"", "UNKNOWN", "NONE", "NULL", "NAN"})

# Attribute keys that are categorical *identifiers* even when integer-valued, per
# the XES org extension and lifecycle semantics (a resource id is not a quantity).
_FORCE_CATEGORICAL_PREFIXES = ("org:",)
_FORCE_CATEGORICAL_KEYS = frozenset({"lifecycle:transition", "concept:name"})

# Categorical attributes whose cardinality exceeds this are skipped as features (a
# hashing embedding is future work); guards against pathological per-value tables.
DEFAULT_MAX_CATEGORICAL_CARDINALITY = 1000


class SplitStrategy(StrEnum):
    """How to partition cases into train/val/test."""

    TEMPORAL = "temporal"  # split by case start time (recommended, leakage-safe)
    CASE_RANDOM = "case_random"  # random by case id


class AttributeKind(StrEnum):
    """Inferred encoding family for an event/case attribute."""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    SKIPPED = "skipped"  # retained in the log but not turned into a feature


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------
def _is_missing(raw: object) -> bool:
    return raw is None or str(raw).strip().upper() in _MISSING_TOKENS


def _to_float(raw: object) -> float | None:
    """Coerce a raw attribute value to float, or ``None`` if missing/non-numeric."""
    if _is_missing(raw):
        return None
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, int | float):
        return float(raw)
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _cat_token(raw: object) -> str:
    """Normalize a raw attribute value to a categorical token string."""
    return UNK if _is_missing(raw) else str(raw)


def _force_categorical(name: str) -> bool:
    """Whether an attribute key is a categorical identifier regardless of value type."""
    return name in _FORCE_CATEGORICAL_KEYS or name.startswith(_FORCE_CATEGORICAL_PREFIXES)


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------
@dataclass
class AttributeSpec:
    """How a single event/case attribute is encoded."""

    name: str
    scope: str  # "event" | "case"
    kind: AttributeKind
    vocab: Vocabulary | None = None  # categorical only
    mean: float = 0.0  # numeric only
    std: float = 1.0  # numeric only


@dataclass
class FeatureSpec:
    """Reusable encoding artifact: vocabularies + normalization statistics.

    Persisted with checkpoints so pretraining, finetuning, and inference encode
    data identically.
    """

    activity_vocab: Vocabulary
    attributes: list[AttributeSpec] = field(default_factory=list)
    time_mean: dict[str, float] = field(default_factory=dict)
    time_std: dict[str, float] = field(default_factory=dict)
    max_seq_len: int = 256

    @property
    def n_activities(self) -> int:
        return len(self.activity_vocab)

    @property
    def categorical_attributes(self) -> list[AttributeSpec]:
        return [a for a in self.attributes if a.kind == AttributeKind.CATEGORICAL]

    @property
    def numeric_attributes(self) -> list[AttributeSpec]:
        return [a for a in self.attributes if a.kind == AttributeKind.NUMERIC]

    @property
    def skipped_attributes(self) -> list[AttributeSpec]:
        """Attributes retained in the log but not turned into features."""
        return [a for a in self.attributes if a.kind == AttributeKind.SKIPPED]

    @property
    def n_time_features(self) -> int:
        return N_TIME_FEATURES

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_attributes)

    @property
    def n_numeric(self) -> int:
        return len(self.numeric_attributes)

    @property
    def categorical_cardinalities(self) -> list[int]:
        """Vocabulary size per categorical attribute (for embedding tables)."""
        return [len(a.vocab) for a in self.categorical_attributes if a.vocab is not None]

    # -- serialization -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the spec to JSON so it can be reloaded with the checkpoint."""
        payload = {
            "activity_vocab": self.activity_vocab.to_list(),
            "time_mean": self.time_mean,
            "time_std": self.time_std,
            "max_seq_len": self.max_seq_len,
            "attributes": [
                {
                    "name": a.name,
                    "scope": a.scope,
                    "kind": a.kind.value,
                    "vocab": a.vocab.to_list() if a.vocab is not None else None,
                    "mean": a.mean,
                    "std": a.std,
                }
                for a in self.attributes
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FeatureSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        attributes = [
            AttributeSpec(
                name=a["name"],
                scope=a["scope"],
                kind=AttributeKind(a["kind"]),
                vocab=Vocabulary.from_list(a["vocab"]) if a["vocab"] is not None else None,
                mean=a["mean"],
                std=a["std"],
            )
            for a in payload["attributes"]
        ]
        return cls(
            activity_vocab=Vocabulary.from_list(payload["activity_vocab"]),
            attributes=attributes,
            time_mean=payload["time_mean"],
            time_std=payload["time_std"],
            max_seq_len=payload["max_seq_len"],
        )


@dataclass
class Splits:
    """Train/val/test partitions of an event log."""

    train: EventLog
    val: EventLog
    test: EventLog


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------
def build_traces(log: EventLog, min_trace_len: int = 1) -> EventLog:
    """Defensively sort events per trace and drop traces shorter than ``min_trace_len``.

    Readers already group + sort, so this mainly enforces invariants and applies an
    optional minimum-length filter (no-op when ``min_trace_len <= 1``).
    """
    traces: list[Trace] = []
    for trace in log.traces:
        if len(trace) < min_trace_len:
            continue
        events = sorted(trace.events, key=lambda e: e.timestamp)
        traces.append(
            Trace(
                case_id=trace.case_id,
                events=events,
                case_attributes=trace.case_attributes,
            )
        )
    activities = {e.activity for t in traces for e in t.events}
    return EventLog(traces=traces, activity_vocab=sorted(activities))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _case_start(trace: Trace) -> datetime:
    return trace.events[0].timestamp


def split_log(
    log: EventLog,
    strategy: SplitStrategy = SplitStrategy.TEMPORAL,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Splits:
    """Partition whole cases into train/val/test (never splits within a case).

    - ``TEMPORAL``: order cases by start time; earliest → train, latest → test.
      This mirrors a realistic deployment and avoids look-ahead leakage.
    - ``CASE_RANDOM``: shuffle cases with a fixed seed and partition by ratio.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={sum(ratios)}).")

    traces = list(log.traces)
    if strategy == SplitStrategy.TEMPORAL:
        traces.sort(key=_case_start)
    else:
        random.Random(seed).shuffle(traces)

    n = len(traces)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    def _pack(subset: list[Trace]) -> EventLog:
        activities = {e.activity for t in subset for e in t.events}
        return EventLog(traces=subset, activity_vocab=sorted(activities))

    return Splits(
        train=_pack(traces[:n_train]),
        val=_pack(traces[n_train : n_train + n_val]),
        test=_pack(traces[n_train + n_val :]),
    )


# ---------------------------------------------------------------------------
# Feature-spec fitting (train split only)
# ---------------------------------------------------------------------------
def _collect_attribute_names(log: EventLog) -> tuple[list[str], list[str]]:
    event_names: dict[str, None] = {}
    case_names: dict[str, None] = {}
    for trace in log.traces:
        for key in trace.case_attributes:
            case_names.setdefault(key, None)
        for event in trace.events:
            for key in event.event_attributes:
                event_names.setdefault(key, None)
    return list(event_names), list(case_names)


def _fit_attribute(
    name: str, scope: str, values: list[object], max_cardinality: int
) -> AttributeSpec:
    """Infer kind from observed values and fit its vocab / normalization stats.

    Rules (in order): missing → skipped; datetime-valued → skipped (retained in the
    log, not featurized here); ``org:*`` / lifecycle keys → categorical regardless
    of value type; all-numeric → standardized numeric; otherwise categorical, but
    skipped if cardinality exceeds ``max_cardinality``.
    """
    present = [v for v in values if not _is_missing(v)]
    if not present:
        return AttributeSpec(name=name, scope=scope, kind=AttributeKind.SKIPPED)

    if any(isinstance(v, datetime) for v in present):
        return AttributeSpec(name=name, scope=scope, kind=AttributeKind.SKIPPED)

    if not _force_categorical(name):
        floats = [f for v in present if (f := _to_float(v)) is not None]
        if len(floats) == len(present):
            mean = sum(floats) / len(floats)
            var = sum((f - mean) ** 2 for f in floats) / len(floats)
            std = math.sqrt(var) or 1.0
            return AttributeSpec(name, scope, AttributeKind.NUMERIC, mean=mean, std=std)

    vocab = Vocabulary.build(str(v) for v in present)
    if len(vocab) > max_cardinality:
        return AttributeSpec(name=name, scope=scope, kind=AttributeKind.SKIPPED)
    return AttributeSpec(name, scope, AttributeKind.CATEGORICAL, vocab=vocab)


def fit_feature_spec(
    train: EventLog,
    max_seq_len: int = 256,
    max_categorical_cardinality: int = DEFAULT_MAX_CATEGORICAL_CARDINALITY,
) -> FeatureSpec:
    """Fit vocabularies and normalization stats on the **train** split only."""
    activity_vocab = Vocabulary.build(e.activity for t in train.traces for e in t.events)

    event_names, case_names = _collect_attribute_names(train)
    attributes: list[AttributeSpec] = []
    for name in event_names:
        values = [e.event_attributes.get(name) for t in train.traces for e in t.events]
        attributes.append(_fit_attribute(name, "event", values, max_categorical_cardinality))
    for name in case_names:
        values = [t.case_attributes.get(name) for t in train.traces]
        attributes.append(_fit_attribute(name, "case", values, max_categorical_cardinality))

    # Temporal normalization statistics over log-scaled deltas / time-since-start.
    log_deltas: list[float] = []
    log_since: list[float] = []
    for trace in train.traces:
        t0 = trace.events[0].timestamp
        prev = None
        for event in trace.events:
            delta = 0.0 if prev is None else max((event.timestamp - prev).total_seconds(), 0.0)
            since = max((event.timestamp - t0).total_seconds(), 0.0)
            log_deltas.append(math.log1p(delta))
            log_since.append(math.log1p(since))
            prev = event.timestamp

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 1.0
        mean = sum(xs) / len(xs)
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) or 1.0
        return mean, std

    dm, ds = _mean_std(log_deltas)
    sm, ss = _mean_std(log_since)
    return FeatureSpec(
        activity_vocab=activity_vocab,
        attributes=attributes,
        time_mean={"log_delta": dm, "log_since": sm},
        time_std={"log_delta": ds, "log_since": ss},
        max_seq_len=max_seq_len,
    )


# ---------------------------------------------------------------------------
# Trace encoding (Trace + FeatureSpec -> tensors)
# ---------------------------------------------------------------------------
def _time_features(events: list[Event], spec: FeatureSpec) -> torch.Tensor:
    feats = torch.zeros((len(events), N_TIME_FEATURES), dtype=torch.float32)
    t0 = events[0].timestamp
    dm, ds = spec.time_mean["log_delta"], spec.time_std["log_delta"]
    sm, ss = spec.time_mean["log_since"], spec.time_std["log_since"]
    n = len(events)
    prev = None
    for i, event in enumerate(events):
        ts = event.timestamp
        delta = 0.0 if prev is None else max((ts - prev).total_seconds(), 0.0)
        since = max((ts - t0).total_seconds(), 0.0)
        z_delta = (math.log1p(delta) - dm) / ds  # normalized log inter-event gap
        z_since = (math.log1p(since) - sm) / ss  # normalized log elapsed-since-start
        hour = 2 * math.pi * ts.hour / 24
        dow = 2 * math.pi * ts.weekday() / 7
        month = 2 * math.pi * (ts.month - 1) / 12
        feats[i] = torch.tensor(
            [
                z_delta,
                z_since,
                math.sin(hour),
                math.cos(hour),
                math.sin(dow),
                math.cos(dow),
                math.sin(month),
                math.cos(month),
                1.0 if ts.weekday() >= 5 else 0.0,  # is_weekend
                1.0 if 9 <= ts.hour < 17 else 0.0,  # is_business_hour
                math.log1p(i),  # log1p(position)
                i / (n - 1) if n > 1 else 0.0,  # position_ratio in [0, 1]
            ],
            dtype=torch.float32,
        )
        prev = ts
    return feats


def _attr_value(attr: AttributeSpec, trace: Trace, event: Event) -> object:
    source = trace.case_attributes if attr.scope == "case" else event.event_attributes
    return source.get(attr.name)


def _truncate(events: list[Event], max_seq_len: int) -> list[Event]:
    """Keep the most recent ``max_seq_len`` events (no-op if shorter / unset)."""
    if max_seq_len and len(events) > max_seq_len:
        return events[-max_seq_len:]
    return events


# Per-event supervision derivable directly from a trace (used by downstream tasks).
NEXT_ACTIVITY_IGNORE_INDEX = -100


def event_targets(
    trace: Trace, spec: FeatureSpec, *, target_activity_vocab: Vocabulary | None = None
) -> dict[str, torch.Tensor]:
    """Self-derivable per-event targets, aligned with :func:`encode_trace`.

    - ``next_activity`` ``(L,)`` long: the next event's activity id; the final
      position (no successor) is ``NEXT_ACTIVITY_IGNORE_INDEX``.
    - ``remaining_time`` ``(L,)`` float: ``log1p`` seconds from each event to the
      trace's last (kept) event.
    - ``next_time`` ``(L,)`` float: ``log1p`` seconds from each event to the *next*
      event; the final position (no successor) is ``NaN`` and masked out downstream.

    ``target_activity_vocab`` overrides the vocabulary used to encode the next-activity
    *target* (default: ``spec.activity_vocab``, same as the input encoding). Pass a
    separate vocab for CROSS-DATASET evaluation: the backbone encodes inputs with its own
    (training) vocab, but the target must be the EVAL dataset's real activity so accuracy
    reflects true predictive performance instead of collapsing to ``UNK``.
    """
    events = _truncate(trace.events, spec.max_seq_len)
    length = len(events)

    tgt_vocab = target_activity_vocab if target_activity_vocab is not None else spec.activity_vocab
    next_activity = torch.full((length,), NEXT_ACTIVITY_IGNORE_INDEX, dtype=torch.long)
    for i in range(length - 1):
        next_activity[i] = tgt_vocab.encode(events[i + 1].activity)

    last_ts = events[-1].timestamp
    remaining_time = torch.tensor(
        [math.log1p(max((last_ts - e.timestamp).total_seconds(), 0.0)) for e in events],
        dtype=torch.float32,
    )

    next_time = torch.full((length,), float("nan"), dtype=torch.float32)
    for i in range(length - 1):
        gap = max((events[i + 1].timestamp - events[i].timestamp).total_seconds(), 0.0)
        next_time[i] = math.log1p(gap)

    return {
        "next_activity": next_activity,
        "remaining_time": remaining_time,
        "next_time": next_time,
    }


def encode_trace(trace: Trace, spec: FeatureSpec) -> dict[str, torch.Tensor]:
    """Encode one trace into model-ready tensors (no CLS token; added by the model).

    Returns a dict with:
        ``activity_ids``      (L,)           long
        ``time_features``     (L, T)         float
        ``categorical_ids``   (L, n_cat)     long
        ``numeric_features``  (L, n_num)     float
        ``length``            ()             long
    Long traces are truncated to the most recent ``spec.max_seq_len`` events.
    """
    events = _truncate(trace.events, spec.max_seq_len)
    length = len(events)

    activity_ids = torch.tensor(
        [spec.activity_vocab.encode(e.activity) for e in events], dtype=torch.long
    )
    time_features = _time_features(events, spec)

    cat_specs = spec.categorical_attributes
    categorical_ids = torch.zeros((length, len(cat_specs)), dtype=torch.long)
    for j, attr in enumerate(cat_specs):
        assert attr.vocab is not None
        for i, event in enumerate(events):
            token = _cat_token(_attr_value(attr, trace, event))
            categorical_ids[i, j] = attr.vocab.encode(token)

    num_specs = spec.numeric_attributes
    numeric_features = torch.zeros((length, len(num_specs)), dtype=torch.float32)
    for j, attr in enumerate(num_specs):
        for i, event in enumerate(events):
            value = _to_float(_attr_value(attr, trace, event))
            if value is not None:
                numeric_features[i, j] = (value - attr.mean) / attr.std

    return {
        "activity_ids": activity_ids,
        "time_features": time_features,
        "categorical_ids": categorical_ids,
        "numeric_features": numeric_features,
        "length": torch.tensor(length, dtype=torch.long),
    }
