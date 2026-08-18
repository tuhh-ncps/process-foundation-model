"""XES event-log reader (IEEE XES standard).

Implemented as a **streaming** parser over ``xml.etree.ElementTree.iterparse`` so
that very large logs (e.g. the ~80 MB BPI Challenge 2012 log) are read with flat
memory: each ``<trace>`` subtree is built, converted to a :class:`Trace`, and then
cleared before moving on. Standard XES extension keys are mapped to canonical
fields; any remaining flat attributes become event/case attributes.

Supported attribute element types: ``string``, ``date``, ``int``, ``float``,
``boolean``, ``id``. Nested (list/container) attributes are ignored.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from pm_foundation.data.readers.base import LogReader
from pm_foundation.data.schema import ColumnMapping, Event, EventLog, Trace

# Canonical XES keys.
_CONCEPT_NAME = "concept:name"
_TIMESTAMP = "time:timestamp"
_LIFECYCLE = "lifecycle:transition"

# Flat XES attribute element tags we know how to read.
_ATTRIBUTE_TAGS = frozenset({"string", "date", "int", "float", "boolean", "id"})


def _localname(tag: str) -> str:
    """Strip an XML ``{namespace}`` prefix from a tag, returning the local name."""
    return tag.rsplit("}", 1)[-1]


def _parse_timestamp(raw: str) -> datetime:
    """Parse an XES ISO-8601 timestamp (e.g. ``2011-10-01T00:38:44.546+02:00``)."""
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _parse_value(tag: str, raw: str | None) -> Any:
    """Convert a raw XES attribute string to a Python value based on its tag."""
    if raw is None:
        return None
    if tag == "date":
        return _parse_timestamp(raw)
    if tag == "int":
        try:
            return int(raw)
        except ValueError:
            return raw
    if tag == "float":
        try:
            return float(raw)
        except ValueError:
            return raw
    if tag == "boolean":
        return raw.strip().lower() == "true"
    return raw  # string / id


class XesLogReader(LogReader):
    """Reads ``.xes`` / ``.xes.gz`` logs into an :class:`EventLog`.

    Args:
        mapping: Unused for standard XES (kept for interface symmetry).
        activity_key: Event attribute key holding the activity name.
        timestamp_key: Event attribute key holding the timestamp.
        lifecycle_key: Event attribute key holding the lifecycle transition.
        include_lifecycle_in_activity: If True, the activity label becomes
            ``"<activity>+<lifecycle>"`` (a common BPI'12 modelling choice).
        keep_event_attributes: If given, only these event-attribute keys are kept;
            ``None`` keeps all non-canonical attributes.
        keep_case_attributes: Same, for trace/case attributes.
        max_traces: Optional cap on the number of traces read (useful for smoke
            tests on large logs).
    """

    def __init__(
        self,
        mapping: ColumnMapping | None = None,
        *,
        activity_key: str = _CONCEPT_NAME,
        timestamp_key: str = _TIMESTAMP,
        lifecycle_key: str = _LIFECYCLE,
        include_lifecycle_in_activity: bool = False,
        keep_event_attributes: list[str] | None = None,
        keep_case_attributes: list[str] | None = None,
        max_traces: int | None = None,
    ) -> None:
        super().__init__(mapping)
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.lifecycle_key = lifecycle_key
        self.include_lifecycle_in_activity = include_lifecycle_in_activity
        self.keep_event_attributes = keep_event_attributes
        self.keep_case_attributes = keep_case_attributes
        self.max_traces = max_traces

    def read(self, path: str | Path) -> EventLog:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XES log not found: {path}")

        traces: list[Trace] = []
        activities: set[str] = set()
        for trace in self._iter_traces(path):
            traces.append(trace)
            activities.update(e.activity for e in trace.events)
            if self.max_traces is not None and len(traces) >= self.max_traces:
                break

        if not traces:
            raise ValueError(f"No usable traces parsed from {path}")
        return EventLog(traces=traces, activity_vocab=sorted(activities))

    # -- internals ---------------------------------------------------------

    def _iter_traces(self, path: Path) -> Iterator[Trace]:
        """Stream ``<trace>`` elements, yielding built traces and freeing memory."""
        opener = gzip.open if path.suffix == ".gz" else open
        handle: IO[bytes]
        with opener(path, "rb") as handle:  # type: ignore[assignment]
            context = ET.iterparse(handle, events=("end",))
            for _event, elem in context:
                if _localname(elem.tag) != "trace":
                    continue
                trace = self._build_trace(elem)
                elem.clear()  # release this trace's events before the next one
                if trace is not None:
                    yield trace

    def _build_trace(self, trace_elem: ET.Element) -> Trace | None:
        case_id: str | None = None
        case_attrs: dict[str, Any] = {}
        event_elems: list[ET.Element] = []

        for child in trace_elem:
            tag = _localname(child.tag)
            if tag == "event":
                event_elems.append(child)
            elif tag in _ATTRIBUTE_TAGS:
                key = child.get("key")
                if key == _CONCEPT_NAME:
                    case_id = child.get("value")
                elif key is not None:
                    case_attrs[key] = _parse_value(tag, child.get("value"))

        if case_id is None:
            return None  # a trace without a case id is unusable

        events = [ev for e in event_elems if (ev := self._build_event(case_id, e))]
        if not events:
            return None
        events.sort(key=lambda e: e.timestamp)  # enforce per-trace time order

        if self.keep_case_attributes is not None:
            case_attrs = {k: v for k, v in case_attrs.items() if k in self.keep_case_attributes}
        return Trace(case_id=str(case_id), events=events, case_attributes=case_attrs)

    def _build_event(self, case_id: str, event_elem: ET.Element) -> Event | None:
        activity: str | None = None
        timestamp: datetime | None = None
        lifecycle: str | None = None
        attrs: dict[str, Any] = {}

        for child in event_elem:
            tag = _localname(child.tag)
            if tag not in _ATTRIBUTE_TAGS:
                continue
            key = child.get("key")
            raw = child.get("value")
            if key == self.activity_key:
                activity = raw
            elif key == self.timestamp_key:
                timestamp = _parse_timestamp(raw) if raw is not None else None
            elif key is not None:
                if key == self.lifecycle_key:
                    lifecycle = raw
                attrs[key] = _parse_value(tag, raw)

        if activity is None or timestamp is None:
            return None  # incomplete event

        if self.include_lifecycle_in_activity and lifecycle is not None:
            activity = f"{activity}+{lifecycle}"

        if self.keep_event_attributes is not None:
            attrs = {k: v for k, v in attrs.items() if k in self.keep_event_attributes}

        return Event(
            case_id=str(case_id),
            activity=str(activity),
            timestamp=timestamp,
            event_attributes=attrs,
        )
