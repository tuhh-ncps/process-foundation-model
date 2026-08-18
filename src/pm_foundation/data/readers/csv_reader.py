"""CSV event-log reader.

Reads a flat table where each row is an event, using a :class:`ColumnMapping` to
identify the ``case_id`` / ``activity`` / ``timestamp`` columns and attributes.
Rows are sorted by ``(case_id, timestamp)`` and grouped into traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pm_foundation.data.readers.base import LogReader
from pm_foundation.data.schema import ColumnMapping, Event, EventLog, Trace

# When a data config gives no explicit ``mapping``, assume the log already uses these
# canonical column names. Converters that target this project (e.g. the MIMIC-IV Story-A
# builder) emit exactly these, so a bare ``format: csv`` config just works.
_CANONICAL_MAPPING = ColumnMapping(case_id="case_id", activity="activity", timestamp="timestamp")


class CsvLogReader(LogReader):
    """Reads CSV logs (via pandas) into an :class:`EventLog`.

    ``max_traces`` caps the number of traces read (the first N case_ids in sorted order) —
    useful for smoke tests and for bounding very large logs (e.g. MIMIC's 546k admissions).
    """

    def __init__(self, mapping: ColumnMapping | None = None, max_traces: int | None = None) -> None:
        super().__init__(mapping)
        self.max_traces = max_traces

    def read(self, path: str | Path) -> EventLog:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV log not found: {path}")
        # No mapping -> canonical case_id/activity/timestamp columns (see _CANONICAL_MAPPING).
        m = self.mapping if self.mapping is not None else _CANONICAL_MAPPING

        df = pd.read_csv(path)

        required = [m.case_id, m.activity, m.timestamp]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s) {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df[m.timestamp] = pd.to_datetime(df[m.timestamp], format=m.timestamp_format, errors="raise")
        df = df.sort_values([m.case_id, m.timestamp])
        if self.max_traces is not None:
            keep = df[m.case_id].drop_duplicates().head(self.max_traces)
            df = df[df[m.case_id].isin(keep)]

        event_attr_cols = [c for c in m.event_attributes if c in df.columns]
        case_attr_cols = [c for c in m.case_attributes if c in df.columns]

        traces: list[Trace] = []
        activities: set[str] = set()
        for case_id, group in df.groupby(m.case_id, sort=False):
            # Column-wise extraction avoids itertuples renaming non-identifier
            # column names (e.g. "concept:name") and is faster than per-row access.
            acts = group[m.activity].astype(str).tolist()
            times = group[m.timestamp].tolist()
            attr_cols = {c: group[c].tolist() for c in event_attr_cols}

            events: list[Event] = []
            for i, activity in enumerate(acts):
                activities.add(activity)
                events.append(
                    Event(
                        case_id=str(case_id),
                        activity=activity,
                        timestamp=pd.Timestamp(times[i]).to_pydatetime(),
                        event_attributes={c: attr_cols[c][i] for c in event_attr_cols},
                    )
                )
            first = group.iloc[0]
            case_attrs: dict[str, Any] = {c: first[c] for c in case_attr_cols}
            traces.append(Trace(case_id=str(case_id), events=events, case_attributes=case_attrs))

        if not traces:
            raise ValueError(f"No traces parsed from {path}")
        return EventLog(traces=traces, activity_vocab=sorted(activities))
