"""Event-log readers. All readers emit a normalized :class:`EventLog`."""

from __future__ import annotations

from typing import Any

from pm_foundation.data.readers.base import LogReader
from pm_foundation.data.readers.csv_reader import CsvLogReader
from pm_foundation.data.readers.xes_reader import XesLogReader
from pm_foundation.data.schema import ColumnMapping

__all__ = [
    "CsvLogReader",
    "LogReader",
    "XesLogReader",
    "build_reader_from_config",
    "get_reader",
]


def get_reader(fmt: str, mapping: ColumnMapping | None = None, **kwargs: Any) -> LogReader:
    """Return a reader for ``fmt`` (``"csv"`` or ``"xes"``).

    Extra keyword arguments are forwarded to the reader (XES options are ignored
    by the CSV reader).
    """
    fmt = fmt.lower().lstrip(".")
    if fmt == "csv":
        return CsvLogReader(mapping, max_traces=kwargs.get("max_traces"))
    if fmt in ("xes", "xes.gz", "gz"):
        return XesLogReader(mapping, **kwargs)
    raise ValueError(f"Unsupported log format: {fmt!r} (expected 'csv' or 'xes').")


def build_reader_from_config(config: dict[str, Any]) -> LogReader:
    """Construct a reader from a (Hydra) data-config mapping.

    Recognised keys: ``format``, ``mapping`` (CSV column mapping), ``xes`` (XES
    options), ``event_attributes``, ``case_attributes``. For XES, empty attribute
    lists mean "keep all non-canonical attributes".
    """
    fmt = str(config.get("format", "csv")).lower()
    event_attrs = list(config.get("event_attributes") or [])
    case_attrs = list(config.get("case_attributes") or [])

    if fmt == "csv":
        raw_map = dict(config.get("mapping") or {})
        mapping = ColumnMapping(
            case_id=raw_map["case_id"],
            activity=raw_map["activity"],
            timestamp=raw_map["timestamp"],
            timestamp_format=raw_map.get("timestamp_format"),
            event_attributes=event_attrs,
            case_attributes=case_attrs,
        )
        return CsvLogReader(mapping)

    if fmt in ("xes", "xes.gz"):
        xes_opts = dict(config.get("xes") or {})
        return XesLogReader(
            include_lifecycle_in_activity=bool(
                xes_opts.get("include_lifecycle_in_activity", False)
            ),
            keep_event_attributes=event_attrs or None,
            keep_case_attributes=case_attrs or None,
            max_traces=xes_opts.get("max_traces"),
        )

    raise ValueError(f"Unsupported log format: {fmt!r} (expected 'csv' or 'xes').")
