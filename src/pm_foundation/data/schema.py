"""Canonical, validated event-log schema.

All readers normalize their input into these structures, so everything downstream
is format-agnostic. See ``docs/input_data.md``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    """Declares how raw columns map to canonical fields (used by readers)."""

    case_id: str
    activity: str
    timestamp: str
    timestamp_format: str | None = None
    event_attributes: list[str] = Field(default_factory=list)
    case_attributes: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """A single event within a case."""

    case_id: str
    activity: str
    timestamp: datetime
    event_attributes: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """An ordered sequence of events sharing a ``case_id``."""

    case_id: str
    events: list[Event] = Field(default_factory=list)
    case_attributes: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)


class EventLog(BaseModel):
    """A collection of traces plus the discovered activity vocabulary."""

    traces: list[Trace] = Field(default_factory=list)
    activity_vocab: list[str] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.traces)
