"""Abstract reader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pm_foundation.data.schema import ColumnMapping, EventLog


class LogReader(ABC):
    """Reads a raw event log from disk into a normalized :class:`EventLog`."""

    def __init__(self, mapping: ColumnMapping | None = None) -> None:
        self.mapping = mapping

    @abstractmethod
    def read(self, path: str | Path) -> EventLog:
        """Parse ``path`` and return a normalized event log."""
        raise NotImplementedError
