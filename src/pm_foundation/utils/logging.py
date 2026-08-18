"""Structured, human-readable console logging (rich-backed)."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``.

    A thin wrapper so the rest of the codebase never configures logging directly.
    """
    raise NotImplementedError("M0: configure rich logging handler and formatter.")
