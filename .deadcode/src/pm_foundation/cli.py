"""``pmf`` command-line entry point.

Dispatches ``pretrain | finetune | evaluate`` to the corresponding entrypoints.
Configuration is composed by Hydra (see ``docs/configuration.md``); this thin
wrapper parses the subcommand and hands off the resolved config.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

SUBCOMMANDS = ("pretrain", "finetune", "evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="pmf", description="PM-Foundation CLI")
    parser.add_argument("command", choices=SUBCOMMANDS, help="Action to run")
    # Remaining args are forwarded to Hydra as config overrides.
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    raise NotImplementedError(
        "M0: parse args, compose Hydra config, dispatch to pretrain/finetune/evaluate."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
