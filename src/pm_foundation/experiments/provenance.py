"""Run registry: identity, manifests, and cross-linking for experiment outputs.

A :class:`RunRegistry` owns an output root (``outputs/`` by default) and lays runs
out under ``backbones/<run_id>/`` and ``label_efficiency/<run_id>/``. Each run has a
``manifest.json`` capturing its config, git revision, data summary, metrics, and
links to other runs. Backbone runs additionally accrue a ``linked_evals.jsonl`` so
the backbone -> evaluation direction is navigable without scanning the whole tree.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Maps a run *kind* to the sub-directory of the output root that holds its runs.
_KIND_SUBDIR = {
    "backbone": "backbones",
    "label_efficiency": "label_efficiency",
    "zero_shot": "zero_shot",
}


def _now() -> datetime:
    return datetime.now()


def _slugify(name: str) -> str:
    """A filesystem-safe fragment for a human-supplied run name."""
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in name.strip().lower()]
    slug = "".join(keep).strip("-")
    return slug[:40]


def generate_run_id(kind: str, name: str | None = None, *, now: datetime | None = None) -> str:
    """A sortable, unique run id: ``<kind>-<YYYYmmdd-HHMMSS>[-<name>]-<hex>``."""
    stamp = (now or _now()).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    parts = [kind.replace("_", "-"), stamp]
    if name:
        parts.append(_slugify(name))
    parts.append(suffix)
    return "-".join(parts)


def git_revision() -> dict[str, Any]:
    """Best-effort current commit + dirty flag; empty on failure (e.g. no git)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {}


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


@dataclass
class RunManifest:
    """Serializable record of one experiment run."""

    run_id: str
    kind: str
    created_at: str
    name: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    links: dict[str, Any] = field(default_factory=dict)
    completed_at: str | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class RunContext:
    """A live run: its id, directory, manifest, and owning registry."""

    run_id: str
    dir: Path
    manifest: RunManifest
    registry: RunRegistry


class RunRegistry:
    """Creates run directories, writes manifests, and links runs together."""

    def __init__(self, root: str | Path = "outputs") -> None:
        self.root = Path(root)

    def _subdir(self, kind: str) -> Path:
        try:
            return self.root / _KIND_SUBDIR[kind]
        except KeyError:
            raise ValueError(
                f"Unknown run kind {kind!r} (expected one of {list(_KIND_SUBDIR)})."
            ) from None

    def run_dir(self, kind: str, run_id: str) -> Path:
        """The directory that holds (or would hold) ``run_id`` of ``kind``."""
        return self._subdir(kind) / run_id

    def start(
        self,
        kind: str,
        config: dict[str, Any],
        *,
        name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunContext:
        """Create a fresh run directory and write an initial manifest.

        The manifest is written up front so a crashed run still leaves a record; call
        :meth:`finish` to append metrics/links and index the run when it completes.
        """
        run_id = generate_run_id(kind, name)
        run_dir = self._subdir(kind) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = RunManifest(
            run_id=run_id,
            kind=kind,
            created_at=_now().isoformat(timespec="seconds"),
            name=name,
            git=git_revision(),
            config=config,
            data=data or {},
        )
        manifest.save(run_dir / "manifest.json")
        return RunContext(run_id=run_id, dir=run_dir, manifest=manifest, registry=self)

    def finish(
        self,
        ctx: RunContext,
        *,
        metrics: dict[str, Any] | None = None,
        links: dict[str, Any] | None = None,
    ) -> None:
        """Persist final metrics/links and append the run to the global index."""
        if metrics:
            ctx.manifest.metrics.update(metrics)
        if links:
            ctx.manifest.links.update(links)
        ctx.manifest.completed_at = _now().isoformat(timespec="seconds")
        ctx.manifest.save(ctx.dir / "manifest.json")
        _append_jsonl(
            self.root / "runs.jsonl",
            {
                "run_id": ctx.run_id,
                "kind": ctx.manifest.kind,
                "name": ctx.manifest.name,
                "created_at": ctx.manifest.created_at,
                "completed_at": ctx.manifest.completed_at,
                "dir": str(ctx.dir.relative_to(self.root)),
                "git_commit": ctx.manifest.git.get("commit"),
                "links": ctx.manifest.links,
            },
        )

    def link_backbone_to_eval(
        self, backbone_run_id: str, eval_run_id: str, *, info: dict[str, Any] | None = None
    ) -> None:
        """Record (append) that ``eval_run_id`` consumed backbone ``backbone_run_id``."""
        bdir = self.run_dir("backbone", backbone_run_id)
        if not bdir.exists():
            return  # e.g. a "random" pseudo-backbone has no directory
        _append_jsonl(
            bdir / "linked_evals.jsonl",
            {
                "eval_run_id": eval_run_id,
                "linked_at": _now().isoformat(timespec="seconds"),
                **(info or {}),
            },
        )
