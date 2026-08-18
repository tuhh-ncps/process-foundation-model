"""Downstream finetuning / linear-probing entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L

from pm_foundation.data.datamodule import ProcessMiningDataModule
from pm_foundation.data.preprocessing import FeatureSpec
from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.tasks.multitask_module import MultiTaskLitModule, build_heads


def finetune(config: dict[str, Any]) -> list[Mapping[str, float]]:
    """Finetune or probe downstream task(s) using a (optionally pretrained) backbone.

    ``config`` has ``data`` / ``model`` / ``task`` / ``trainer`` sections. When
    ``checkpoint`` points at an M5 output dir (or a ``teacher_backbone.pt``), the
    pretrained backbone **and its feature spec** are loaded so encodings match;
    ``task.backbone.freeze`` selects linear-probe vs full finetune. Returns the
    test metrics.
    """
    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)

    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    task_cfg = dict(config["task"])
    trainer_cfg = dict(config.get("trainer") or {})

    # Reuse the pretraining feature spec when finetuning a loaded backbone.
    checkpoint = config.get("checkpoint")
    backbone_path, feature_spec = _resolve_checkpoint(checkpoint)

    datamodule = ProcessMiningDataModule(data_cfg, mode="downstream", feature_spec=feature_spec)
    datamodule.setup()
    assert datamodule.feature_spec is not None
    spec = datamodule.feature_spec

    backbone = TraceBackbone.from_config(model_cfg, spec)
    heads, weights = build_heads(task_cfg, int(model_cfg.get("d_model", 256)), spec)
    freeze = bool(dict(task_cfg.get("backbone") or {}).get("freeze", True))
    module = MultiTaskLitModule(
        backbone,
        heads,
        head_weights=weights,
        freeze_backbone=freeze,
        optimizer_cfg=task_cfg.get("optimizer"),
    )
    if backbone_path is not None:
        module.load_backbone_state(str(backbone_path))

    output_dir = Path(config.get("output_dir", "outputs"))
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": int(trainer_cfg.get("max_epochs", 50)),
        "accelerator": str(trainer_cfg.get("accelerator", "auto")),
        "devices": trainer_cfg.get("devices", 1),
        "precision": trainer_cfg.get("precision", "32-true"),
        "gradient_clip_val": float(trainer_cfg.get("gradient_clip_val", 1.0)),
        "log_every_n_steps": int(trainer_cfg.get("log_every_n_steps", 25)),
        "default_root_dir": str(output_dir),
    }
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)
    return trainer.test(module, datamodule=datamodule)


def _resolve_checkpoint(checkpoint: str | None) -> tuple[Path | None, FeatureSpec | None]:
    """Resolve a checkpoint path/dir to (backbone weights, feature spec)."""
    if not checkpoint:
        return None, None
    path = Path(checkpoint)
    ckpt_dir = path if path.is_dir() else path.parent
    backbone_path = ckpt_dir / "teacher_backbone.pt" if path.is_dir() else path
    spec_path = ckpt_dir / "feature_spec.json"
    feature_spec = FeatureSpec.load(spec_path) if spec_path.exists() else None
    return backbone_path, feature_spec
