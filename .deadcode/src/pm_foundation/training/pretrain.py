"""DINO self-supervised pretraining entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch

from pm_foundation.data.datamodule import ProcessMiningDataModule
from pm_foundation.ssl.dino_module import build_dino_module


def pretrain(config: dict[str, Any]) -> Path:
    """Run DINO pretraining and save the teacher backbone + feature spec.

    ``config`` is a resolved config mapping with ``data`` / ``model`` / ``ssl`` /
    ``trainer`` sections (plus ``seed`` / ``output_dir``). Returns the directory
    holding the saved artifacts.
    """
    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)

    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    ssl_cfg = dict(config["ssl"])
    trainer_cfg = dict(config.get("trainer") or {})
    augmentation = dict(ssl_cfg.get("augmentation") or {})

    datamodule = ProcessMiningDataModule(data_cfg, mode="ssl", augmentation=augmentation)
    datamodule.setup()
    assert datamodule.feature_spec is not None

    module = build_dino_module(model_cfg, ssl_cfg, datamodule.feature_spec)

    output_dir = Path(config.get("output_dir", "outputs"))
    # dict-typed kwargs keep Lightning's strict Literal arg types (e.g. precision)
    # out of the way; values come from config.
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": int(trainer_cfg.get("max_epochs", 100)),
        "accelerator": str(trainer_cfg.get("accelerator", "auto")),
        "devices": trainer_cfg.get("devices", 1),
        "precision": trainer_cfg.get("precision", "32-true"),
        "gradient_clip_val": float(trainer_cfg.get("gradient_clip_val", 3.0)),
        "accumulate_grad_batches": int(trainer_cfg.get("accumulate_grad_batches", 1)),
        "log_every_n_steps": int(trainer_cfg.get("log_every_n_steps", 25)),
        "default_root_dir": str(output_dir),
    }
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)

    # The teacher backbone is the artifact we keep; pair it with the feature spec.
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(module.teacher_backbone.state_dict(), ckpt_dir / "teacher_backbone.pt")
    datamodule.feature_spec.save(ckpt_dir / "feature_spec.json")
    return ckpt_dir
