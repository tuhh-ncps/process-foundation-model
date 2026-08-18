"""Typed configuration models and Hydra/OmegaConf helpers.

Raw Hydra configs are parsed into these Pydantic models so the rest of the code
receives validated, typed objects instead of untyped dictionaries.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Backbone architecture hyper-parameters."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 256


class DinoConfig(BaseModel):
    """DINO self-distillation hyper-parameters."""

    out_dim: int = 4096
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    teacher_momentum: float = 0.996
    center_momentum: float = 0.9
    n_global_views: int = 2
    n_local_views: int = 6


class TrainerConfig(BaseModel):
    """Lightning Trainer settings."""

    max_epochs: int = 100
    precision: str = "bf16-mixed"
    devices: int = 1
    accelerator: str = "auto"
    gradient_clip_val: float = 3.0
    accumulate_grad_batches: int = 1


class ExperimentConfig(BaseModel):
    """Top-level resolved experiment configuration."""

    seed: int = 42
    output_dir: str = "outputs"
    model: ModelConfig = Field(default_factory=ModelConfig)
    ssl: DinoConfig = Field(default_factory=DinoConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    data: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)


def parse_config(raw: Any) -> ExperimentConfig:
    """Validate a raw (Hydra/OmegaConf) config into an :class:`ExperimentConfig`."""
    raise NotImplementedError("M0: convert OmegaConf -> dict -> ExperimentConfig.")
