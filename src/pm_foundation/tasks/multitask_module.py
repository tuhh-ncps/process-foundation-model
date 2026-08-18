"""Multi-task LightningModule: one shared backbone, many task heads.

Supports frozen-backbone probing or end-to-end finetuning, with a weighted-sum
loss across heads and per-head metric logging. See ``docs/heads.md`` §3.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.tasks.anomaly import AnomalyHead
from pm_foundation.tasks.base import TaskHead
from pm_foundation.tasks.next_activity import NextActivityHead
from pm_foundation.tasks.next_time import NextTimeHead
from pm_foundation.tasks.outcome import OutcomeHead
from pm_foundation.tasks.remaining_time import RemainingTimeHead


def build_heads(
    task_cfg: dict[str, Any], d_model: int, feature_spec: Any
) -> tuple[dict[str, TaskHead], dict[str, float]]:
    """Construct task heads + their loss weights from a task config.

    ``task_cfg["heads"]`` maps a head name (``next_activity`` / ``remaining_time`` /
    ``next_time`` / ``outcome`` / ``anomaly``) to ``{weight, ...}``.
    """
    heads: dict[str, TaskHead] = {}
    weights: dict[str, float] = {}
    for name, raw in dict(task_cfg.get("heads") or {}).items():
        cfg = dict(raw or {})
        if name == "next_activity":
            heads[name] = NextActivityHead(d_model, feature_spec.n_activities)
        elif name == "remaining_time":
            heads[name] = RemainingTimeHead(d_model)
        elif name == "next_time":
            heads[name] = NextTimeHead(d_model)
        elif name == "outcome":
            heads[name] = OutcomeHead(d_model, int(cfg["n_classes"]))
        elif name == "anomaly":
            heads[name] = AnomalyHead(d_model)
        else:
            raise ValueError(f"Unknown task head: {name!r}")
        weights[name] = float(cfg.get("weight", 1.0))
    if not heads:
        raise ValueError("task config defines no heads")
    return heads, weights


class MultiTaskLitModule(L.LightningModule):
    """Attaches one or more :class:`TaskHead` modules to a shared backbone."""

    backbone: TraceBackbone

    def __init__(
        self,
        backbone: TraceBackbone,
        heads: dict[str, TaskHead],
        head_weights: dict[str, float] | None = None,
        freeze_backbone: bool = True,
        optimizer_cfg: dict[str, Any] | None = None,
        feature_norm: bool = False,
        d_model: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.head_weights = head_weights or {name: 1.0 for name in heads}
        self.freeze_backbone = freeze_backbone
        self.optimizer_cfg = optimizer_cfg or {}
        # Optional per-DIMENSION standardization (BatchNorm) of the frozen features before every
        # head — a better-conditioned input lets a linear probe converge from fewer labels. NOT
        # LayerNorm: the encoder already ends with a LayerNorm, so features are per-sample
        # normalized, but per-dimension variances still differ (~0.3-1.5x); BatchNorm equalizes
        # them, which is what conditions a linear probe. Separate stats for the per-event states
        # vs the pooled trace embedding. Default off => heads read the raw backbone output as before.
        if feature_norm and d_model is None:
            raise ValueError("feature_norm=True requires d_model")
        self.feat_norm_events = nn.BatchNorm1d(d_model) if feature_norm else None
        self.feat_norm_trace = nn.BatchNorm1d(d_model) if feature_norm else None
        self.val_metrics = nn.ModuleDict(
            {name: head.build_metrics(prefix=f"val/{name}/") for name, head in heads.items()}
        )
        self.test_metrics = nn.ModuleDict(
            {name: head.build_metrics(prefix=f"test/{name}/") for name, head in heads.items()}
        )

    def setup(self, stage: str | None = None) -> None:
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # -- forward / targets -------------------------------------------------
    def _backbone_out(self, batch: dict[str, torch.Tensor]) -> Any:
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                out = self.backbone.forward_batch(batch)
        else:
            out = self.backbone.forward_batch(batch)
        if self.feat_norm_events is not None:  # applied OUTSIDE no_grad so the BN params get grad
            es = out.event_states
            b, length, d = es.shape
            es = self.feat_norm_events(es.reshape(b * length, d)).reshape(b, length, d)
            out = replace(
                out, event_states=es, trace_embedding=self.feat_norm_trace(out.trace_embedding)
            )
        return out

    def _targets(self, batch: dict[str, torch.Tensor], head: TaskHead) -> torch.Tensor:
        if head.target_key in batch:
            return batch[head.target_key]
        if head.target_key == "next_activity":
            return NextActivityHead.build_targets(batch["activity_ids"], batch["padding_mask"])
        raise KeyError(
            f"Batch is missing target '{head.target_key}'; provide it via the datamodule."
        )

    # -- steps -------------------------------------------------------------
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self._backbone_out(batch)
        mask = batch["padding_mask"]
        total = torch.zeros((), device=mask.device)
        for name, module in self.heads.items():
            head = cast(TaskHead, module)
            loss = head.loss(head(out, mask), self._targets(batch, head), mask)
            total = total + self.head_weights[name] * loss
            self.log(f"train/{name}_loss", loss, batch_size=mask.shape[0])
        self.log("train/loss", total, prog_bar=True, batch_size=mask.shape[0])
        return total

    def _eval_step(self, batch: dict[str, Any], metrics: nn.ModuleDict) -> None:
        out = self._backbone_out(batch)
        mask = batch["padding_mask"]
        for name, module in self.heads.items():
            head = cast(TaskHead, module)
            collection = cast(MetricCollection, metrics[name])
            head.update_metrics(collection, head(out, mask), self._targets(batch, head), mask)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._eval_step(batch, self.val_metrics)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._eval_step(batch, self.test_metrics)

    def _log_metrics(self, metrics: nn.ModuleDict) -> None:
        for collection in metrics.values():
            assert isinstance(collection, MetricCollection)
            self.log_dict(collection.compute())
            collection.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics)

    # -- optim -------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        cfg = self.optimizer_cfg
        head_lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.01))
        # The feature-norm BatchNorms train alongside the head (backbone stays frozen).
        head_params = list(self.heads.parameters())
        if self.feat_norm_events is not None:
            head_params += list(self.feat_norm_events.parameters())
            head_params += list(self.feat_norm_trace.parameters())
        groups: list[dict[str, Any]] = [{"params": head_params, "lr": head_lr}]
        if not self.freeze_backbone:
            backbone_lr = float(cfg.get("backbone_lr", head_lr * 0.1))
            groups.append({"params": list(self.backbone.parameters()), "lr": backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))

        def lr_lambda(step: int) -> float:
            return 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def load_backbone_state(self, path: str) -> None:
        """Load pretrained backbone weights (e.g. a DINO teacher backbone)."""
        state = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(state)
