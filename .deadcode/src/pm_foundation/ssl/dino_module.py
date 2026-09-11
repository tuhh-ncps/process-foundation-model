"""LightningModule orchestrating DINO pretraining.

Holds the student (backbone + projection head) and an EMA teacher, runs multi-view
forward passes (teacher on global views, student on all views), computes the DINO
loss, and updates the teacher + center each step.
"""

from __future__ import annotations

import math
from typing import Any, cast

import lightning as L
import torch
from torch import nn

from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.models.heads.projection import DinoProjectionHead
from pm_foundation.ssl.dino_loss import DinoLoss
from pm_foundation.ssl.teacher_student import EmaTeacher


class DinoEncoder(nn.Module):
    """A trace backbone followed by a DINO projection head.

    ``forward`` takes a collated batch dict and returns prototype logits for the
    pooled (CLS) trace embedding: ``(B, out_dim)``.
    """

    backbone: TraceBackbone
    head: DinoProjectionHead

    def __init__(self, backbone: TraceBackbone, head: DinoProjectionHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out: torch.Tensor = self.head(self.backbone.forward_batch(batch).trace_embedding)
        return out


class DinoLitModule(L.LightningModule):
    """Self-supervised pretraining of a :class:`TraceBackbone` via DINO."""

    student: DinoEncoder
    teacher: EmaTeacher
    loss: DinoLoss

    def __init__(
        self,
        student: DinoEncoder,
        teacher: EmaTeacher,
        loss: DinoLoss,
        optimizer_cfg: dict[str, Any] | None = None,
        final_teacher_temp: float | None = None,
        teacher_temp_warmup_epochs: int = 0,
        freeze_last_layer_epochs: int = 1,
        variance_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg or {}
        # DINO freezes the projection head's prototype (last) layer for the first
        # epoch(s) — the key guard against early collapse to a uniform output.
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        # VICReg-style variance regularization on trace embeddings: a hinge keeping
        # each embedding dimension's batch std >= 1. At small model/data scale plain
        # DINO collapses (uniform or single-prototype); this term keeps embeddings
        # diverse. 0 disables it (vanilla DINO). See ``docs/ssl_dino.md``.
        self.variance_coef = variance_coef
        # Teacher-temperature warmup: anneal from the loss's initial temp (start)
        # to ``final_teacher_temp`` over the first ``warmup`` epochs. A too-sharp
        # teacher early on destabilizes training; warming softens it.
        self.start_teacher_temp = loss.teacher_temp
        self.final_teacher_temp = (
            final_teacher_temp if final_teacher_temp is not None else loss.teacher_temp
        )
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs

    @staticmethod
    def _teacher_temp_at(epoch: int, start: float, final: float, warmup_epochs: int) -> float:
        """Linearly interpolate teacher temperature over the warmup epochs."""
        if warmup_epochs <= 0 or epoch >= warmup_epochs:
            return final
        return start + (final - start) * (epoch / warmup_epochs)

    def on_train_epoch_start(self) -> None:
        self.loss.teacher_temp = self._teacher_temp_at(
            self.current_epoch,
            self.start_teacher_temp,
            self.final_teacher_temp,
            self.teacher_temp_warmup_epochs,
        )

    @property
    def teacher_backbone(self) -> TraceBackbone:
        """The EMA teacher's trace backbone (the artifact kept after pretraining)."""
        return cast(DinoEncoder, self.teacher.teacher).backbone

    @staticmethod
    def _split_views(logits: torch.Tensor, n_views: int) -> list[torch.Tensor]:
        """Split view-major rows ``(n_views * B, D)`` into ``n_views`` of ``(B, D)``."""
        return list(logits.reshape(n_views, -1, logits.shape[-1]))

    @staticmethod
    def _variance_loss(embeddings: torch.Tensor) -> torch.Tensor:
        """VICReg variance hinge: penalize embedding dims with batch std < 1."""
        std = torch.sqrt(embeddings.var(dim=0) + 1e-4)
        return torch.relu(1.0 - std).mean()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        n_global = int(batch["n_global"])
        n_local = int(batch["n_local"])

        # Student backbone embeddings (needed for the variance term), then head.
        global_emb = self.student.backbone.forward_batch(batch["global"]).trace_embedding
        student_views = self._split_views(self.student.head(global_emb), n_global)
        if n_local:
            local_emb = self.student.backbone.forward_batch(batch["local"]).trace_embedding
            student_views += self._split_views(self.student.head(local_emb), n_local)

        # Teacher sees only the global views.
        with torch.no_grad():
            teacher_global = self.teacher.teacher(batch["global"])
        teacher_views = self._split_views(teacher_global, n_global)

        dino_loss: torch.Tensor = self.loss(student_views, teacher_views)
        self.loss.update_center(teacher_global)

        batch_size = teacher_views[0].shape[0]
        self.log("train/dino_loss", dino_loss, prog_bar=True, batch_size=batch_size)
        # Collapse monitor: mean per-dim embedding std (→ 0 means collapse).
        self.log("train/embedding_std", global_emb.std(dim=0).mean(), batch_size=batch_size)

        total = dino_loss
        if self.variance_coef > 0:
            variance_loss = self._variance_loss(global_emb)
            total = dino_loss + self.variance_coef * variance_loss
            self.log("train/variance_loss", variance_loss, batch_size=batch_size)
        self.log("train/loss", total, batch_size=batch_size)
        return total

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        # Cancel the prototype-layer gradients during the freeze window so the
        # student backbone adapts before the prototypes start moving.
        if self.current_epoch < self.freeze_last_layer_epochs:
            self.student.head.prototypes.grad = None

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        total_steps = int(self.trainer.estimated_stepping_batches)
        momentum = self.teacher.momentum_at(self.global_step, total_steps)
        self.teacher.update(self.student, momentum)
        self.log("train/teacher_momentum", momentum)

    def configure_optimizers(self) -> Any:
        cfg = self.optimizer_cfg
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.get("lr", 5e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.04)),
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs = self.trainer.max_epochs or 1
        warmup_steps = int(cfg.get("warmup_epochs", 0) * total_steps / max(1, max_epochs))
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_dino_module(
    model_cfg: dict[str, Any],
    ssl_cfg: dict[str, Any],
    feature_spec: Any,
) -> DinoLitModule:
    """Assemble a :class:`DinoLitModule` (student + EMA teacher + loss) from config."""
    out_dim = int(ssl_cfg.get("out_dim", 4096))
    head_cfg = dict(ssl_cfg.get("projection_head") or {})

    def _make_encoder() -> DinoEncoder:
        backbone = TraceBackbone.from_config(model_cfg, feature_spec)
        head = DinoProjectionHead(
            in_dim=int(model_cfg.get("d_model", 256)),
            out_dim=out_dim,
            hidden_dim=int(head_cfg.get("hidden_dim", 2048)),
            bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
            n_layers=int(head_cfg.get("n_layers", 3)),
        )
        return DinoEncoder(backbone, head)

    student = _make_encoder()
    teacher_net = _make_encoder()
    teacher_net.load_state_dict(student.state_dict())  # identical initialization
    teacher = EmaTeacher(teacher_net, base_momentum=float(ssl_cfg.get("teacher_momentum", 0.996)))

    final_teacher_temp = float(ssl_cfg.get("teacher_temp", 0.04))
    # Optional softer-start temperature for the warmup; defaults to no warmup.
    start_teacher_temp = float(ssl_cfg.get("warmup_teacher_temp", final_teacher_temp))
    loss = DinoLoss(
        out_dim=out_dim,
        student_temp=float(ssl_cfg.get("student_temp", 0.1)),
        teacher_temp=start_teacher_temp,
        center_momentum=float(ssl_cfg.get("center_momentum", 0.9)),
    )
    return DinoLitModule(
        student,
        teacher,
        loss,
        optimizer_cfg=ssl_cfg.get("optimizer"),
        final_teacher_temp=final_teacher_temp,
        teacher_temp_warmup_epochs=int(ssl_cfg.get("teacher_temp_warmup_epochs", 0)),
        freeze_last_layer_epochs=int(ssl_cfg.get("freeze_last_layer_epochs", 1)),
        variance_coef=float(ssl_cfg.get("variance_coef", 0.0)),
    )
