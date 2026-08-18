"""Custom Lightning callbacks (EMA/center scheduling, learning-curve capture, etc.).

Standard concerns (checkpointing, early stopping, LR monitoring) use Lightning's
built-in callbacks, configured via the ``trainer`` config group. Project-specific
callbacks live here.
"""

from __future__ import annotations

from typing import Any

import lightning as L


class LearningCurveRecorder(L.Callback):
    """Captures logged metrics once per training epoch into an in-memory history.

    At each ``on_train_epoch_end`` it snapshots ``trainer.callback_metrics`` (which,
    by that point, also holds any validation metrics from the epoch's val loop) as a
    single row ``{"epoch": e, <metric>: <float>, ...}``. The accumulated ``history``
    is written to ``learning_curve.csv`` by the pretraining entrypoint.
    """

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Record only LOSS curves — mixing a 0–1 accuracy onto the same axis as the losses just
        # confuses the plot. Non-loss metrics (e.g. next-activity accuracy) still reach the
        # experiment logger (W&B) directly via self.log; they're only excluded from this curve.
        row: dict[str, Any] = {"epoch": int(trainer.current_epoch)}
        for key, value in trainer.callback_metrics.items():
            if not key.endswith("loss"):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):  # non-scalar metric — skip
                continue
        self.history.append(row)


class TeacherMomentumScheduler(L.Callback):
    """Updates the DINO teacher-momentum (``λ``) on a cosine schedule per step."""

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        raise NotImplementedError("M5: set teacher momentum from global step.")
