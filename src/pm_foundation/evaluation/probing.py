"""Linear probing: measure representation quality with a frozen backbone.

A linear probe trains only a small head on top of a **frozen** backbone, isolating
the quality of the learned representations from head capacity. It is the standard
diagnostic for self-supervised encoders.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from pm_foundation.data.datamodule import ProcessMiningDataModule
from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.tasks.base import TaskHead
from pm_foundation.tasks.multitask_module import MultiTaskLitModule


def linear_probe(
    backbone: TraceBackbone,
    datamodule: ProcessMiningDataModule,
    head: TaskHead,
    *,
    max_epochs: int = 10,
    lr: float = 1e-2,
    accelerator: str = "auto",
    devices: int = 1,
) -> dict[str, float]:
    """Train ``head`` over a frozen ``backbone`` and return test metrics.

    The backbone is frozen (no gradient), so only the head learns — a true linear
    probe when ``head`` has no hidden layer. The head is keyed by its ``target_key``.
    """
    module = MultiTaskLitModule(
        backbone,
        {head.target_key: head},
        freeze_backbone=True,
        optimizer_cfg={"lr": lr},
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)
    results = trainer.test(module, datamodule=datamodule, verbose=False)
    return {k: float(v) for k, v in results[0].items()}


@torch.no_grad()
def extract_trace_embeddings(
    backbone: TraceBackbone, dataloader: DataLoader[Any], device: str = "cpu"
) -> torch.Tensor:
    """Run the frozen backbone over a loader and return pooled trace embeddings ``(N, D)``.

    Useful for embedding-space analysis (clustering, visualization, closed-form
    probes) independent of any task head.
    """
    backbone.to(device)
    backbone.eval()
    embeddings: list[torch.Tensor] = []
    for batch in dataloader:
        tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = backbone.forward_batch(tensors)
        embeddings.append(out.trace_embedding.cpu())
    return torch.cat(embeddings, dim=0)
