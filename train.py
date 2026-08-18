"""Single Hydra entrypoint for HPC — drives the REAL pm_foundation training flows.

The SAME command runs on a laptop or the cluster; only the ``trainer`` (device/strategy) and
``hydra/launcher`` (where it runs) change, as config — never the code.

    # local AR pretrain (CPU/MPS/1 GPU) — BPI12 is the default dataset
    python train.py experiment=role_rope trainer=local

    # Slurm AR pretrain — 2 nodes x 1 GPU, DDP (submitit writes + submits the job)
    python train.py -m experiment=role_rope trainer=ddp hydra/launcher=slurm \
        hydra.launcher.partition=gpu hydra.launcher.nodes=2 hydra.launcher.gpus_per_node=1

    # downstream label-efficiency eval (single GPU)
    python train.py task=evaluate evaluate=label_efficiency \
        evaluate.backbones.ar=<backbone_run_id>

Under Slurm the DDP topology (devices per node, node count, world size) is read from the Slurm
environment and folded into the trainer config, so `trainer=ddp` needs no hand-edited numbers.
Only global rank 0 prints and owns artifacts (the underlying flow is rank-safe).
"""

from __future__ import annotations

import os
from pathlib import Path

# Anchor all default paths to THIS project directory (self-contained), regardless of the cwd the
# job is launched from. An explicit PROJECT_ROOT / DATA_DIR / OUTPUT_DIR still wins (see configs/
# paths/default.yaml) — this only sets the fallback so `python train.py` works with nothing exported.
os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from pm_foundation.evaluation.label_efficiency import run_label_efficiency
from pm_foundation.training.ar_pretrain_hpc import (
    global_rank,
    pretrain_autoregressive_ddp,
    world_size,
)


def _slurm_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _apply_slurm_topology(trainer_cfg: dict) -> dict:
    """Fill devices / num_nodes / strategy from the Slurm allocation (identity when not on Slurm).

    One task == one GPU: ``--ntasks-per-node`` = GPUs/node = Lightning ``devices``;
    ``--nodes`` = ``num_nodes``. DDP kicks in automatically once world size > 1.
    """
    nodes = _slurm_int("SLURM_NNODES", int(trainer_cfg.get("num_nodes", 1)))
    per_node = _slurm_int("SLURM_NTASKS_PER_NODE", int(trainer_cfg.get("devices", 1)))
    world = _slurm_int("SLURM_NTASKS", nodes * per_node)

    trainer_cfg["devices"] = per_node
    trainer_cfg["num_nodes"] = nodes
    if world > 1 and str(trainer_cfg.get("strategy", "auto")) == "auto":
        trainer_cfg["strategy"] = "ddp"
    return trainer_cfg


def _compose_datasets(run_cfg: dict, data_dir: str) -> dict:
    """Multi-dataset composition for ``data=multi``.

    When ``data.datasets=[name, ...]`` is set, concatenate the log specs of every named
    ``configs/data/<name>.yaml`` (their ``vocab_logs``/``train_logs``) into this run's corpus.
    The existing single-dataset config is the single source of truth for each log's path/format;
    we only borrow its log lists (never its split/batch/etc — those come from ``multi.yaml``).
    Each dataset still contributes only its TRAIN split downstream (shared ``split``), so a
    multi-dataset backbone = union-vocab model trained on 70% of every corpus, val/test held out.
    Identity for a plain single-dataset run (no ``datasets`` key) — nothing composed.
    """
    names = list(run_cfg.get("datasets") or [])
    if not names:
        if not (run_cfg.get("train_logs") or run_cfg.get("vocab_logs")):
            raise ValueError(
                "data=multi requires a dataset list, e.g. ++data.datasets=[mimic_transfers,bpi12]"
            )
        return run_cfg  # e.g. a normal single-dataset config that happens to lack `datasets`
    root = Path(os.environ["PROJECT_ROOT"]) / "configs" / "data"
    ctx = OmegaConf.create({"paths": {"data_dir": data_dir}})  # resolve ${paths.data_dir}
    vocab_logs: list = []
    train_logs: list = []
    for name in names:
        cfg_path = root / f"{name}.yaml"
        if not cfg_path.exists():
            raise ValueError(f"unknown dataset {name!r}: no config at {cfg_path}")
        dc = OmegaConf.to_container(OmegaConf.merge(ctx, OmegaConf.load(cfg_path)), resolve=True)
        vlogs = dc.get("vocab_logs") or []
        vocab_logs.extend(vlogs)
        train_logs.extend(dc.get("train_logs") or vlogs)
    run_cfg["vocab_logs"] = vocab_logs
    run_cfg["train_logs"] = train_logs
    run_cfg["datasets"] = names  # keep for manifest provenance
    return run_cfg


def _build_logger(cfg: DictConfig):
    """Build a Lightning logger from cfg.logger, or None. Built on ALL ranks — the WandbLogger is
    lazy, so wandb only actually initializes on global rank 0 (Lightning guards logger access)."""
    node = cfg.get("logger")
    if node is None:
        return None
    lg = OmegaConf.to_container(node, resolve=True)
    if not lg.get("enabled", False):
        return None
    from lightning.pytorch.loggers import WandbLogger

    offline = str(lg.get("mode", "offline")) == "offline"
    os.environ.setdefault("WANDB_MODE", "offline" if offline else "online")
    return WandbLogger(
        project=lg.get("project", "pm-foundation"),
        entity=lg.get("entity"),
        group=lg.get("group"),
        tags=list(lg.get("tags") or []),
        name=str(cfg.name),
        save_dir=str(cfg.output_dir),  # -> ${output_dir}/wandb/  (inside the project)
        offline=offline,
    )


def _pretrain(cfg: DictConfig) -> str:
    """Compose the pm_foundation pretrain dict from the Hydra tree and run DDP pretraining."""
    run_cfg: dict = OmegaConf.to_container(cfg.data, resolve=True)  # data-config keys at top level
    run_cfg = _compose_datasets(run_cfg, str(cfg.paths.data_dir))  # data=multi -> concat corpora
    run_cfg["model"] = OmegaConf.to_container(cfg.model, resolve=True)
    run_cfg["ar"] = OmegaConf.to_container(cfg.ar, resolve=True)
    run_cfg["trainer"] = _apply_slurm_topology(OmegaConf.to_container(cfg.trainer, resolve=True))
    run_cfg["seed"] = int(cfg.seed)
    run_cfg["name"] = str(cfg.name)
    run_cfg["output_dir"] = str(cfg.output_dir)
    run_cfg["init_from"] = cfg.get("init_from")  # continue-pretraining: warm-start from a prior run
    run_cfg["logger"] = _build_logger(cfg)  # attached to the Trainer inside the flow (rank-0 safe)

    if global_rank() == 0:
        print(
            f"[{cfg.name}] AR pretrain — world_size={world_size()} "
            f"devices/node={run_cfg['trainer']['devices']} nodes={run_cfg['trainer']['num_nodes']} "
            f"strategy={run_cfg['trainer'].get('strategy', 'auto')}"
        )
    return str(pretrain_autoregressive_ddp(run_cfg))


def _evaluate(cfg: DictConfig) -> str:
    """Downstream frozen-probe label-efficiency (single process; DDP not needed)."""
    if cfg.get("evaluate") is None:
        raise ValueError(
            "task=evaluate requires an `evaluate` config, e.g. evaluate=label_efficiency"
        )
    # The probe grid spins up one Trainer per probe; mute Lightning's per-probe INFO banners
    # (GPU/seed/SLURM/estimate-batches) so the run reads as clean progress, not a runaway loop.
    import logging

    for _name in (
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.seed",
        "lightning.pytorch.accelerators.cuda",
    ):
        logging.getLogger(_name).setLevel(logging.WARNING)
    eval_cfg: dict = OmegaConf.to_container(cfg.evaluate, resolve=True)
    eval_cfg.setdefault("output_dir", str(cfg.output_dir))
    mode = str(eval_cfg.get("mode", "label_efficiency"))
    print(f"[{eval_cfg.get('name', mode)}] evaluate ({mode}) — backbones={eval_cfg['backbones']}")
    if mode == "zero_shot":
        from pm_foundation.evaluation.zero_shot import run_zero_shot_matching

        return str(run_zero_shot_matching(eval_cfg))
    return str(run_label_efficiency(eval_cfg))


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> float:
    # Use Tensor Cores for fp32 matmuls (H100/L40S/A100) — a free speedup, no accuracy impact
    # at this scale. No-op on CPU/MPS.
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    task = str(cfg.task)
    if task == "pretrain":
        out = _pretrain(cfg)
    elif task == "evaluate":
        out = _evaluate(cfg)
    else:
        raise ValueError(f"unknown task={task!r} (expected 'pretrain' or 'evaluate')")

    if global_rank() == 0:
        print(f"[{cfg.name}] done -> {out}")
    return 0.0  # Hydra sweepers optimize the return; the flows persist their own metrics


if __name__ == "__main__":
    main()
