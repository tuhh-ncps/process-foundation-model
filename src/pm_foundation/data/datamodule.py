"""Lightning DataModule wiring readers -> preprocessing -> datasets -> loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightning as L
from torch.utils.data import DataLoader, Dataset

from pm_foundation.data.augmentations import build_augmenter
from pm_foundation.data.dataset import (
    DinoTraceDataset,
    OutcomeDataset,
    SupervisedTraceDataset,
    TraceDataset,
    collate_dino,
    collate_outcome,
    collate_supervised,
    collate_traces,
)
from pm_foundation.data.labeling import OutcomeLabeler
from pm_foundation.data.preprocessing import (
    FeatureSpec,
    SplitStrategy,
    build_traces,
    fit_feature_spec,
    split_log,
)
from pm_foundation.data.readers import build_reader_from_config
from pm_foundation.data.schema import EventLog


class ProcessMiningDataModule(L.LightningDataModule):
    """End-to-end data pipeline for both pretraining and downstream tasks.

    ``prepare_data`` reads the raw log; ``setup`` builds traces, fits (or reuses) the
    :class:`FeatureSpec` on the train split, and constructs datasets. Modes:

    - ``"ssl"``: train yields DINO multi-crop batches (``collate_dino``); val/test
      yield plain encoded batches for monitoring.
    - ``"downstream"``: all splits yield encoded batches **plus per-event targets**
      (``collate_supervised``) for finetuning / probing.

    Pass ``feature_spec`` to reuse a pretrained model's encoding (vocab + stats)
    instead of fitting a new one — required when finetuning a loaded backbone.
    """

    def __init__(
        self,
        config: dict[str, Any],
        mode: str = "ssl",
        augmentation: dict[str, Any] | None = None,
        feature_spec: FeatureSpec | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.augmentation = augmentation or {}
        self.feature_spec: FeatureSpec | None = feature_spec
        self.event_log: EventLog | None = None
        self.train_set: Dataset[Any] | None = None
        self.val_set: TraceDataset | None = None
        self.test_set: TraceDataset | None = None

    def prepare_data(self) -> None:
        """Read the raw log (CSV/XES) into a normalized :class:`EventLog`.

        Note: assigning ``self.event_log`` here is convenient for single-process
        runs; a future revision will cache parsing to an on-disk artifact so it is
        DDP-safe and not repeated per worker. Skips if a log was already injected.
        """
        if self.event_log is not None:
            return
        reader = build_reader_from_config(self.config)
        self.event_log = reader.read(self.config["path"])

    def setup(self, stage: str | None = None) -> None:
        if self.event_log is None:
            self.prepare_data()
        assert self.event_log is not None

        min_len = int(self.config.get("min_trace_len", 1))
        max_len = int(self.config.get("max_trace_len", 256))
        split_cfg = dict(self.config.get("split") or {})
        strategy = SplitStrategy(split_cfg.get("strategy", "temporal"))
        ratios = tuple(split_cfg.get("ratios", (0.7, 0.15, 0.15)))
        seed = int(self.config.get("seed", 42))

        log = build_traces(self.event_log, min_trace_len=min_len)
        splits = split_log(log, strategy=strategy, ratios=ratios, seed=seed)

        # Reuse an injected feature spec (e.g. from a pretrained checkpoint), else
        # fit on TRAIN ONLY to prevent leakage.
        if self.feature_spec is None:
            max_card = int(self.config.get("max_categorical_cardinality", 1000))
            self.feature_spec = fit_feature_spec(
                splits.train, max_seq_len=max_len, max_categorical_cardinality=max_card
            )
        spec = self.feature_spec

        if self.mode == "ssl":
            augmenter = build_augmenter(self.augmentation, spec)
            self.train_set = DinoTraceDataset(splits.train, spec, augmenter)
            self.val_set = TraceDataset(splits.val, spec)
            self.test_set = TraceDataset(splits.test, spec)
        else:
            self.train_set = SupervisedTraceDataset(splits.train, spec)
            self.val_set = SupervisedTraceDataset(splits.val, spec)
            self.test_set = SupervisedTraceDataset(splits.test, spec)

    # -- dataloaders -------------------------------------------------------
    @property
    def _train_collate(self) -> Callable[[list[Any]], dict[str, Any]]:
        return collate_dino if self.mode == "ssl" else collate_supervised

    @property
    def _eval_collate(self) -> Callable[[list[Any]], dict[str, Any]]:
        return collate_traces if self.mode == "ssl" else collate_supervised

    def _loader(
        self,
        dataset: Dataset[Any] | None,
        *,
        shuffle: bool,
        collate: Callable[[list[Any]], dict[str, Any]],
    ) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting loaders.")
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=shuffle,
            num_workers=int(self.config.get("num_workers", 0)),
            collate_fn=collate,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.train_set, shuffle=True, collate=self._train_collate)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.val_set, shuffle=False, collate=self._eval_collate)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.test_set, shuffle=False, collate=self._eval_collate)


class OutcomeDataModule(L.LightningDataModule):
    """Data pipeline for a trace-level **outcome** task (e.g. BPI'12 application outcome).

    Applies an :class:`OutcomeLabeler` (label from full trace + leak-free prefix
    stripping), splits the labeled traces temporally by prefix start time, fits (or
    reuses) the :class:`FeatureSpec` on train only, and yields ``collate_outcome``
    batches with a trace-level ``outcome`` target. Plugs into ``linear_probe`` /
    ``MultiTaskLitModule`` like any other downstream datamodule.
    """

    def __init__(
        self,
        config: dict[str, Any],
        labeler: OutcomeLabeler,
        feature_spec: FeatureSpec | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.labeler = labeler
        self.feature_spec: FeatureSpec | None = feature_spec
        self.event_log: EventLog | None = None
        self.n_classes = labeler.n_classes
        self.train_set: OutcomeDataset | None = None
        self.val_set: OutcomeDataset | None = None
        self.test_set: OutcomeDataset | None = None

    def prepare_data(self) -> None:
        if self.event_log is not None:
            return
        reader = build_reader_from_config(self.config)
        self.event_log = reader.read(self.config["path"])

    def setup(self, stage: str | None = None) -> None:
        if self.event_log is None:
            self.prepare_data()
        assert self.event_log is not None

        log = build_traces(self.event_log, min_trace_len=int(self.config.get("min_trace_len", 1)))
        traces, labels = self.labeler.apply(log)

        # Temporal split by prefix start time (whole-case, leakage-safe).
        order = sorted(range(len(traces)), key=lambda i: traces[i].events[0].timestamp)
        ratios = tuple(dict(self.config.get("split") or {}).get("ratios", (0.7, 0.15, 0.15)))
        n = len(order)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        parts = {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
        }

        if self.feature_spec is None:
            train_traces = [traces[i] for i in parts["train"]]
            acts = sorted({e.activity for t in train_traces for e in t.events})
            train_log = EventLog(traces=train_traces, activity_vocab=acts)
            self.feature_spec = fit_feature_spec(
                train_log, max_seq_len=int(self.config.get("max_trace_len", 256))
            )
        spec = self.feature_spec

        def _ds(split: str) -> OutcomeDataset:
            idx = parts[split]
            return OutcomeDataset([traces[i] for i in idx], [labels[i] for i in idx], spec)

        self.train_set, self.val_set, self.test_set = _ds("train"), _ds("val"), _ds("test")

    def _loader(self, dataset: OutcomeDataset | None, *, shuffle: bool) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError("OutcomeDataModule.setup() must be called first.")
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=shuffle,
            num_workers=int(self.config.get("num_workers", 0)),
            collate_fn=collate_outcome,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.test_set, shuffle=False)
