"""Torch datasets and collation for traces.

``TraceDataset`` encodes traces into tensors using a :class:`FeatureSpec` (via
:func:`encode_trace`). ``collate_traces`` pads variable-length traces into a batch
and builds the padding mask. For self-supervised pretraining, ``DinoTraceDataset``
additionally applies the multi-crop augmenter (implemented in M3).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from pm_foundation.data.augmentations import MultiCropTraceAugmenter, TraceView
from pm_foundation.data.preprocessing import (
    NEXT_ACTIVITY_IGNORE_INDEX,
    FeatureSpec,
    encode_trace,
    event_targets,
)
from pm_foundation.data.schema import EventLog
from pm_foundation.data.vocabulary import RESERVED_TOKENS, Vocabulary

Batch = dict[str, Any]

# Padding uses id 0 across activity and categorical streams; this is the reserved
# PAD token (``RESERVED_TOKENS[0]``), so padded positions embed to the PAD vector.
_PAD_ID = 0
assert RESERVED_TOKENS[0] == "<PAD>"

# Per-event supervision targets: (pad value, dtype) used when collating.
_EVENT_TARGET_PAD: dict[str, tuple[float, torch.dtype]] = {
    "next_activity": (NEXT_ACTIVITY_IGNORE_INDEX, torch.long),
    "remaining_time": (0.0, torch.float32),
    "next_time": (float("nan"), torch.float32),  # last event has no successor -> NaN, masked
    "role_ids": (_PAD_ID, torch.long),  # current-catalogue ids for the role channel
}
_FEATURE_KEYS = (
    "activity_ids",
    "time_features",
    "categorical_ids",
    "numeric_features",
    "length",
)


def _encode_role_ids(trace: Any, spec: FeatureSpec, role_vocab: Vocabulary) -> torch.Tensor:
    """Events encoded with the CURRENT catalogue's vocab, truncation-aligned with
    :func:`encode_trace` (keep the most recent ``max_seq_len`` events)."""
    events = trace.events
    if spec.max_seq_len and len(events) > spec.max_seq_len:
        events = events[-spec.max_seq_len :]
    return torch.tensor([role_vocab.encode(e.activity) for e in events], dtype=torch.long)


class TraceDataset(Dataset[dict[str, torch.Tensor]]):
    """Encodes each trace into model-ready tensors (no augmentation)."""

    def __init__(self, log: EventLog, feature_spec: FeatureSpec) -> None:
        self.log = log
        self.feature_spec = feature_spec

    def __len__(self) -> int:
        return len(self.log)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return encode_trace(self.log.traces[index], self.feature_spec)


class SupervisedTraceDataset(TraceDataset):
    """Encodes each trace plus self-derivable per-event targets (next-activity,
    remaining-time, next-time) for downstream finetuning / probing.

    ``target_activity_vocab`` (optional) sets the vocabulary for the next-activity
    *target*, decoupled from the ``feature_spec`` used to encode inputs. Pass the EVAL
    dataset's vocab for cross-dataset probing so the target is the real next activity
    (not ``UNK``) even when the backbone was trained on a different activity set.

    ``role_vocab`` (optional) adds ``role_ids`` — the events encoded with the CURRENT
    dataset's catalogue — so a role-channel backbone can look up e(a_i) in the eval
    catalogue's table even when the backbone's own vocab would UNK the activity.
    """

    def __init__(
        self,
        log: EventLog,
        feature_spec: FeatureSpec,
        target_activity_vocab: Vocabulary | None = None,
        role_vocab: Vocabulary | None = None,
    ) -> None:
        super().__init__(log, feature_spec)
        self.target_activity_vocab = target_activity_vocab
        self.role_vocab = role_vocab

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trace = self.log.traces[index]
        item = encode_trace(trace, self.feature_spec)
        item.update(
            event_targets(
                trace, self.feature_spec, target_activity_vocab=self.target_activity_vocab
            )
        )
        if self.role_vocab is not None:
            item["role_ids"] = _encode_role_ids(trace, self.feature_spec, self.role_vocab)
        return item


class OutcomeDataset(Dataset[dict[str, torch.Tensor]]):
    """Encodes (already prefix-stripped) traces with a trace-level outcome label.

    ``role_vocab`` (optional): see :class:`SupervisedTraceDataset`.
    """

    def __init__(
        self,
        traces: list[Any],
        labels: list[int],
        feature_spec: FeatureSpec,
        role_vocab: Vocabulary | None = None,
    ) -> None:
        self.traces = traces
        self.labels = labels
        self.feature_spec = feature_spec
        self.role_vocab = role_vocab

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = encode_trace(self.traces[index], self.feature_spec)
        item["outcome"] = torch.tensor(self.labels[index], dtype=torch.long)
        if self.role_vocab is not None:
            item["role_ids"] = _encode_role_ids(
                self.traces[index], self.feature_spec, self.role_vocab
            )
        return item


class DinoTraceDataset(Dataset[dict[str, list[TraceView]]]):
    """Yields ``{"global": [...], "local": [...]}`` augmented views per trace.

    The trace is encoded once, then the multi-crop augmenter produces the global
    (teacher) and local (student-only) views consumed by the DINO objective.
    """

    def __init__(
        self,
        log: EventLog,
        feature_spec: FeatureSpec,
        augmenter: MultiCropTraceAugmenter,
    ) -> None:
        self.log = log
        self.feature_spec = feature_spec
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.log)

    def __getitem__(self, index: int) -> dict[str, list[TraceView]]:
        base = encode_trace(self.log.traces[index], self.feature_spec)
        return self.augmenter(base)


def collate_traces(batch: list[dict[str, torch.Tensor]]) -> Batch:
    """Pad a list of encoded traces to a batch with a boolean padding mask.

    Output tensors (``B`` = batch size, ``L`` = max length in batch):
        ``activity_ids``     (B, L)            long
        ``time_features``    (B, L, T)         float
        ``categorical_ids``  (B, L, n_cat)     long
        ``numeric_features`` (B, L, n_num)     float
        ``padding_mask``     (B, L)            bool  (True where padded)
        ``lengths``          (B,)              long
    """
    bsz = len(batch)
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max())

    n_time = batch[0]["time_features"].shape[-1]
    n_cat = batch[0]["categorical_ids"].shape[-1]
    n_num = batch[0]["numeric_features"].shape[-1]

    activity_ids = torch.full((bsz, max_len), _PAD_ID, dtype=torch.long)
    time_features = torch.zeros((bsz, max_len, n_time), dtype=torch.float32)
    categorical_ids = torch.full((bsz, max_len, n_cat), _PAD_ID, dtype=torch.long)
    numeric_features = torch.zeros((bsz, max_len, n_num), dtype=torch.float32)
    padding_mask = torch.ones((bsz, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = int(item["length"])
        activity_ids[i, :n] = item["activity_ids"]
        time_features[i, :n] = item["time_features"]
        if n_cat:
            categorical_ids[i, :n] = item["categorical_ids"]
        if n_num:
            numeric_features[i, :n] = item["numeric_features"]
        padding_mask[i, :n] = False

    return {
        "activity_ids": activity_ids,
        "time_features": time_features,
        "categorical_ids": categorical_ids,
        "numeric_features": numeric_features,
        "padding_mask": padding_mask,
        "lengths": lengths,
    }


def collate_supervised(batch: list[dict[str, torch.Tensor]]) -> Batch:
    """Collate encoded traces + per-event targets, padding both consistently.

    Feature tensors are padded by :func:`collate_traces`; each known per-event
    target is padded with its task-specific pad value (``next_activity`` →
    ignore-index, ``remaining_time`` → 0, masked out by ``padding_mask``).
    """
    features = [{k: item[k] for k in _FEATURE_KEYS} for item in batch]
    out = collate_traces(features)
    max_len = out["activity_ids"].shape[1]

    for name, (pad_value, dtype) in _EVENT_TARGET_PAD.items():
        if name not in batch[0]:
            continue
        padded = torch.full((len(batch), max_len), pad_value, dtype=dtype)
        for i, item in enumerate(batch):
            n = int(item["length"])
            padded[i, :n] = item[name]
        out[name] = padded
    return out


def collate_outcome(batch: list[dict[str, torch.Tensor]]) -> Batch:
    """Collate encoded traces + a trace-level ``outcome`` label ``(B,)``."""
    out = collate_traces([{k: item[k] for k in _FEATURE_KEYS} for item in batch])
    out["outcome"] = torch.stack([item["outcome"] for item in batch])
    if "role_ids" in batch[0]:  # current-catalogue ids for the role channel
        max_len = out["activity_ids"].shape[1]
        role_ids = torch.full((len(batch), max_len), _PAD_ID, dtype=torch.long)
        for i, item in enumerate(batch):
            role_ids[i, : int(item["length"])] = item["role_ids"]
        out["role_ids"] = role_ids
    return out


def collate_dino(batch: list[dict[str, list[TraceView]]]) -> Batch:
    """Collate DINO multi-crop samples into per-crop-group batches.

    Each sample is ``{"global": [n_global views], "local": [n_local views]}``.
    Views are flattened **view-major** (view index outer, batch inner) and padded
    via :func:`collate_traces`, so the result can be reshaped to ``(n_views, B, ...)``
    downstream. Global and local groups are padded independently (different lengths).

    Returns ``{"global": <batch>, "local": <batch>, "n_global": int, "n_local": int}``
    (``"local"`` omitted when ``n_local == 0``).
    """
    n_global = len(batch[0]["global"])
    n_local = len(batch[0]["local"])

    global_items = [sample["global"][k] for k in range(n_global) for sample in batch]
    out: Batch = {
        "global": collate_traces(global_items),
        "n_global": n_global,
        "n_local": n_local,
    }
    if n_local:
        local_items = [sample["local"][k] for k in range(n_local) for sample in batch]
        out["local"] = collate_traces(local_items)
    return out
