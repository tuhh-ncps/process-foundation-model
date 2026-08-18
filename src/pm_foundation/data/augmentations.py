"""DINO-style trace augmentations (multi-view generation).

Augmentations operate on an **encoded** trace view (the tensor dict produced by
:func:`pm_foundation.data.preprocessing.encode_trace`) so a trace is encoded once
and then turned into several cheap views. Each operation preserves a valid,
non-empty trace. See ``docs/ssl_dino.md`` §3.

The :class:`MultiCropTraceAugmenter` returns ``2`` global + ``V`` local views per
trace: the teacher sees the global views, the student sees all of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pm_foundation.data.preprocessing import FeatureSpec

# A single encoded trace view: per-event tensors plus a scalar ``length``.
TraceView = dict[str, torch.Tensor]

# Per-event tensors that a crop slices along the time axis (everything except
# the scalar ``length``).
_PER_EVENT_KEYS = ("activity_ids", "time_features", "categorical_ids", "numeric_features")


def _rand_int(low: int, high: int) -> int:
    """Uniform integer in ``[low, high]`` (inclusive), via torch for worker seeding."""
    if high <= low:
        return low
    return int(torch.randint(low, high + 1, (1,)).item())


class TraceAugmentation(ABC):
    """Base class for a single trace-view transformation (operates in place)."""

    @abstractmethod
    def __call__(self, view: TraceView) -> TraceView:
        raise NotImplementedError


class SubTraceCrop(TraceAugmentation):
    """Sample a contiguous event window (global = long, local = short).

    The window length is a random fraction in ``[min_frac, max_frac]`` of the trace
    length (at least one event). Returns a fresh view (clones the slice).
    """

    def __init__(self, min_frac: float, max_frac: float) -> None:
        self.min_frac = min_frac
        self.max_frac = max_frac

    def __call__(self, view: TraceView) -> TraceView:
        length = int(view["length"])
        w_min = max(1, round(self.min_frac * length))
        w_max = max(w_min, min(length, round(self.max_frac * length)))
        window = _rand_int(w_min, w_max)
        start = _rand_int(0, length - window)
        sl = slice(start, start + window)
        out: TraceView = {k: view[k][sl].clone() for k in _PER_EVENT_KEYS}
        out["length"] = torch.tensor(window, dtype=torch.long)
        return out


class EventMasking(TraceAugmentation):
    """Replace a fraction of activity ids with the ``MASK`` token (BERT-style)."""

    def __init__(self, mask_prob: float, mask_id: int) -> None:
        self.mask_prob = mask_prob
        self.mask_id = mask_id

    def __call__(self, view: TraceView) -> TraceView:
        if self.mask_prob <= 0:
            return view
        ids = view["activity_ids"]
        mask = torch.rand(ids.shape) < self.mask_prob
        ids[mask] = self.mask_id
        return view


class AttributeDropout(TraceAugmentation):
    """Drop a random subset of attribute values.

    Dropped categoricals become ``UNK``; dropped numerics become ``0`` (the
    standardized mean), i.e. "value unknown".
    """

    def __init__(self, drop_prob: float, unk_id: int) -> None:
        self.drop_prob = drop_prob
        self.unk_id = unk_id

    def __call__(self, view: TraceView) -> TraceView:
        if self.drop_prob <= 0:
            return view
        cat = view["categorical_ids"]
        if cat.numel():
            cat[torch.rand(cat.shape) < self.drop_prob] = self.unk_id
        num = view["numeric_features"]
        if num.numel():
            num[torch.rand(num.shape) < self.drop_prob] = 0.0
        return view


class TemporalJitter(TraceAugmentation):
    """Perturb the magnitude time features (delta / time-since-start) with noise.

    Only the first ``columns`` of ``time_features`` are jittered; the cyclical
    encodings (already bounded in ``[-1, 1]``) are left untouched.
    """

    def __init__(self, sigma: float, columns: tuple[int, ...] = (0, 1)) -> None:
        self.sigma = sigma
        self.columns = columns

    def __call__(self, view: TraceView) -> TraceView:
        if self.sigma <= 0:
            return view
        tf = view["time_features"]
        if tf.numel():
            for c in self.columns:
                if c < tf.shape[-1]:
                    tf[:, c] += torch.randn(tf.shape[0]) * self.sigma
        return view


@dataclass
class MultiCropTraceAugmenter:
    """Produces ``n_global`` global + ``n_local`` local views per trace."""

    global_crop: SubTraceCrop
    local_crop: SubTraceCrop
    n_global: int = 2
    n_local: int = 6
    extra: tuple[TraceAugmentation, ...] = field(default_factory=tuple)

    def _make(self, view: TraceView, crop: SubTraceCrop) -> TraceView:
        out = crop(view)
        for aug in self.extra:
            out = aug(out)
        return out

    def __call__(self, view: TraceView) -> dict[str, list[TraceView]]:
        return {
            "global": [self._make(view, self.global_crop) for _ in range(self.n_global)],
            "local": [self._make(view, self.local_crop) for _ in range(self.n_local)],
        }


def build_augmenter(config: dict[str, Any], feature_spec: FeatureSpec) -> MultiCropTraceAugmenter:
    """Construct a :class:`MultiCropTraceAugmenter` from an ssl ``augmentation`` config."""
    g_lo, g_hi = config.get("global_crop_frac", (0.5, 1.0))
    l_lo, l_hi = config.get("local_crop_frac", (0.1, 0.4))
    vocab = feature_spec.activity_vocab
    extra: tuple[TraceAugmentation, ...] = (
        EventMasking(float(config.get("event_mask_prob", 0.15)), vocab.mask_id),
        AttributeDropout(float(config.get("attribute_dropout_prob", 0.1)), vocab.unk_id),
        TemporalJitter(float(config.get("temporal_jitter_sigma", 0.05))),
    )
    return MultiCropTraceAugmenter(
        global_crop=SubTraceCrop(float(g_lo), float(g_hi)),
        local_crop=SubTraceCrop(float(l_lo), float(l_hi)),
        n_global=int(config.get("n_global_views", 2)),
        n_local=int(config.get("n_local_views", 6)),
        extra=extra,
    )
