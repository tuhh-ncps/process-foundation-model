"""Remaining-time via autoregressive suffix generation ("process-GPT rollout").

Instead of regressing remaining time directly, roll the generative AR model forward from
an observed prefix: at each step predict the next activity and next Δt, append the
predicted event, and repeat until the model emits ``END`` (or a step cap). The remaining
time is the sum of the predicted future Δt. This composes the two signals the AR backbone
learns *well* (next-activity, next-Δt) to estimate the quantity it regresses *poorly*.

Requires an END-aware AR run (``ar_heads.pt`` produced by ``pretrain_autoregressive`` with
the END class). Supports greedy decoding and Monte-Carlo rollout (sample K suffixes per
prefix and average the summed times, which tames branching and exposure bias).

The per-event time features mirror ``preprocessing._time_features`` exactly; because each
event's features depend only on its own / previous timestamp and the trace start, they are
computed once when an event is appended and cached (no re-encode cost per step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from pm_foundation.data.preprocessing import N_TIME_FEATURES, FeatureSpec
from pm_foundation.models import TraceBackbone

if TYPE_CHECKING:
    from collections.abc import Sequence

_MAX_DELTA_SEC = 365 * 86400.0  # clamp a runaway predicted gap to one year


@dataclass
class _RolloutState:
    """Mutable per-rollout state during batched suffix generation."""

    acts: list[int]
    feats: list[list[float]]  # one 8-dim feature row per event
    last_ts: datetime
    t0: datetime
    remaining: float = 0.0
    n_steps: int = 0
    done: bool = False
    origin: int = 0  # index of the source prefix (for Monte-Carlo averaging)


@dataclass
class SuffixRemainingTime:
    """Generative remaining-time estimator over a frozen END-aware AR model."""

    backbone: TraceBackbone
    activity_head: nn.Module
    time_head: nn.Module
    spec: FeatureSpec
    end_id: int
    max_steps: int = 64
    device: str = "cpu"
    _stats: tuple[float, float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        self.backbone = self.backbone.to(self.device).eval()
        self.activity_head = self.activity_head.to(self.device).eval()
        self.time_head = self.time_head.to(self.device).eval()
        self._stats = (
            self.spec.time_mean["log_delta"],
            self.spec.time_std["log_delta"],
            self.spec.time_mean["log_since"],
            self.spec.time_std["log_since"],
        )

    @classmethod
    def from_run_dir(
        cls, run_dir: str | Path, *, device: str = "cpu", max_steps: int = 64
    ) -> SuffixRemainingTime:
        """Load a backbone + persisted AR heads from a pretraining run directory."""
        run_dir = Path(run_dir)
        spec = FeatureSpec.load(run_dir / "feature_spec.json")
        model_cfg = _model_cfg_from_manifest(run_dir)
        backbone = TraceBackbone.from_config(model_cfg, spec)
        backbone.load_state_dict(torch.load(run_dir / "backbone.pt", map_location="cpu"))
        heads = torch.load(run_dir / "ar_heads.pt", map_location="cpu")
        d_model = backbone.embedding.d_model
        activity_head = nn.Linear(d_model, int(heads["n_activities"]) + 1)
        activity_head.load_state_dict(heads["activity_head"])
        time_head = nn.Linear(d_model, 1)
        time_head.load_state_dict(heads["time_head"])
        return cls(
            backbone,
            activity_head,
            time_head,
            spec,
            int(heads["end_id"]),
            max_steps=max_steps,
            device=device,
        )

    # ----- feature reconstruction (mirrors preprocessing._time_features) ------ #
    def _event_features(self, ts: datetime, prev_ts: datetime | None, t0: datetime) -> list[float]:
        dm, ds, sm, ss = self._stats
        delta = 0.0 if prev_ts is None else max((ts - prev_ts).total_seconds(), 0.0)
        since = max((ts - t0).total_seconds(), 0.0)
        hour = 2 * math.pi * ts.hour / 24
        dow = 2 * math.pi * ts.weekday() / 7
        month = 2 * math.pi * (ts.month - 1) / 12
        return [
            (math.log1p(delta) - dm) / ds,
            (math.log1p(since) - sm) / ss,
            math.sin(hour),
            math.cos(hour),
            math.sin(dow),
            math.cos(dow),
            math.sin(month),
            math.cos(month),
        ]

    def _init_state(
        self, acts: Sequence[int], times: Sequence[datetime], origin: int
    ) -> _RolloutState:
        t0 = times[0]
        feats, prev = [], None
        for ts in times:
            feats.append(self._event_features(ts, prev, t0))
            prev = ts
        return _RolloutState(acts=list(acts), feats=feats, last_ts=times[-1], t0=t0, origin=origin)

    def _batch(self, states: list[_RolloutState]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        b = len(states)
        lengths = torch.tensor([len(s.acts) for s in states])
        lmax = int(lengths.max())
        activity_ids = torch.zeros(b, lmax, dtype=torch.long)
        time_features = torch.zeros(b, lmax, N_TIME_FEATURES)
        padding_mask = torch.ones(b, lmax, dtype=torch.bool)
        for i, s in enumerate(states):
            n = len(s.acts)
            activity_ids[i, :n] = torch.tensor(s.acts, dtype=torch.long)
            time_features[i, :n] = torch.tensor(s.feats)
            padding_mask[i, :n] = False
        batch = {
            "activity_ids": activity_ids,
            "time_features": time_features,
            "categorical_ids": torch.zeros(
                b, lmax, len(self.spec.categorical_cardinalities), dtype=torch.long
            ),
            "numeric_features": torch.zeros(b, lmax, self.spec.n_numeric),
            "padding_mask": padding_mask,
        }
        return {k: v.to(self.device) for k, v in batch.items()}, lengths

    @torch.no_grad()
    def estimate(
        self,
        prefixes: Sequence[tuple[Sequence[int], Sequence[datetime]]],
        *,
        samples: int = 1,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """Estimated remaining seconds per prefix.

        ``prefixes`` are ``(activity_ids, timestamps)`` for each observed prefix. With
        ``samples > 1`` each prefix is rolled out ``samples`` times with sampled activities
        and the summed times are averaged (Monte-Carlo). ``samples == 1`` is greedy.
        """
        dm, ds, _, _ = self._stats
        gen = torch.Generator(device="cpu").manual_seed(seed)
        states = [
            self._init_state(a, t, origin=i)
            for i, (a, t) in enumerate(prefixes)
            for _ in range(samples)
        ]

        active = list(range(len(states)))
        for _ in range(self.max_steps):
            if not active:
                break
            sub = [states[i] for i in active]
            batch, lengths = self._batch(sub)
            enc = self.backbone.forward_batch(batch)
            last = enc.event_states[torch.arange(len(sub)), (lengths - 1).to(self.device)]
            logits = self.activity_head(last)  # (b, n_act + 1)
            time_out = self.time_head(last)  # (b, 1) point or (b, 2) log-normal (mu, log_sigma)
            z_delta = time_out[..., 0].double().cpu()  # point estimate / log-normal mean of log-Δt
            if samples > 1:
                probs = torch.softmax(logits / temperature, dim=-1).cpu()
                nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
            else:
                nxt = logits.argmax(dim=-1).cpu()

            still_active: list[int] = []
            for k, gi in enumerate(active):
                s = states[gi]
                act = int(nxt[k])
                if act == self.end_id:
                    s.done = True
                    continue
                delta_sec = min(math.expm1(float(z_delta[k]) * ds + dm), _MAX_DELTA_SEC)
                delta_sec = max(delta_sec, 0.0)
                s.remaining += delta_sec
                ts = s.last_ts + timedelta(seconds=delta_sec)
                s.feats.append(self._event_features(ts, s.last_ts, s.t0))
                s.acts.append(act)
                s.last_ts = ts
                s.n_steps += 1
                still_active.append(gi)
            active = still_active

        out = torch.zeros(len(prefixes), dtype=torch.double)
        counts = torch.zeros(len(prefixes), dtype=torch.double)
        for s in states:
            out[s.origin] += s.remaining
            counts[s.origin] += 1
        return out / counts.clamp(min=1)


def _model_cfg_from_manifest(run_dir: Path) -> dict[str, object]:
    import json

    cfg = dict(json.loads((run_dir / "manifest.json").read_text())["config"]["model"])
    cfg["causal"] = True
    return cfg
