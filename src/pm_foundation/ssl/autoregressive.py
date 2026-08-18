"""Autoregressive next-event pretraining ("process-GPT").

A causal backbone predicts, at every event, the **next event's activity AND its
time-delta** from left context only. This is leak-free by construction (causal mask)
and teaches *both* control-flow and timing in one objective — the single-objective
recipe aimed at a reusable multi-task backbone. Pair with ``TraceEncoder(causal=True)``.

By default this is the plain **factorized** objective ``Loss = CE(next activity) +
time_weight·MSE(next Δt)`` (see ``docs/architecture.pdf``) — activity and time predicted
conditionally-independently. An **opt-in** ``objective="joint_nll"`` trains the marked
temporal-point-process joint likelihood ``L = -log P(a, Δt | prefix) = CE(a) - log f(Δt |
a)``: a proper log-normal Δt density *conditioned on the next activity* (teacher-forced), so
both terms are log-likelihoods and there is no arbitrary time weight (docs §16).

An **opt-in** ``time_dist="lognormal"`` (within the factorized objective) replaces the
Δt MSE (a point estimate) with the negative log-likelihood of a **log-normal** density
(the time head predicts ``mu, log_sigma`` of a Gaussian over the standardized log-Δt), so
the model captures timing *uncertainty* and can be sampled — MSE is the special case with
a fixed variance. An **opt-in** ``predict_end=True`` adds one extra
activity class, ``END`` (id ``n_activities``), and trains the last real event to predict it
— teaching the generative model *when a trace stops*, which a downstream autoregressive
rollout needs to terminate (``evaluation/rollout.py``). ``END`` is experimental and OFF by
default: the main pretraining pipeline trains exactly the diagram above. When enabled,
``END`` is an output class only — it never appears as an input event, so the input
embedding and the feature spec are untouched.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from pm_foundation.models.foundation_model import TraceBackbone
from pm_foundation.models.heads.regression import RegressionHead
from pm_foundation.ssl.teacher_student import EmaTeacher

IGNORE_INDEX = -100
# Column 0 of ``time_features`` is the standardized log inter-event delta.
_DELTA_COL = 0
_LOG_2PI = math.log(2 * math.pi)
_TIME_DISTS = ("point", "huber", "lognormal")
_OBJECTIVES = ("factorized", "joint_nll")


class JepaPredictor(nn.Module):
    """Dense multi-horizon latent predictor: EVERY position predicts 1..K steps ahead.

    Queries are **learned horizon embeddings only** (horizon ``j`` = "the event j steps
    after this position") — deliberately NOT the true future Δt or any attribute of the
    target events, so no information about the answer reaches the student side. Input:
    the (projected) causal state at every position; output: one predicted proj_dim
    vector per (position, horizon) pair.
    """

    def __init__(self, proj_dim: int, max_horizon: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.horizon_emb = nn.Embedding(max_horizon, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, L, p)`` -> predictions ``(B, L, K, p)`` for horizons ``1..K``."""
        bsz, length, p = z.shape
        k = self.horizon_emb.num_embeddings
        q = self.horizon_emb.weight.view(1, 1, k, p).expand(bsz, length, -1, -1)
        x = torch.cat([z.unsqueeze(2).expand(-1, -1, k, -1), q], dim=-1)
        return self.net(x)


class AutoregressiveLitModule(L.LightningModule):
    """Predicts the next event's activity + time-delta from causal context."""

    backbone: TraceBackbone

    time_cond_emb: nn.Embedding | None

    def __init__(
        self,
        backbone: TraceBackbone,
        n_activities: int,
        time_weight: float = 0.3,
        act_weight: float = 1.0,
        optimizer_cfg: dict[str, Any] | None = None,
        predict_end: bool = False,
        time_dist: str = "point",
        objective: str = "factorized",
        time_cond_dim: int = 32,
        huber_delta: float = 1.0,
        time_head_hidden: int = 0,
        remaining_time_weight: float = 0.0,
        remaining_head_hidden: int = 0,
        outcome_classes: int = 0,
        outcome_weight: float = 0.0,
        outcome_head_hidden: int = 0,
        jepa_weight: float = 0.0,
        jepa_block: int = 4,
        jepa_pred_hidden: int = 256,
        jepa_momentum: float = 0.996,
        jepa_proj_dim: int = 128,
        role_contrast_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
        # The joint NLL objective L = -log P(a, Δt | prefix) = CE(a) - log f(Δt | a) needs a
        # proper Δt *density* (log-normal) and conditions it on the next activity; both terms
        # are log-likelihoods (nats) so there is no arbitrary time weight.
        self.objective = objective
        if objective == "joint_nll":
            time_dist = "lognormal"
        if time_dist not in _TIME_DISTS:
            raise ValueError(f"time_dist must be one of {_TIME_DISTS}, got {time_dist!r}")
        self.backbone = backbone
        d_model = backbone.embedding.d_model
        self.n_activities = n_activities
        # Opt-in END class (id == n_activities): OFF by default so the main pipeline
        # trains the plain next-activity + next-Δt objective (see docs/architecture.pdf).
        self.end_id = n_activities if predict_end else None
        # Open-vocabulary CANDIDATE MATCHING replaces the fixed linear head when the
        # backbone has a role channel: logits(c) = scale * cos(q_i, e(c)) over the tied
        # ActivityEncoder table. New catalogue => new output space, backbone untouched.
        role_encoder = getattr(backbone.embedding, "role_encoder", None)
        self.uses_matching = role_encoder is not None
        if self.uses_matching:
            if predict_end:
                raise ValueError(
                    "predict_end is unsupported with candidate matching (no END candidate)"
                )
            self.match_query = nn.Linear(d_model, role_encoder.role_dim)
            self.match_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.activity_head = None
        else:
            self.match_query = None
            self.activity_head = nn.Linear(d_model, n_activities + (1 if predict_end else 0))
        self.role_contrast_weight = float(role_contrast_weight)
        # Time head: "point" -> a single Δt estimate trained with MSE (default, the
        # diagram). "huber" -> the same point estimate trained with Huber/smooth-L1 loss —
        # quadratic near zero, linear in the tails, so heavy-tailed long-wait outliers can't
        # dominate the timing gradient (docs §17). "lognormal" -> (mu, log_sigma) of a
        # Gaussian over the standardized log-Δt trained with NLL (docs §12/§13). For the
        # joint objective the head is additionally conditioned on the (teacher-forced) next
        # activity via an embedding — the P(Δt | a) factor of the joint likelihood.
        self.time_dist = time_dist
        self.huber_delta = huber_delta
        self.conditioned = objective == "joint_nll"
        self.time_cond_emb = nn.Embedding(n_activities, time_cond_dim) if self.conditioned else None
        cond_dim = time_cond_dim if self.conditioned else 0
        # Same architecture as the downstream NextTimeHead: a RegressionHead (linear when
        # time_head_hidden=0, else an MLP d->hidden->GELU->Dropout->out). So the pretext head
        # has the same capacity the probe will use — and warm-start transfers the output layer.
        time_out = 2 if self.time_dist == "lognormal" else 1
        self.time_head = RegressionHead(
            d_model + cond_dim, out_dim=time_out, hidden_dim=time_head_hidden or None
        )
        # Auxiliary per-event remaining-time pretext head (self-supervised; masked MAE on the
        # log1p seconds-to-case-end target). Same RegressionHead the downstream probe uses.
        self.remaining_time_weight = float(remaining_time_weight)
        self.remaining_head = (
            RegressionHead(d_model, out_dim=1, hidden_dim=remaining_head_hidden or None)
            if remaining_time_weight > 0
            else None
        )
        # Auxiliary trace-level OUTCOME pretext head (supervised; CE). Trained on the labeler's
        # stripped prefixes (a separate data path) so it's leak-free, and reads the pooled
        # trace embedding. Enabled only when outcome_classes > 0.
        self.outcome_weight = float(outcome_weight)
        self.outcome_head = (
            RegressionHead(d_model, out_dim=outcome_classes, hidden_dim=outcome_head_hidden or None)
            if outcome_classes > 0
            else None
        )
        # JEPA pretext (joint with AR): DENSE multi-horizon latent prediction — every causal
        # position predicts the EMA teacher's representation 1..jepa_block steps ahead. The
        # target is a latent, not a vocabulary token — vocabulary-agnostic on the prediction
        # side. The loss lives in a small PROJECTED space (student projector + EMA teacher
        # projector, BYOL-style) so the regression pressure cannot directly reshape the raw
        # backbone states the AR heads read. Leak-freedom: contexts are CAUSAL states
        # (attend to their prefix only); predictor queries are horizon embeddings (no future
        # info); the target side is an EMA teacher under no_grad/stop-grad.
        self.jepa_weight = float(jepa_weight)
        self.jepa_block = int(jepa_block)
        if self.jepa_weight > 0:
            self.jepa_proj = nn.Sequential(
                nn.Linear(d_model, jepa_proj_dim),
                nn.GELU(),
                nn.Linear(jepa_proj_dim, jepa_proj_dim),
            )
            self.jepa_teacher = EmaTeacher(copy.deepcopy(backbone), base_momentum=jepa_momentum)
            self.jepa_proj_ema = EmaTeacher(
                copy.deepcopy(self.jepa_proj), base_momentum=jepa_momentum
            )
            self.jepa_predictor = JepaPredictor(jepa_proj_dim, self.jepa_block, jepa_pred_hidden)
        else:
            self.jepa_proj = None
            self.jepa_teacher = None
            self.jepa_proj_ema = None
            self.jepa_predictor = None
        self.time_weight = time_weight
        self.act_weight = float(act_weight)  # 0 disables the next-activity loss (ablation)
        self.optimizer_cfg = optimizer_cfg or {}

    def _time_loss(
        self, time_out: torch.Tensor, next_delta: torch.Tensor, valid_time: torch.Tensor
    ) -> torch.Tensor:
        """Δt loss on valid positions: MSE (point), Huber (huber), or Gaussian NLL (lognormal)."""
        if not valid_time.any():
            return torch.zeros((), device=time_out.device)
        target = next_delta[valid_time]
        if self.time_dist == "lognormal":
            mu = time_out[..., 0][valid_time]
            log_sigma = time_out[..., 1][valid_time].clamp(-5.0, 3.0)
            nll = 0.5 * ((target - mu) * torch.exp(-log_sigma)) ** 2 + log_sigma + 0.5 * _LOG_2PI
            return nll.mean()
        if self.time_dist == "huber":
            return F.huber_loss(time_out[..., 0][valid_time], target, delta=self.huber_delta)
        return F.mse_loss(time_out[..., 0][valid_time], target)

    def _jepa_loss(
        self, states: torch.Tensor, batch: dict[str, Any], padding_mask: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Dense multi-horizon JEPA loss (leak-free). ``None`` if no trace is long enough.

        EVERY real position ``i`` predicts the teacher's PROJECTED latent at ``i+j`` for all
        horizons ``j = 1..jepa_block`` that stay inside the trace — the latent-space analog
        of the per-position AR heads (~L pairs per trace instead of one random cut, so short
        traces contribute proper signal and no single position soaks up the gradient).
        Context = the STUDENT's causal state at ``i`` (attends only to events ``<= i`` — the
        causal mask is the leakage guarantee); targets come from the EMA teacher + EMA
        projector under no_grad. Both sides are parameter-free layer-normed; loss is
        smooth-L1 over all valid (trace, position, horizon) triples.

        Returns ``{"loss", "tgt_std", "pred_std", "n_pairs"}`` — the stds are per-dim std
        across pairs (collapse telemetry: healthy reps keep std well above 0).
        """
        assert self.jepa_predictor is not None and self.jepa_proj is not None
        assert self.jepa_teacher is not None and self.jepa_proj_ema is not None
        max_len = states.shape[1]
        lengths = (~padding_mask).sum(dim=1)  # (B,)
        if int(lengths.max()) < 2:
            return None

        z = self.jepa_proj(states)  # (B, L, p) — student projection
        pred = self.jepa_predictor(z)  # (B, L, K, p), horizon j at index j-1

        with torch.no_grad():  # stop-grad target side: EMA backbone -> EMA projector
            t_states = self.jepa_teacher.teacher.forward_batch(batch).event_states
            tgt_full = self.jepa_proj_ema.teacher(t_states)  # (B, L, p)
        p = tgt_full.shape[-1]

        pos = torch.arange(max_len, device=states.device)
        preds, tgts = [], []
        for j in range(1, self.jepa_block + 1):
            n = max_len - j
            if n <= 0:
                break
            # position i predicts i+j; valid iff i+j is a real event: i + j <= length - 1
            valid = (pos[:n].unsqueeze(0) + j) <= (lengths - 1).unsqueeze(1)  # (B, n)
            if valid.any():
                preds.append(pred[:, :n, j - 1][valid])
                tgts.append(tgt_full[:, j:, :][valid])
        if not preds:
            return None
        pred_cat = torch.cat(preds)  # (P, p)
        tgt_cat = torch.cat(tgts)  # (P, p)
        pred_n = F.layer_norm(pred_cat, (p,))
        tgt_n = F.layer_norm(tgt_cat, (p,))
        with torch.no_grad():  # collapse telemetry: per-dim std across pairs, averaged
            tgt_std = tgt_n.std(dim=0).mean()
            pred_std = pred_n.std(dim=0).mean()
        return {
            "loss": F.smooth_l1_loss(pred_n, tgt_n),
            "tgt_std": tgt_std,
            "pred_std": pred_std,
            "n_pairs": torch.tensor(pred_cat.shape[0]),
        }

    @staticmethod
    def build_targets(
        activity_ids: torch.Tensor,
        time_features: torch.Tensor,
        padding_mask: torch.Tensor,
        end_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Next-event targets: ``(next_activity, next_delta, valid_activity, valid_time)``.

        At position ``i`` the targets are event ``i+1``'s activity and delta; the last real
        event (no successor) and pad positions are ignored — this is the default. When
        ``end_id`` is given, the **last real event** of each trace instead predicts
        ``end_id`` (learning to stop), so ``valid_activity`` covers real successors **and**
        the END step while ``valid_time`` stays on real successors only (there is no Δt into
        END). With ``end_id=None`` the two masks coincide. All shapes ``(B, L)``.
        """
        next_activity = torch.full_like(activity_ids, IGNORE_INDEX)
        next_activity[:, :-1] = activity_ids[:, 1:]
        next_is_pad = torch.zeros_like(padding_mask)
        next_is_pad[:, :-1] = padding_mask[:, 1:]
        next_activity[next_is_pad] = IGNORE_INDEX
        next_activity[padding_mask] = IGNORE_INDEX
        valid_time = next_activity != IGNORE_INDEX  # a real successor exists

        delta = time_features[..., _DELTA_COL]
        next_delta = torch.zeros_like(delta)
        next_delta[:, :-1] = delta[:, 1:]

        if end_id is None:
            return next_activity, next_delta, valid_time, valid_time
        # Last real event of each trace -> predict END (its Δt stays out of valid_time).
        lengths = (~padding_mask).sum(dim=1)
        rows = torch.arange(activity_ids.shape[0], device=activity_ids.device)
        last_idx = (lengths - 1).clamp(min=0)
        has_events = lengths > 0
        next_activity[rows[has_events], last_idx[has_events]] = end_id
        valid_activity = next_activity != IGNORE_INDEX
        return next_activity, next_delta, valid_activity, valid_time

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # With the outcome pretext on, a CombinedLoader yields {"ar":..., "outcome":...};
        # otherwise the batch is the AR feature dict directly.
        outcome_batch = None
        if "activity_ids" not in batch:
            outcome_batch = batch.get("outcome")
            batch = batch["ar"]
        padding_mask = batch["padding_mask"]
        out = self.backbone.forward_batch(batch)  # causal event states
        states = out.event_states  # (B, L, d)
        if self.uses_matching:
            # Candidate matching over the tied ActivityEncoder table (open vocabulary).
            table = self.backbone.embedding.role_encoder()  # (V, r)
            q = F.normalize(self.match_query(states), dim=-1)
            cand = F.normalize(table, dim=-1, eps=1e-8)
            act_logits = self.match_logit_scale.exp().clamp(max=100.0) * (q @ cand.t())
        else:
            act_logits = self.activity_head(states)  # (B, L, n_classes)

        next_act, next_delta, valid_act, valid_time = self.build_targets(
            batch["activity_ids"], batch["time_features"], padding_mask, self.end_id
        )

        # Δt head input: causal state, plus the (teacher-forced) next activity for the joint
        # objective's P(Δt | a) factor. next_act is clamped in-range for the embedding; END /
        # ignore positions are excluded from the time loss by valid_time anyway.
        if self.conditioned:
            assert self.time_cond_emb is not None
            cond = self.time_cond_emb(next_act.clamp(0, self.n_activities - 1))
            time_out = self.time_head(torch.cat([states, cond], dim=-1))
        else:
            time_out = self.time_head(states)

        act_loss = F.cross_entropy(
            act_logits.reshape(-1, act_logits.shape[-1]),
            next_act.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        time_loss = self._time_loss(time_out, next_delta, valid_time)
        # Joint NLL: both terms are log-likelihoods, so weight 1.0 (no arbitrary balance).
        weight = 1.0 if self.objective == "joint_nll" else self.time_weight
        total = self.act_weight * act_loss + weight * time_loss

        # Auxiliary remaining-time pretext (masked MAE over non-pad positions), if enabled and
        # the batch carries the target (SupervisedTraceDataset provides it).
        rem_loss = None
        if self.remaining_head is not None and "remaining_time" in batch:
            rem_pred = self.remaining_head(states).squeeze(-1)  # (B, L)
            rem_valid = ~padding_mask
            rem_loss = F.l1_loss(rem_pred[rem_valid], batch["remaining_time"][rem_valid])
            total = total + self.remaining_time_weight * rem_loss

        # Auxiliary outcome pretext (CE) on the labeler's stripped prefixes — a leak-free second
        # data path; the head reads the pooled trace embedding of those prefixes.
        out_loss = None
        if self.outcome_head is not None and outcome_batch is not None:
            o = self.backbone.forward_batch(outcome_batch)
            logits = self.outcome_head(o.trace_embedding)  # (B, n_classes)
            out_loss = F.cross_entropy(logits, outcome_batch["outcome"])
            total = total + self.outcome_weight * out_loss

        # JEPA pretext (joint with AR): dense multi-horizon latent prediction against the
        # EMA teacher. Reuses the SAME student forward (causal states) — no extra student pass.
        jepa = None
        if self.jepa_predictor is not None:
            jepa = self._jepa_loss(states, batch, padding_mask)
            if jepa is not None:
                total = total + self.jepa_weight * jepa["loss"]

        # Role-space contrastive (InfoNCE over augmented DFG views): shapes the activity
        # encoder independently of the dataset-specific heads; prevents role collapse.
        role_loss = None
        role_encoder = getattr(self.backbone.embedding, "role_encoder", None)
        if role_encoder is not None and self.role_contrast_weight > 0:
            role_loss = role_encoder.contrastive_loss()
            if role_loss is not None:
                total = total + self.role_contrast_weight * role_loss

        bs = padding_mask.shape[0]
        # Log EPOCH MEANS (on_epoch=True), not the last mini-batch. With length-bucketed batches
        # the final batch of an epoch is length-homogeneous (all-short -> acc≈1.0, all-long -> hard),
        # so last-batch logging makes the learning curve swing wildly; the epoch mean is stable.
        log_kw = {"on_step": False, "on_epoch": True, "batch_size": bs}
        self.log("train/ar_loss", total, prog_bar=True, **log_kw)
        # Log a head's loss only when it's ACTIVE (weight > 0). A disabled head still gets computed
        # but flat-lines with no gradient; logging it just clutters the curve.
        time_active = self.objective == "joint_nll" or self.time_weight > 0
        if self.act_weight > 0:
            self.log("train/ar_act_loss", act_loss, **log_kw)
        if time_active:
            self.log("train/ar_time_loss", time_loss, **log_kw)
        if rem_loss is not None:
            self.log("train/ar_remaining_loss", rem_loss, **log_kw)
        if out_loss is not None:
            self.log("train/ar_outcome_loss", out_loss, **log_kw)
        if role_loss is not None:
            self.log("train/ar_role_loss", role_loss, **log_kw)
        if jepa is not None:
            self.log("train/ar_jepa_loss", jepa["loss"], **log_kw)
            # Collapse telemetry (not *_loss -> kept off the learning curve, still in W&B):
            # healthy target/pred std stays well above 0; a slide toward 0 means collapse.
            self.log("train/ar_jepa_tgt_std", jepa["tgt_std"], **log_kw)
            self.log("train/ar_jepa_pred_std", jepa["pred_std"], **log_kw)
        with torch.no_grad():
            if self.act_weight > 0 and valid_act.any():
                acc = (act_logits.argmax(-1)[valid_act] == next_act[valid_act]).float().mean()
                self.log("train/ar_next_act_acc", acc, **log_kw)
        return total

    def set_role_graph(self, graph: dict[str, torch.Tensor]) -> None:
        """Install the TRAINING corpus's role graph (student AND the JEPA teacher copy).

        ``graph`` comes from ``fit_role_graph`` on the pretrain TRAIN split — never on
        traces that will be scored (see the leakage contract in data/roles.py).
        """
        role_encoder = getattr(self.backbone.embedding, "role_encoder", None)
        if role_encoder is None:
            raise ValueError("set_role_graph called but the backbone has no role encoder")
        role_encoder.set_graph(graph)
        if self.jepa_teacher is not None:  # keep the EMA teacher's copy consistent
            self.jepa_teacher.teacher.embedding.role_encoder.set_graph(graph)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        # EMA update AFTER the optimizer step: teacher tracks the student on a cosine
        # momentum schedule (base -> 1.0). Identical on every DDP rank (students are synced),
        # so teachers stay in sync without gradient communication.
        if self.jepa_teacher is not None:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            m = self.jepa_teacher.momentum_at(self.trainer.global_step, total)
            self.jepa_teacher.update(self.backbone, m)
            self.jepa_proj_ema.update(self.jepa_proj, m)  # keep the projector pair in sync

    def configure_optimizers(self) -> Any:
        cfg = self.optimizer_cfg
        optimizer = torch.optim.AdamW(
            # Exclude the frozen EMA teacher (requires_grad=False) — it is updated by
            # momentum in on_train_batch_end, never by the optimizer.
            [p for p in self.parameters() if p.requires_grad],
            lr=float(cfg.get("lr", 5e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs = self.trainer.max_epochs or 1
        warmup = min(
            int(cfg.get("warmup_epochs", 0) * total_steps / max(1, max_epochs)),
            max(0, total_steps - 1),
        )

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_autoregressive_module(
    model_cfg: dict[str, Any], ar_cfg: dict[str, Any], feature_spec: Any
) -> AutoregressiveLitModule:
    """Assemble an AR module on a **causal** backbone (forced for leak-free training).

    ``ar_cfg["objective"]`` selects the training loss: ``"factorized"`` (default) is
    ``CE(a) + time_weight·time_loss``; ``"joint_nll"`` is the marked-TPP joint likelihood
    ``L = -log P(a, Δt | prefix) = CE(a) - log f(Δt | a)`` (forces a log-normal Δt density
    conditioned on the next activity, weight 1.0). ``ar_cfg["predict_end"]`` (default
    ``False``) opts into the experimental END class used by rollout; ``ar_cfg["time_dist"]``
    (``"point"``/``"lognormal"``) selects the Δt loss when ``objective="factorized"``. The
    main pipeline leaves everything at defaults (the plain objective in
    ``docs/architecture.pdf``).
    """
    causal_cfg = {**model_cfg, "causal": True}
    backbone = TraceBackbone.from_config(causal_cfg, feature_spec)
    return AutoregressiveLitModule(
        backbone,
        n_activities=feature_spec.n_activities,
        time_weight=float(ar_cfg.get("time_weight", 0.3)),
        act_weight=float(ar_cfg.get("act_weight", 1.0)),
        optimizer_cfg=ar_cfg.get("optimizer"),
        predict_end=bool(ar_cfg.get("predict_end", False)),
        time_dist=str(ar_cfg.get("time_dist", "point")),
        objective=str(ar_cfg.get("objective", "factorized")),
        time_cond_dim=int(ar_cfg.get("time_cond_dim", 32)),
        huber_delta=float(ar_cfg.get("huber_delta", 1.0)),
        time_head_hidden=int(ar_cfg.get("time_head_hidden", 0)),
        remaining_time_weight=float(ar_cfg.get("remaining_time_weight", 0.0)),
        remaining_head_hidden=int(ar_cfg.get("remaining_head_hidden", 0)),
        outcome_classes=int(ar_cfg.get("outcome_classes", 0)),
        outcome_weight=float(ar_cfg.get("outcome_weight", 0.0)),
        outcome_head_hidden=int(ar_cfg.get("outcome_head_hidden", 0)),
        jepa_weight=float(ar_cfg.get("jepa_weight", 0.0)),
        jepa_block=int(ar_cfg.get("jepa_block", 4)),
        jepa_pred_hidden=int(ar_cfg.get("jepa_pred_hidden", 256)),
        jepa_momentum=float(ar_cfg.get("jepa_momentum", 0.996)),
        jepa_proj_dim=int(ar_cfg.get("jepa_proj_dim", 128)),
        role_contrast_weight=float(ar_cfg.get("role_contrast_weight", 0.0)),
    )
