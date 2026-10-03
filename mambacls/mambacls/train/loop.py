"""Training loop: bf16 autocast with fp32 master weights, grad clipping, periodic evaluation,
early stopping on validation macro-F1, best-checkpoint restore (spec §3.2, §6)."""

import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from mambacls.eval.metrics import compute_metrics, predict
from mambacls.train.callbacks import Callback, EarlyStopping
from mambacls.train.distill import kd_loss
from mambacls.train.optim import NO_DECAY_DEFAULT, build_optimizer, warmup_cosine

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int = 3
    lr_backbone: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    no_decay: Sequence[str] = NO_DECAY_DEFAULT
    warmup_ratio: float = 0.06
    schedule: str = "cosine"
    clip: float = 1.0
    precision: str = "bf16_amp"  # bf16 autocast on CUDA with fp32 params; "fp32" disables
    eval_every: float = 0.25  # fraction of an epoch
    patience: int = 2
    monitor: str = "macro_f1"
    max_steps: Optional[int] = None
    grad_accum: int = 1
    seed: int = 0
    deterministic: bool = False
    kd_tau: float = 2.0
    kd_weight: float = 0.5

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class TrainResult:
    history: pd.DataFrame
    best_metrics: Dict
    best_step: int
    total_steps: int
    train_time_s: float
    train_tokens: int
    nan_steps: int = 0

    @property
    def train_tokens_per_s(self):
        return self.train_tokens / max(self.train_time_s, 1e-9)


class Trainer:
    def __init__(self, model, cfg: TrainConfig, train_loader, val_loader, n_classes: int, device=None,
                 callbacks: Optional[List[Callback]] = None, teacher=None, multilabel: bool = False):
        self.model, self.cfg, self.train_loader, self.val_loader = model, cfg, train_loader, val_loader
        self.n_classes, self.multilabel, self.teacher = n_classes, multilabel, teacher
        self.device = device or next(model.parameters()).device
        self.callbacks = list(callbacks or [])
        if not any(isinstance(c, EarlyStopping) for c in self.callbacks) and cfg.patience:
            self.callbacks.append(EarlyStopping(cfg.monitor, cfg.patience))
        steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
        self.total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = build_optimizer(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay, cfg.no_decay)
        self.scheduler = warmup_cosine(self.optimizer, self.total_steps, cfg.warmup_ratio, cfg.schedule)
        self.should_stop = False
        self.history: List[Dict] = []

    @property
    def autocast_dtype(self):
        return torch.bfloat16 if self.cfg.precision == "bf16_amp" and self.device.type == "cuda" else None

    def _autocast(self):
        dt = self.autocast_dtype
        return torch.autocast("cuda", dtype=dt) if dt is not None else torch.autocast("cpu", enabled=False)

    def _trainable_state(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def evaluate(self):
        p = predict(self.model, self.val_loader, device=self.device, autocast_dtype=self.autocast_dtype)
        return compute_metrics(p.logits, p.labels, self.n_classes, multilabel=self.multilabel)

    def fit(self) -> TrainResult:
        set_seed(self.cfg.seed, self.cfg.deterministic)
        cfg, model = self.cfg, self.model
        eval_interval = max(1, int(round(cfg.eval_every * self.steps_per_epoch))) if cfg.eval_every else self.steps_per_epoch
        best, best_state, best_step = None, None, 0
        step, tokens, nan_steps = 0, 0, 0
        for cb in self.callbacks:
            cb.on_train_start(self)
        t0 = time.perf_counter()
        for epoch in range(cfg.epochs if cfg.max_steps is None else 10 ** 9):
            model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                with self._autocast():
                    out = model(batch)
                    loss = out.loss
                    if self.teacher is not None:
                        with torch.no_grad():
                            t_logits = self.teacher(batch).logits.float()
                        loss = kd_loss(out.logits.float(), t_logits, batch.labels, cfg.kd_tau, cfg.kd_weight, self.multilabel)
                if not torch.isfinite(loss):
                    nan_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    log.warning("non-finite loss at step %d; batch skipped", step)
                    continue
                (loss / cfg.grad_accum).backward()
                tokens += int(batch.lengths.sum())
                if (i + 1) % cfg.grad_accum:
                    continue
                params = [p for g in self.optimizer.param_groups for p in g["params"]]
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip) if cfg.clip else torch.tensor(float("nan"))
                for cb in self.callbacks:  # grads are still populated here
                    cb.on_step_end(self, step + 1, {"loss": float(loss.detach())})
                self.optimizer.step()
                model.adapter.after_optimizer_step() if hasattr(model, "adapter") else None
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                row = {"step": step, "epoch": epoch, "split": "train", "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                       "lr_backbone": self._lr("backbone"), "lr_head": self._lr("head")}
                self.history.append(row)
                if step % eval_interval == 0 or step == self.total_steps:
                    metrics = self.evaluate()
                    self.history.append({"step": step, "epoch": epoch, "split": "val", **metrics})
                    for cb in self.callbacks:
                        cb.on_eval(self, step, metrics)
                    if best is None or metrics[cfg.monitor] > best[cfg.monitor]:
                        best, best_state, best_step = metrics, self._trainable_state(), step
                    model.train()
                if self.should_stop or step >= self.total_steps:
                    break
            if hasattr(model, "adapter"):
                model.adapter.on_epoch_end(epoch, self)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)
            if self.should_stop or step >= self.total_steps:
                break
        train_time = time.perf_counter() - t0
        if best_state is not None:
            with torch.no_grad():
                params = dict(model.named_parameters())
                for n, v in best_state.items():
                    params[n].copy_(v)
        for cb in self.callbacks:
            cb.on_train_end(self)
        return TrainResult(pd.DataFrame(self.history), best or {}, best_step, step, train_time, tokens, nan_steps)

    def _lr(self, tag):
        for g in self.optimizer.param_groups:
            if g.get("group", "").startswith(tag):
                return g["lr"]
        return float("nan")


# ----------------------------------------------------------------------------------------------
# Linear probe on cached features (Phase A)
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device=None, per_layer: bool = False, return_ids: bool = False):
    """Pooled features with a frozen backbone. ``per_layer`` returns (N, n_layers, D) features
    pooled with the pooler's base strategy on every layer (for scalar_mix probes)."""
    model.eval()
    feats, labels, ids = [], [], []
    for batch in loader:
        batch = batch.to(device) if device is not None else batch
        if per_layer:
            out = model.backbone(batch.input_ids, lengths=batch.lengths, return_all_layers=True)
            base = getattr(model.pooler, "base", model.pooler)
            feats.append(torch.stack([base(h, out.mask).pooled for h in out.layers], dim=1).float().cpu())
        else:
            pool, _ = model.encode(batch)
            feats.append(pool.pooled.float().cpu())
        labels.append(batch.labels.cpu())
        if batch.example_ids is not None:
            ids.append(batch.example_ids.cpu())
    if return_ids:
        return torch.cat(feats), torch.cat(labels), torch.cat(ids) if ids else None
    return torch.cat(feats), torch.cat(labels)


def fit_probe_head(train_x, train_y, n_classes, val_x=None, val_y=None, weight_decay: float = 1e-4, epochs: int = 100,
                   lr: float = 1e-2, seed: int = 0, scalar_mix: bool = False):
    """Logistic regression (optionally with a learned softmax mix over layers) on cached features."""
    torch.manual_seed(seed)
    d = train_x.shape[-1]
    mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-6)
    head = torch.nn.Linear(d, n_classes)
    mix = torch.nn.Parameter(torch.zeros(train_x.shape[1])) if scalar_mix else None
    params = list(head.parameters()) + ([mix] if mix is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def fwd(x):
        x = (x - mean) / std
        if mix is not None:
            x = torch.einsum("k,nkd->nd", torch.softmax(mix, 0), x)
        return head(x)

    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(fwd(train_x), train_y)
        loss.backward()
        opt.step()
    result = {"head": head, "mix": None if mix is None else torch.softmax(mix.detach(), 0), "predict": fwd}
    if val_x is not None:
        with torch.no_grad():
            result["val_logits"] = fwd(val_x).numpy()
            result["val_metrics"] = compute_metrics(result["val_logits"], val_y.numpy(), n_classes)
    return result
