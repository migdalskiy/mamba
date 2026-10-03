"""Training callbacks: early stopping, grad-norm logging per layer group, probe capture."""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch


class Callback:
    def on_train_start(self, trainer):
        pass

    def on_step_end(self, trainer, step: int, logs: Dict):
        pass

    def on_eval(self, trainer, step: int, metrics: Dict):
        pass

    def on_epoch_end(self, trainer, epoch: int):
        pass

    def on_train_end(self, trainer):
        pass


class EarlyStopping(Callback):
    """Stop when ``monitor`` (higher is better) has not improved for ``patience`` evaluations."""

    def __init__(self, monitor: str = "macro_f1", patience: int = 2, min_delta: float = 0.0):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.best, self.bad = -float("inf"), 0

    def on_eval(self, trainer, step, metrics):
        value = metrics[self.monitor]
        if value > self.best + self.min_delta:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                trainer.should_stop = True


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_group(name: str) -> str:
    m = _LAYER_RE.search(name)
    if m:
        return f"layer_{int(m.group(1)):03d}"
    if "head" in name or "pooler" in name:
        return "head"
    if "embedding" in name:
        return "embedding"
    return "other"


class GradNormLogger(Callback):
    """Per-layer-group gradient norms every ``every`` steps (feeds the layer x step x grad-norm
    surface used to spot dt / A instability), plus the dt / A_log groups separately."""

    def __init__(self, every: int = 10):
        self.every = every
        self.rows: List[Dict] = []

    def on_step_end(self, trainer, step, logs):
        if step % self.every:
            return
        sq = defaultdict(float)
        for n, p in trainer.model.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.detach().float().norm()) ** 2
            sq[layer_group(n)] += g
            leaf = n.split(".")[-1]
            if leaf in ("A_log", "dt_bias"):
                sq[f"{layer_group(n)}/{leaf}"] += g
        for group, v in sq.items():
            self.rows.append({"step": step, "group": group, "grad_norm": v ** 0.5})


class ProbeCaptureCallback(Callback):
    """Capture internals on a fixed probe batch at the given fractions of training {0, .25, .5, 1}."""

    def __init__(self, recorder_factory, probe_batch, at: Sequence[float] = (0.0, 0.25, 0.5, 1.0), sink=None):
        self.factory, self.batch, self.at, self.sink = recorder_factory, probe_batch, sorted(at), sink
        self.done = set()

    def _maybe(self, trainer, step):
        frac = step / max(trainer.total_steps, 1)
        for a in self.at:
            if a not in self.done and frac >= a:
                self.done.add(a)
                rec = self.factory(trainer.model)
                rec.run(self.batch.to(trainer.device))
                if self.sink is not None:
                    self.sink(rec, step, a)

    def on_train_start(self, trainer):
        self._maybe(trainer, 0)

    def on_step_end(self, trainer, step, logs):
        self._maybe(trainer, step)

    def on_train_end(self, trainer):
        self._maybe(trainer, trainer.total_steps)
