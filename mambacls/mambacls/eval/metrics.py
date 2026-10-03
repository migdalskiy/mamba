"""Quality metrics: accuracy, macro-F1, per-class F1, NLL, ECE, Brier (spec §6)."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from mambacls.eval.calibration import brier, ece, fit_temperature


@dataclass
class Predictions:
    logits: np.ndarray  # (N, C)
    labels: np.ndarray  # (N,) or (N, C) multilabel
    example_ids: Optional[np.ndarray] = None
    lengths: Optional[np.ndarray] = None
    pooled: Optional[np.ndarray] = None

    @property
    def correct(self) -> np.ndarray:
        return (self.logits.argmax(1) == self.labels).astype(np.float64)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def f1_per_class(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        f1[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return f1


def compute_metrics(logits: np.ndarray, labels: np.ndarray, n_classes: Optional[int] = None, multilabel: bool = False,
                    temperature: Optional[float] = None, n_bins: int = 15) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n_classes = n_classes or logits.shape[1]
    if multilabel:
        probs = 1 / (1 + np.exp(-logits))
        pred = probs > 0.5
        tp = (pred & (labels > 0.5)).sum(0)
        fp = (pred & (labels < 0.5)).sum(0)
        fn = (~pred & (labels > 0.5)).sum(0)
        f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        return {"acc": float((pred == (labels > 0.5)).all(1).mean()), "macro_f1": float(f1.mean()),
                "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))}
    probs = softmax(logits)
    pred = probs.argmax(1)
    per_class = f1_per_class(pred, labels, n_classes)
    present = np.unique(labels)
    out = {
        "acc": float((pred == labels).mean()),
        "macro_f1": float(per_class[present].mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()),
        "ece": ece(probs, labels, n_bins),
        "brier": brier(probs, labels),
        "n": int(len(labels)),
    }
    out.update({f"f1_class_{c}": float(per_class[c]) for c in range(n_classes)})
    if temperature is not None:
        pt = softmax(logits, temperature)
        out.update({"temperature": float(temperature), "ece_ts": ece(pt, labels, n_bins),
                    "nll_ts": float(-np.log(np.clip(pt[np.arange(len(labels)), labels], 1e-12, None)).mean())})
    return out


@torch.no_grad()
def predict(model, loader, device=None, autocast_dtype=None, keep_pooled: bool = False) -> Predictions:
    model.eval()
    logits, labels, ids, lengths, pooled = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device) if device is not None else batch
        ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None and batch.input_ids.is_cuda else _null()
        with ctx:
            out = model(batch)
        logits.append(out.logits.float().cpu())
        if batch.labels is not None:
            labels.append(batch.labels.cpu())
        if batch.example_ids is not None:
            ids.append(batch.example_ids.cpu())
        lengths.append(batch.lengths.cpu())
        if keep_pooled and out.pooled is not None:
            pooled.append(out.pooled.float().cpu())
    cat = lambda xs: torch.cat(xs).numpy() if xs else None  # noqa: E731
    return Predictions(cat(logits), cat(labels), cat(ids), cat(lengths), cat(pooled))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def evaluate(model, loader, n_classes, device=None, multilabel=False, temperature=None, autocast_dtype=None) -> Dict[str, float]:
    """acc, macro_f1, ece, brier, nll (+ per-class F1)."""
    p = predict(model, loader, device=device, autocast_dtype=autocast_dtype)
    return compute_metrics(p.logits, p.labels, n_classes, multilabel=multilabel, temperature=temperature)


def evaluate_with_temperature(model, val_loader, test_loader, n_classes, device=None):
    """Fit T on validation, report test metrics before and after temperature scaling."""
    pv = predict(model, val_loader, device=device)
    t = fit_temperature(pv.logits, pv.labels)
    pt = predict(model, test_loader, device=device)
    return compute_metrics(pt.logits, pt.labels, n_classes, temperature=t), pt
