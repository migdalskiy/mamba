"""Calibration: ECE with equal-mass bins, Brier score, reliability bins, temperature scaling (spec §6)."""

import numpy as np
import pandas as pd
import torch


def _conf_correct(probs: np.ndarray, labels: np.ndarray):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return conf, correct


def equal_mass_bins(conf: np.ndarray, n_bins: int):
    order = np.argsort(conf, kind="stable")
    return np.array_split(order, n_bins)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with ``n_bins`` equal-mass (quantile) bins."""
    conf, correct = _conf_correct(probs, labels)
    n = len(labels)
    return float(sum(len(b) / n * abs(correct[b].mean() - conf[b].mean()) for b in equal_mass_bins(conf, n_bins) if len(b)))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf, correct = _conf_correct(probs, labels)
    return reliability_bins_from(conf, correct, n_bins)


def reliability_bins_from(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Equal-mass reliability bins from per-example top-class confidence and correctness."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for i, b in enumerate(equal_mass_bins(conf, n_bins)):
        if len(b):
            rows.append({"bin": i, "confidence": conf[b].mean(), "accuracy": correct[b].mean(), "count": len(b),
                         "conf_lo": conf[b].min(), "conf_hi": conf[b].max()})
    return pd.DataFrame(rows)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits, labels, max_iter: int = 200) -> float:
    """Temperature T > 0 minimising validation NLL of softmax(logits / T)."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())
