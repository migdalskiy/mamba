import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from conftest import make_classifier
from mambacls.eval.calibration import brier, ece, fit_temperature, reliability_bins
from mambacls.eval.efficiency import benchmark_efficiency
from mambacls.eval.metrics import compute_metrics, f1_per_class, softmax
from mambacls.eval.significance import (bootstrap_ci, compare_methods, corrected_resampled_ttest, holm, paired_bootstrap,
                                        wilcoxon_seeds)


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 300)
    logits = rng.normal(size=(300, 5)) + 2 * np.eye(5)[labels] * (rng.random(300) < 0.6)[:, None]
    m = compute_metrics(logits, labels, 5)
    pred = logits.argmax(1)
    assert m["acc"] == pytest.approx(accuracy_score(labels, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(labels, pred, average="macro"))
    assert np.allclose(f1_per_class(pred, labels, 5), f1_score(labels, pred, average=None))


def test_ece_equal_mass():
    # perfectly calibrated: confidence 0.8 with exactly 80% correct in every bin
    n = 1500
    labels = np.zeros(n, dtype=int)
    probs = np.tile([0.8, 0.2], (n, 1))
    labels[np.arange(n) % 5 == 0] = 1
    assert ece(probs, labels, 15) == pytest.approx(0.0, abs=1e-9)
    # always-wrong confident model: ECE = mean confidence
    assert ece(np.tile([0.9, 0.1], (100, 1)), np.ones(100, dtype=int)) == pytest.approx(0.9)
    bins = reliability_bins(probs, labels, 15)
    assert len(bins) == 15 and bins["count"].sum() == n
    assert brier(np.array([[1.0, 0.0]]), np.array([0])) == 0.0


def test_temperature_scaling_reduces_nll():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 2000)
    logits = 5.0 * (np.eye(3)[labels] * (rng.random(2000) < 0.7)[:, None] + rng.normal(size=(2000, 3)) * 0.8)
    t = fit_temperature(logits, labels)
    before = compute_metrics(logits, labels, 3)
    after = compute_metrics(logits, labels, 3, temperature=t)
    assert t > 1.0 and after["nll_ts"] < before["nll"]


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.5]
    adj = holm(p)
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.5*1=0.5
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_paired_bootstrap_detects_difference_and_null():
    rng = np.random.default_rng(0)
    a = (rng.random(2000) < 0.8).astype(float)
    b = (rng.random(2000) < 0.7).astype(float)
    res = paired_bootstrap(a, b, n=2000)
    assert res.diff == pytest.approx(a.mean() - b.mean()) and res.ci_excludes_zero and res.p_value < 0.01
    null = paired_bootstrap(a, a.copy(), n=500)
    assert null.diff == 0 and null.p_value == 1.0
    mean, lo, hi = bootstrap_ci(a, n=2000)
    assert lo < mean < hi


def test_wilcoxon_and_corrected_ttest():
    assert wilcoxon_seeds([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert wilcoxon_seeds([0.9] * 5, [0.8, 0.81, 0.79, 0.82, 0.8]) == pytest.approx(0.0625)  # min p with 5 seeds
    r = corrected_resampled_ttest([0.02, 0.03, 0.01, 0.04, 0.02], n_train=516, n_test=129)
    plain = stats.ttest_1samp([0.02, 0.03, 0.01, 0.04, 0.02], 0).pvalue
    assert r["p"] > plain  # the correction widens the variance


def test_compare_methods_decision_rule():
    rng = np.random.default_rng(1)
    rows, seeds = [], []
    for method, acc in (("good", 0.9), ("bad", 0.6)):
        for s in range(5):
            c = (rng.random(500) < acc).astype(float)
            rows += [{"method": method, "seed": s, "example_id": i, "correct": v} for i, v in enumerate(c)]
            seeds.append({"method": method, "seed": s, "macro_f1": c.mean() + rng.normal(0, 0.005)})
    t = compare_methods(pd.DataFrame(rows), pd.DataFrame(seeds), n_boot=1000)
    assert len(t) == 1 and bool(t.significant.iloc[0])
    assert abs(t.diff_acc_pts.iloc[0]) > 20


def test_benchmark_efficiency_cpu():
    model = make_classifier("Mamba2", pooler="mean")
    df = benchmark_efficiency(model, [8, 16], [1, 2], vocab_size=100, n_classes=3, warmup=1, iters=2)
    assert len(df) == 4 and (df.tokens_per_s > 0).all() and df.latency_p95_ms.ge(df.latency_p50_ms).all()
    tr = benchmark_efficiency(model, [8], [2], vocab_size=100, n_classes=3, warmup=1, iters=2, train=True)
    assert tr["mode"].iloc[0] == "train"
