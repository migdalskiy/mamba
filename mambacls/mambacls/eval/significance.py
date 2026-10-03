"""Statistics (spec §6): bootstrap CIs, paired bootstrap on per-example correctness, Wilcoxon over
seed-level scores, Holm-Bonferroni, and the corrected resampled t-test for repeated k-fold CV.

Decision rule: claim "A > B" only if the Holm-adjusted p < 0.05 **and** the CI excludes 0."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BootstrapResult:
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def bootstrap_ci(values: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile CI. ``values``: per-example scores (pool them across seeds)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0,
                     chunk: int = 1000) -> BootstrapResult:
    """Resample test examples (rows; columns may be seeds) jointly for A and B. Two-sided p value
    = 2 * min(P(diff* <= 0), P(diff* >= 0)) under the bootstrap distribution."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs aligned per-example arrays")
    d = (a - b).reshape(len(a), -1).mean(1)
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        boots.append(d[rng.integers(0, len(d), (m, len(d)))].mean(1))
    boots = np.concatenate(boots)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return BootstrapResult(float(d.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)),
                           float(min(1.0, p)), len(d))


def wilcoxon_seeds(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Wilcoxon signed-rank test over paired seed-level scores. Note: with 5 seeds the smallest
    attainable two-sided p is 0.0625, so this test alone can never reach 0.05."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def corrected_resampled_ttest(diffs: Sequence[float], n_train: int, n_test: int) -> Dict[str, float]:
    """Nadeau & Bengio corrected resampled t-test for repeated k-fold CV score differences."""
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"t": float("inf") if d.mean() else 0.0, "p": 0.0 if d.mean() else 1.0, "mean_diff": float(d.mean())}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"t": float(t), "p": float(p), "mean_diff": float(d.mean())}


def compare_methods(per_example: pd.DataFrame, seed_scores: pd.DataFrame, baseline: Optional[str] = None,
                    metric: str = "macro_f1", n_boot: int = 10_000, alpha: float = 0.05) -> pd.DataFrame:
    """All pairwise (or vs ``baseline``) comparisons within one backbone x dataset family.

    per_example: columns method, seed, example_id, correct.
    seed_scores: columns method, seed, <metric>.
    Returns diff (points), CI, bootstrap p, Wilcoxon p, Holm-adjusted p (over the bootstrap
    p values of the family) and the ``significant`` decision.
    """
    methods = sorted(per_example["method"].unique())
    pairs = [(m, baseline) for m in methods if m != baseline] if baseline else \
        [(a, b) for i, a in enumerate(methods) for b in methods[i + 1:]]
    rows = []
    for a, b in pairs:
        pa = per_example[per_example.method == a].pivot_table(index="example_id", columns="seed", values="correct")
        pb = per_example[per_example.method == b].pivot_table(index="example_id", columns="seed", values="correct")
        common_idx = pa.index.intersection(pb.index)
        common_seeds = pa.columns.intersection(pb.columns)
        res = paired_bootstrap(pa.loc[common_idx, common_seeds].values, pb.loc[common_idx, common_seeds].values, n=n_boot, alpha=alpha)
        sa = seed_scores[seed_scores.method == a].set_index("seed")[metric]
        sb = seed_scores[seed_scores.method == b].set_index("seed")[metric]
        seeds = sa.index.intersection(sb.index)
        p_w = wilcoxon_seeds(sa.loc[seeds], sb.loc[seeds]) if len(seeds) > 1 else float("nan")
        rows.append({"method_a": a, "method_b": b, "diff_acc_pts": 100 * res.diff, "ci_low_pts": 100 * res.ci_low,
                     "ci_high_pts": 100 * res.ci_high, "p_bootstrap": res.p_value, "p_wilcoxon": p_w,
                     f"diff_{metric}_pts": 100 * float(sa.loc[seeds].mean() - sb.loc[seeds].mean()) if len(seeds) else float("nan")})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_bootstrap"].values)
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].fillna(1.0).values)
        out["significant"] = (out["p_holm"] < alpha) & ((out["ci_low_pts"] > 0) | (out["ci_high_pts"] < 0))
    return out
