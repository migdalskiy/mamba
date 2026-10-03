"""Phase significance tables from the results store (spec §6, statistics).

    python scripts/significance_report.py --store results/store --phase B --baseline last/full
Writes rows to the ``significance`` table and prints them. Comparisons are made within each
backbone x dataset family, with Holm correction across the family.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mambacls.eval.significance import compare_methods  # noqa: E402
from mambacls.store.results import ResultsStore  # noqa: E402


def report(store: ResultsStore, phase=None, baseline=None, n_boot=10_000):
    preds = store.read("predictions")
    metrics = store.read("metrics")
    metrics = metrics[metrics.metric_set == "test"]
    if phase is not None and "phase" in metrics:
        metrics = metrics[metrics.phase == phase]
        preds = preds[preds.run_id.isin(metrics.run_id)]
    metrics = metrics.assign(method=metrics.pooler.astype(str) + "/" + metrics.adapter.astype(str))
    out = []
    for (bb, ds), p in preds.groupby(["backbone", "dataset"]):
        s = metrics[(metrics.backbone == bb) & (metrics.dataset == ds)][["method", "seed", "macro_f1", "acc"]]
        t = compare_methods(p, s, baseline=baseline, n_boot=n_boot)
        if len(t):
            out.append(t.assign(backbone=bb, dataset=ds))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results/store")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--baseline", default=None)
    a = ap.parse_args()
    store = ResultsStore(a.store)
    t = report(store, a.phase, a.baseline)
    if len(t):
        store.write("significance", t)
    pd.set_option("display.width", 200)
    print(t)


if __name__ == "__main__":
    main()
