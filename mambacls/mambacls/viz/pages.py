"""Dashboard pages as pure functions: (ResultsStore, ArtifactStore | None, filters) -> [(id, figure)].

The Dash app, the static report export and the tests all call these."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mambacls.data import stats as data_stats
from mambacls.eval.calibration import reliability_bins_from
from mambacls.viz.figures import (data, evaluation, landscape, length, pareto, pooling, representations, ssm,
                                  tokenization, training)
from mambacls.viz.theme import empty

Figures = List[Tuple[str, object]]


def _filter(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    for k, v in (filters or {}).items():
        if v not in (None, "", "all") and k in df.columns:
            df = df[df[k] == v]
    return df


def _method(df):
    """method = pooler/adapter, filling rows (e.g. test metrics) that lack an explicit method."""
    if {"pooler", "adapter"} <= set(df.columns):
        derived = df.pooler.astype(str) + "/" + df.adapter.astype(str)
        df = df.assign(method=df["method"].fillna(derived) if "method" in df else derived)
    return df


def page_data(store, artifacts, filters) -> Figures:
    lengths = _filter(store.read("lengths"), {k: filters.get(k) for k in ("dataset",)})
    counts = _filter(store.read("data_stats"), {k: filters.get(k) for k in ("dataset",)})
    if lengths.empty:
        return [("data-empty", empty("Data", "No ingest statistics in the store yet"))]
    lengths = lengths.drop_duplicates(subset=[c for c in ("dataset", "tokenizer", "split", "example_id") if c in lengths])
    counts = counts.drop_duplicates(subset=["dataset", "split", "label"])
    ds = filters.get("dataset") or lengths.dataset.iloc[0]
    figs = [("label-balance", data.label_balance(counts, ds)),
            ("length-hist", data.length_hist_ecdf(lengths, ds)),
            ("length-violin", data.length_by_label_violin(lengths, ds)),
            ("dataset-length-3d", data.dataset_length_label_3d(store.read("lengths").drop_duplicates(
                subset=[c for c in ("dataset", "tokenizer", "split", "example_id")])))]
    return figs


def page_tokenization(store, artifacts, filters) -> Figures:
    lengths = store.read("lengths")
    if lengths.empty:
        return [("tok-empty", empty("Tokenization", "No tokenisation statistics yet"))]
    lengths = lengths.drop_duplicates(subset=[c for c in ("dataset", "tokenizer", "split", "example_id")])
    figs = [("tokens-per-word", tokenization.tokens_per_word(data_stats.summary(lengths))),
            ("pct-truncated", tokenization.pct_truncated(data_stats.truncation_table(lengths)))]
    toks = sorted(lengths.tokenizer.unique())
    if len(toks) >= 2:
        a = lengths[lengths.tokenizer == toks[0]].set_index(["dataset", "split", "example_id"]).n_tokens
        b = lengths[lengths.tokenizer == toks[1]].set_index(["dataset", "split", "example_id"]).n_tokens
        j = a.to_frame("a").join(b.to_frame("b"), how="inner")
        figs.append(("tokenizer-surface", tokenization.tokenizer_length_density_surface(j.a, j.b, toks[0], toks[1])))
    return figs


def page_training(store, artifacts, filters) -> Figures:
    hist = _filter(store.read("history"), filters)
    gn = _filter(store.read("grad_norms"), {"run_id": filters.get("run_id")})
    figs = [("train-loss", training.training_curves(hist, "loss", "train")),
            ("val-f1", training.training_curves(hist, "macro_f1", "val")),
            ("lr", training.lr_schedule(hist)),
            ("grad-norm", training.grad_norm_lines(gn)),
            ("gradnorm-surface", training.layer_step_gradnorm_surface(gn)),
            ("gradnorm-surface-dt", training.layer_step_gradnorm_surface(gn, "dt_bias")),
            ("seed-ribbons", training.seed_step_ribbons_3d(hist, "macro_f1"))]
    m = _filter(store.read("metrics"), filters)
    m = m[m.metric_set == "test"] if "metric_set" in m else m
    if not m.empty and "n_trainable" in m:
        r = m.iloc[-1]
        figs.append(("trainable-pie", training.trainable_params_pie(int(r.n_trainable or 0), int(r.n_params or 1),
                                                                     f"Trainable parameters: {r.adapter}")))
    return figs


def _runs_with(artifacts, name):
    out = []
    for run in artifacts.runs() if artifacts is not None else []:
        for step in artifacts.steps(run):
            ent = artifacts.entries(run, step)
            if any(name in v for v in ent.values()):
                out.append((run, step))
    return out


def page_representations(store, artifacts, filters) -> Figures:
    runs = [r for r in _runs_with(artifacts, "test_pooled") if filters.get("run_id") in (None, "", "all", r[0])]
    figs: Figures = []
    if runs:
        run, step = runs[-1]
        X = artifacts.read(run, step, None, "test_pooled")
        y = artifacts.read(run, step, None, "test_labels")
        figs.append(("emb-label", representations.embedding_2d(X, y, color_by="label", method=filters.get("reduction", "pca"))))
        preds = _filter(store.read("predictions"), {"run_id": run})
        if not preds.empty and len(preds) >= len(y):
            figs.append(("emb-correct", representations.embedding_2d(X, y, preds.pred.values[: len(y)], color_by="correct")))
    probe_runs = [r for r in _runs_with(artifacts, "resid") if filters.get("run_id") in (None, "", "all", r[0])]
    if probe_runs:
        run, step = probe_runs[-1]
        ent = artifacts.entries(run, step)
        layers = sorted(int(k.split("_")[1]) for k, v in ent.items() if k.startswith("layer_") and "resid" in v)
        streams = [artifacts.read(run, step, l, "resid") for l in layers]  # (B, L, D)
        lens = artifacts.read(run, step, None, "lengths") if "lengths" in ent.get("global", []) else None
        n0 = int(lens[0]) if lens is not None else None
        figs.append(("layer-pos-norm", representations.layer_position_norm_surface([s[0] for s in streams], n0)))
        pooled = [np.stack([s[b, : int(lens[b]) if lens is not None else None].mean(0) for b in range(s.shape[0])]) for s in streams]
        labels_ = np.zeros(len(pooled[0]))
        figs.append(("layer-slider", representations.embedding_3d_layer_slider(pooled, labels_)))
        figs.append(("cka", representations.cka_surface(representations.cka_matrix(pooled))))
    probe = _filter(store.read("metrics"), {"metric_set": "layer_probe"})
    if not probe.empty:
        figs.append(("layer-probe", representations.layer_probe_accuracy(probe)))
    return figs or [("rep-empty", empty("Representations", "No embeddings or probe captures yet"))]


def page_ssm(store, artifacts, filters) -> Figures:
    runs = [r for r in _runs_with(artifacts, "dt") if filters.get("run_id") in (None, "", "all", r[0])]
    if not runs:
        return [("ssm-empty", empty("SSM internals", "No probe captures yet (enable probe.enabled)"))]
    run, step = runs[-1]
    ent = artifacts.entries(run, step)
    layers = sorted(int(k.split("_")[1]) for k, v in ent.items() if k.startswith("layer_") and "dt" in v)
    layer = int(filters.get("layer") or layers[-1]) if str(filters.get("layer", "")).isdigit() else layers[-1]
    head = int(filters.get("head") or 0)
    ex = 0
    lens = artifacts.read(run, step, None, "lengths") if "lengths" in ent.get("global", []) else None
    n = int(lens[ex]) if lens is not None else None
    names = ent.get(f"layer_{layer:03d}", [])
    attrs = artifacts.attrs(run, step, layer, "dt")
    exp = bool(attrs.get("experimental", False))
    dt = artifacts.read(run, step, layer, "dt")[ex][:n]
    dt2 = dt if dt.ndim == 2 else dt.reshape(dt.shape[0], -1)
    figs = [("dt-heatmap", ssm.dt_heatmap(dt2, layer)), ("dt-surface", ssm.dt_surface(dt2, layer))]
    if "state_norm" in names:
        sn = artifacts.read(run, step, layer, "state_norm")[ex][:n]
        figs.append(("state-norm", ssm.state_norm_lines(sn, layer)))
    if "ssd_matrix" in names:
        M = artifacts.read(run, step, layer, "ssd_matrix")[ex]
        while M.ndim > 3:  # Mamba-3 (h, R, r, l, l): sum over ranks
            M = M.sum(axis=1)
        M = M[min(head, M.shape[0] - 1)][:n, :n]
        figs.append(("hidden-attention", ssm.hidden_attention_heatmap(M, layer, head, exp)))
        figs.append(("hidden-attention-3d", ssm.hidden_attention_surface(M, layer, head, exp)))
    if "state" in names:
        S = artifacts.read(run, step, layer, "state")[ex][:n]
        figs.append(("state-trajectory", ssm.state_trajectory_3d(S[:, min(head, S.shape[1] - 1)], layer, head)))
    return figs


def page_pooling(store, artifacts, filters) -> Figures:
    m = store.read("metrics")
    figs: Figures = []
    if not m.empty and "metric_set" in m:
        mix = _filter(m[m.metric_set == "scalar_mix_weights"], {"dataset": filters.get("dataset")})
        if not mix.empty:
            cols = sorted([c for c in mix if c.startswith("layer_")], key=lambda c: int(c.split("_")[1]))
            figs.append(("scalar-mix", pooling.scalar_mix_bar(mix[cols].dropna(axis=1).iloc[-1].values)))
    weights_runs = _runs_with(artifacts, "pool_weights")
    if weights_runs:
        run, step = weights_runs[-1]
        W = artifacts.read(run, step, None, "pool_weights")
        lens = artifacts.read(run, step, None, "pool_lengths") if "pool_lengths" in artifacts.entries(run, step).get("global", []) else None
        figs.append(("pool-surface", pooling.pool_weight_surface(W, lens)))
        toks = [str(t) for t in range(int(lens[0]) if lens is not None else W.shape[1])]
        figs.append(("pool-text", pooling.token_weight_heatmap(toks, W[0][: len(toks)])))
    return figs or [("pool-empty", empty("Pooling", "No pooling weights captured yet"))]


def page_evaluation(store, artifacts, filters) -> Figures:
    preds = _method(_filter(store.read("predictions"), filters))
    metrics = _method(store.read("metrics"))
    if preds.empty:
        return [("eval-empty", empty("Evaluation", "No predictions in the store yet"))]
    run = filters.get("run_id") if filters.get("run_id") not in (None, "", "all") else preds.run_id.iloc[-1]
    p = preds[preds.run_id == run]
    figs = [("confusion", evaluation.confusion_matrix(p.label.values, p.pred.values))]
    figs.append(("reliability", evaluation.reliability_diagram(reliability_bins_from(p.confidence.values, p.correct.values))))
    m = metrics[(metrics.get("run_id") == run) & (metrics.metric_set == "test")] if "run_id" in metrics else pd.DataFrame()
    if not m.empty:
        figs.append(("per-class-f1", evaluation.per_class_f1(m.iloc[-1].dropna().to_dict())))
    figs.append(("error-length", evaluation.error_vs_length(_filter(preds, {"dataset": p.dataset.iloc[0]}))))
    if p.label.nunique() > 12:
        figs.append(("confusion-towers", evaluation.confusion_towers_3d(p.label.values, p.pred.values)))
    test = metrics[metrics.metric_set == "test"] if "metric_set" in metrics else metrics
    if not test.empty:
        figs.append(("method-dataset-3d", evaluation.method_dataset_bars_3d(test, "macro_f1")))
    return figs


def page_landscape(store, artifacts, filters) -> Figures:
    df = _filter(store.read("landscape"), filters)
    if df.empty:
        return [("landscape-empty", empty("Loss landscape", "Run scripts/landscape.py to populate"))]
    figs = []
    one = df[df.kind == "1d"]
    if not one.empty:
        figs.append(("interp-1d", landscape.interp_1d(one)))
    two = df[df.kind == "2d"]
    surfaces = {}
    for m, g in two.groupby("method"):
        xs, ys = np.sort(g.x.unique()), np.sort(g.y.unique())
        Z = g.pivot_table(index="x", columns="y", values="loss").loc[xs, ys].values
        surfaces[m] = {"x": xs, "y": ys, "z": Z}
    if surfaces:
        first = next(iter(surfaces))
        figs.append(("contour-2d", landscape.contour_2d(surfaces[first], f"Loss landscape: {first}")))
        figs.append(("basins-3d", landscape.surface_3d(surfaces)))
    return figs


def page_length(store, artifacts, filters) -> Figures:
    m = _method(store.read("metrics"))
    eff = _method(store.read("efficiency"))
    figs: Figures = []
    if not m.empty and "eval_len" in m:
        lm = _filter(m[m.eval_len.notna()], {"dataset": filters.get("dataset")})
        if not lm.empty:
            figs += [("acc-length", length.acc_vs_length(lm)), ("acc-length-surface", length.acc_length_method_surface(lm))]
    if not eff.empty:
        figs += [("throughput-length", length.efficiency_vs_length(eff, "tokens_per_s")),
                 ("memory-length", length.efficiency_vs_length(eff, "peak_mem_mb")),
                 ("latency-surface", length.latency_surface(eff))]
    return figs or [("length-empty", empty("Length scaling", "No length sweep or efficiency results yet"))]


def page_pareto(store, artifacts, filters) -> Figures:
    m = _method(store.read("metrics"))
    eff = _method(store.read("efficiency"))
    if m.empty or eff.empty:
        return [("pareto-empty", empty("Pareto", "Need test metrics and efficiency results"))]
    test = m[m.metric_set == "test"].groupby(["method", "backbone", "dataset"])[["acc", "macro_f1"]].mean().reset_index()
    e = eff.groupby(["method", "backbone", "dataset"])[["tokens_per_s", "peak_mem_mb"]].mean().reset_index()
    j = _filter(test.merge(e, on=["method", "backbone", "dataset"]), {"dataset": filters.get("dataset")})
    if j.empty:
        return [("pareto-empty", empty("Pareto", "No runs with both metrics and efficiency"))]
    figs = [("pareto-throughput", pareto.pareto_2d(j, "tokens_per_s", "acc"))]
    if j.peak_mem_mb.notna().any():
        figs += [("pareto-memory", pareto.pareto_2d(j.dropna(subset=["peak_mem_mb"]), "peak_mem_mb", "acc", x_better="min")),
                 ("pareto-3d", pareto.pareto_3d(j.dropna(subset=["peak_mem_mb"])))]
    tm = m[m.metric_set == "test"].dropna(subset=["n_trainable"]) if "n_trainable" in m else pd.DataFrame()
    if not tm.empty:
        figs.append(("params-f1", pareto.pareto_2d(tm.groupby("method")[["n_trainable", "macro_f1"]].mean().reset_index(),
                                                   "n_trainable", "macro_f1", x_better="min")))
    return figs


PAGES: Dict[str, Callable] = {
    "Data": page_data, "Tokenization": page_tokenization, "Training": page_training,
    "Representations": page_representations, "SSM internals": page_ssm, "Pooling": page_pooling,
    "Evaluation": page_evaluation, "Loss landscape": page_landscape, "Length scaling": page_length, "Pareto": page_pareto,
}
