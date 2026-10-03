"""Evaluation stage (spec §7.1)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mambacls.viz.theme import CATEGORICAL, SEQUENTIAL, STATUS, color_map, empty, style


def confusion_matrix(labels, preds, class_names=None, normalize: bool = True, title: str = "Confusion matrix") -> go.Figure:
    labels, preds = np.asarray(labels), np.asarray(preds)
    if labels.size == 0:
        return empty(title)
    n = int(max(labels.max(), preds.max()) + 1) if class_names is None else len(class_names)
    C = np.zeros((n, n))
    np.add.at(C, (labels, preds), 1)
    Z = C / C.sum(1, keepdims=True).clip(min=1) if normalize else C
    names = [str(c) for c in (class_names or range(n))]
    fig = go.Figure(go.Heatmap(z=Z, x=names, y=names, customdata=C, zmin=0, zmax=1 if normalize else None,
                               colorbar=dict(title="row %" if normalize else "count"),
                               texttemplate="%{z:.2f}" if n <= 12 else None,
                               hovertemplate="true %{y} → pred %{x}<br>%{z:.2%} (%{customdata:.0f})<extra></extra>"))
    return style(fig, title + (" (row-normalised)" if normalize else ""), xaxis_title="predicted", yaxis_title="true",
                 xaxis_type="category", yaxis_type="category", yaxis_autorange="reversed", height=max(420, 22 * n + 160))


def reliability_diagram(bins: pd.DataFrame, title: str = "Reliability diagram") -> go.Figure:
    """bins from eval.calibration.reliability_bins; reliability (top) and confidence histogram (bottom)."""
    if bins.empty:
        return empty(title)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#8a8984", width=1),
                             name="perfect calibration", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=bins.confidence, y=bins.accuracy, mode="lines+markers", line_color=CATEGORICAL[0], name="model",
                             hovertemplate="confidence %{x:.3f}<br>accuracy %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=bins.confidence, y=bins["count"], marker_color=SEQUENTIAL[4], name="examples per bin",
                         width=(bins.conf_hi - bins.conf_lo).clip(lower=0.005), showlegend=False,
                         hovertemplate="%{y} examples<extra></extra>"), 2, 1)
    fig.update_yaxes(title_text="accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="examples", row=2, col=1)
    fig.update_xaxes(title_text="confidence (equal-mass bins)", range=[0, 1], row=2, col=1)
    return style(fig, title, height=560)


def per_class_f1(metrics: dict, class_names=None, title: str = "Per-class F1") -> go.Figure:
    keys = sorted([k for k in metrics if k.startswith("f1_class_")], key=lambda k: int(k.rsplit("_", 1)[1]))
    if not keys:
        return empty(title)
    vals = [metrics[k] for k in keys]
    names = class_names or [k.rsplit("_", 1)[1] for k in keys]
    order = np.argsort(vals)
    fig = go.Figure(go.Bar(x=[vals[i] for i in order], y=[str(names[i]) for i in order], orientation="h", marker_color=CATEGORICAL[0],
                           hovertemplate="%{y}: F1 %{x:.3f}<extra></extra>"))
    return style(fig, title, xaxis_title="F1", xaxis_range=[0, 1], yaxis_type="category", yaxis_title="class",
                 height=max(360, 18 * len(keys) + 140))


def error_vs_length(preds: pd.DataFrame, bins: int = 12, title: str = "Error rate vs input length") -> go.Figure:
    """preds: length, correct (one row per test example); binned error rate with counts in hover."""
    df = preds.dropna(subset=["length"])
    if df.empty:
        return empty(title)
    edges = np.unique(np.quantile(df.length, np.linspace(0, 1, bins + 1)))
    df = df.assign(bucket=pd.cut(df.length, edges, include_lowest=True))
    keys = sorted(df.method.unique()) if "method" in df else ["model"]
    cmap = color_map(keys)
    fig = go.Figure()
    for k in keys:
        d = df[df.method == k] if "method" in df else df
        g = d.groupby("bucket", observed=True).agg(err=("correct", lambda c: 1 - c.mean()), n=("correct", "size"),
                                                   mid=("length", "median")).reset_index()
        fig.add_trace(go.Scatter(x=g.mid, y=g.err, mode="lines+markers", name=k, line_color=cmap[k], customdata=g.n,
                                 hovertemplate="median length %{x:.0f}<br>error %{y:.1%} (n=%{customdata})<extra>" + k + "</extra>"))
    return style(fig, title, xaxis_title="tokens", xaxis_type="log", yaxis_title="error rate", yaxis_tickformat=".0%")


def confusion_towers_3d(labels, preds, class_names=None, max_classes: int = 77) -> go.Figure:
    """3D bar 'towers' of the confusion counts (useful for many-class sets like Banking77)."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    if labels.size == 0:
        return empty("Confusion towers")
    n = min(int(max(labels.max(), preds.max()) + 1), max_classes)
    C = np.zeros((n, n))
    np.add.at(C, (labels[labels < n], preds[labels < n].clip(max=n - 1)), 1)
    xs, ys, zs, i_, j_, k_, inten = [], [], [], [], [], [], []
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    for t in range(n):
        for p in range(n):
            h = C[t, p]
            if h <= 0:
                continue
            base = len(xs)
            for dz in (0, h):
                for dx, dy in ((0, 0), (0.8, 0), (0.8, 0.8), (0, 0.8)):
                    xs.append(p + dx); ys.append(t + dy); zs.append(dz); inten.append(np.log1p(h))
            for a, b, c in faces:
                i_.append(base + a); j_.append(base + b); k_.append(base + c)
    fig = go.Figure(go.Mesh3d(x=xs, y=ys, z=zs, i=i_, j=j_, k=k_, intensity=inten, flatshading=True,
                              colorscale=[[0, SEQUENTIAL[3]], [1, SEQUENTIAL[-1]]], colorbar=dict(title="log(1+count)"),
                              hovertemplate="pred %{x:.0f}, true %{y:.0f}: %{z:.0f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="predicted", yaxis_title="true", zaxis_title="count"))
    return style(fig, "Confusion towers", height=640)


def method_dataset_bars_3d(metrics: pd.DataFrame, metric: str = "macro_f1", method_col: str = "method") -> go.Figure:
    """3D bars method × dataset × metric (mean over seeds)."""
    if metrics.empty or metric not in metrics:
        return empty(f"method × dataset × {metric}")
    m = metrics.groupby([method_col, "dataset"])[metric].mean().reset_index()
    methods, datasets = sorted(m[method_col].unique()), sorted(m.dataset.unique())
    cmap = color_map(methods, methods)
    fig = go.Figure()
    for meth in methods:
        d = m[m[method_col] == meth]
        x = [methods.index(meth)] * len(d)
        y = [datasets.index(ds) for ds in d.dataset]
        for xi, yi, zi, ds in zip(x, y, d[metric], d.dataset):
            fig.add_trace(go.Scatter3d(x=[xi, xi], y=[yi, yi], z=[0, zi], mode="lines", line=dict(color=cmap[meth], width=14),
                                       name=meth, legendgroup=meth, showlegend=bool(ds == d.dataset.iloc[0]),
                                       hovertemplate=f"{meth} · {ds}: {zi:.3f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis=dict(title="method", tickvals=list(range(len(methods))), ticktext=methods),
                                 yaxis=dict(title="dataset", tickvals=list(range(len(datasets))), ticktext=datasets),
                                 zaxis=dict(title=metric)))
    return style(fig, f"Method × dataset × {metric}", height=640)
