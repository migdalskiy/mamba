"""Pareto views (spec §7.1): accuracy vs throughput / memory, 2D and 3D."""

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mambacls.viz.theme import color_map, empty, style


def pareto_front(df: pd.DataFrame, maximize: Sequence[str] = (), minimize: Sequence[str] = ()) -> pd.Series:
    """Boolean mask of non-dominated rows."""
    cols = list(maximize) + list(minimize)
    X = np.column_stack([df[c].values * (1 if c in maximize else -1) for c in cols]).astype(float)
    n = len(X)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dominated.any():
            mask[i] = False
    return pd.Series(mask, index=df.index)


def pareto_2d(df: pd.DataFrame, x: str = "tokens_per_s", y: str = "acc", label: str = "method", x_better: str = "max") -> go.Figure:
    if df.empty:
        return empty(f"{y} vs {x}")
    front = pareto_front(df, maximize=[y] + ([x] if x_better == "max" else []), minimize=[x] if x_better == "min" else [])
    cmap = color_map(df[label].unique())
    fig = go.Figure()
    f = df[front].sort_values(x)
    fig.add_trace(go.Scatter(x=f[x], y=f[y], mode="lines", line=dict(color="#8a8984", dash="dot", width=1), name="Pareto front",
                             hoverinfo="skip"))
    for m, g in df.groupby(label):
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="markers", name=m,
                                 marker=dict(color=cmap[m], size=np.where(front[g.index], 13, 9),
                                             line=dict(width=np.where(front[g.index], 2, 1), color="#0b0b0b")),
                                 hovertemplate=f"{m}<br>{x} %{{x:.3g}}<br>{y} %{{y:.3f}}<extra></extra>"))
    return style(fig, f"{y} vs {x} (front highlighted)", xaxis_title=x, yaxis_title=y, xaxis_type="log")


def pareto_3d(df: pd.DataFrame, x: str = "tokens_per_s", y: str = "peak_mem_mb", z: str = "acc", label: str = "method") -> go.Figure:
    """accuracy × throughput × memory; front points joined by a mesh."""
    if df.empty:
        return empty("accuracy × throughput × memory")
    front = pareto_front(df, maximize=[z, x], minimize=[y])
    cmap = color_map(df[label].unique())
    fig = go.Figure()
    f = df[front]
    if len(f) >= 3:
        fig.add_trace(go.Mesh3d(x=np.log10(f[x]), y=np.log10(f[y]), z=f[z], opacity=0.25, color="#86b6ef", name="front",
                                hoverinfo="skip", alphahull=-1))
    for m, g in df.groupby(label):
        fig.add_trace(go.Scatter3d(x=np.log10(g[x]), y=np.log10(g[y]), z=g[z], mode="markers", name=m,
                                   marker=dict(size=np.where(front[g.index], 7, 4), color=cmap[m]),
                                   hovertemplate=f"{m}<br>{z} %{{z:.3f}}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title=f"log10 {x}", yaxis_title=f"log10 {y}", zaxis_title=z))
    return style(fig, "Pareto: accuracy × throughput × memory", height=640)
