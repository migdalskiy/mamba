"""Pooling stage (spec §7.1)."""

import html

import numpy as np
import plotly.graph_objects as go

from mambacls.viz.theme import CATEGORICAL, empty, style


def token_weight_heatmap(tokens, weights, title: str = "Attention-pool weights over tokens", per_row: int = 16) -> go.Figure:
    """Text heatmap: tokens laid out in rows, each cell shaded by its pooling weight."""
    w = np.asarray(weights, dtype=float)
    n = len(tokens)
    if n == 0:
        return empty(title)
    rows = int(np.ceil(n / per_row))
    Z = np.full((rows, per_row), np.nan)
    T = np.full((rows, per_row), "", dtype=object)
    for i, (t, v) in enumerate(zip(tokens, w)):
        Z[i // per_row, i % per_row] = v
        T[i // per_row, i % per_row] = html.escape(str(t))
    fig = go.Figure(go.Heatmap(z=Z, text=T, texttemplate="%{text}", textfont=dict(size=11), colorbar=dict(title="weight"),
                               hovertemplate="%{text}: %{z:.4f}<extra></extra>", xgap=2, ygap=2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    return style(fig, title, height=120 + 36 * rows)


def scalar_mix_bar(layer_weights) -> go.Figure:
    w = np.asarray(layer_weights, dtype=float)
    if w.size == 0:
        return empty("Scalar-mix layer weights")
    fig = go.Figure(go.Bar(x=np.arange(len(w)), y=w, marker_color=CATEGORICAL[0],
                           hovertemplate="layer %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1 / len(w), line_dash="dot", line_color="#8a8984", annotation_text="uniform")
    return style(fig, "Scalar-mix layer weights", xaxis_title="layer", yaxis_title="softmax weight")


def pool_weight_surface(weights, lengths=None) -> go.Figure:
    """weights (B, L): example × position × pool weight (padding shown as gaps)."""
    W = np.asarray(weights, dtype=float).copy()
    if W.size == 0:
        return empty("Pool weights across a batch")
    if lengths is not None:
        for b, n in enumerate(lengths):
            W[b, int(n):] = np.nan
    fig = go.Figure(go.Surface(z=W, x=np.arange(W.shape[1]), y=np.arange(W.shape[0]), colorbar=dict(title="weight"),
                               hovertemplate="example %{y}<br>pos %{x}<br>%{z:.4f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="position", yaxis_title="example", zaxis_title="pool weight"))
    return style(fig, "Example × position × pool weight", height=600)
