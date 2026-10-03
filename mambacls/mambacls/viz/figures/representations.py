"""Representations stage (spec §7.1)."""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mambacls.viz.theme import STATUS, color_map, empty, style


def reduce(X, n_components: int = 2, method: str = "pca", seed: int = 0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=min(15, len(X) - 1)).fit_transform(X)
        except Exception:
            method = "pca"
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_components, random_state=seed, perplexity=min(30, max(2, len(X) // 4))).fit_transform(X)
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ vt[:n_components].T
    if P.shape[1] < n_components:
        P = np.pad(P, ((0, 0), (0, n_components - P.shape[1])))
    return P


def embedding_2d(X, labels, preds=None, color_by: str = "label", method: str = "pca", title: str = None) -> go.Figure:
    """Pooled embeddings in 2D, coloured by label, prediction or correctness."""
    if len(X) == 0:
        return empty("Pooled embeddings")
    P = reduce(X, 2, method)
    labels = np.asarray(labels)
    if color_by == "correct" and preds is not None:
        groups = np.where(np.asarray(preds) == labels, "correct", "error")
        cmap = {"correct": STATUS["good"], "error": STATUS["critical"]}
    else:
        groups = np.asarray(preds if color_by == "pred" and preds is not None else labels)
        cmap = color_map(np.unique(groups), sorted(np.unique(groups)))
    fig = go.Figure([go.Scatter(x=P[groups == g, 0], y=P[groups == g, 1], mode="markers", name=str(g),
                                marker=dict(color=cmap[g], size=8, opacity=0.8),
                                hovertemplate=f"{color_by} {g}<extra></extra>")
                     for g in (sorted(np.unique(groups)) if color_by != "correct" else ["correct", "error"]) if (groups == g).any()])
    return style(fig, title or f"Pooled embeddings ({method.upper()}), coloured by {color_by}",
                 xaxis_title=f"{method} 1", yaxis_title=f"{method} 2")


def layer_probe_accuracy(df: pd.DataFrame, metric: str = "acc") -> go.Figure:
    """df: backbone, layer, <metric> (linear-probe accuracy per layer)."""
    if df.empty:
        return empty("Linear-probe accuracy per layer")
    cmap = color_map(df.backbone.unique())
    fig = go.Figure([go.Scatter(x=g.layer, y=g[metric], mode="lines+markers", name=b, line_color=cmap[b])
                     for b, g in df.sort_values("layer").groupby("backbone")])
    return style(fig, "Linear-probe accuracy per layer", xaxis_title="layer", yaxis_title=metric)


def embedding_3d_layer_slider(per_layer: Sequence[np.ndarray], labels, method: str = "pca") -> go.Figure:
    """3D projection per layer with a slider across layers (representation evolution)."""
    if not len(per_layer):
        return empty("Representation evolution across layers")
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels))
    cmap = color_map(classes, classes)
    frames = []
    for li, X in enumerate(per_layer):
        P = reduce(X, 3, method)
        frames.append(go.Frame(name=str(li), data=[
            go.Scatter3d(x=P[labels == c, 0], y=P[labels == c, 1], z=P[labels == c, 2], mode="markers", name=str(c),
                         marker=dict(size=3, color=cmap[c])) for c in classes]))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(sliders=[dict(active=0, currentvalue=dict(prefix="layer "), steps=[
        dict(label=f.name, method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
        for f in frames])], scene=dict(xaxis_title="c1", yaxis_title="c2", zaxis_title="c3"))
    return style(fig, f"Representation evolution across layers ({method.upper()})", height=640)


def layer_position_norm_surface(streams: Sequence[np.ndarray], length: int = None) -> go.Figure:
    """streams: per-layer (L, D) residual stream of one example -> surface layer × position × ‖h‖."""
    if not len(streams):
        return empty("layer × position × ‖h‖")
    Z = np.stack([np.linalg.norm(np.asarray(s)[:length], axis=-1) for s in streams])
    fig = go.Figure(go.Surface(z=Z, x=np.arange(Z.shape[1]), y=np.arange(Z.shape[0]), colorbar=dict(title="‖h‖"),
                               hovertemplate="pos %{x}<br>layer %{y}<br>‖h‖ %{z:.2f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="position", yaxis_title="layer", zaxis_title="‖h‖"))
    return style(fig, "Residual-stream norm: layer × position", height=600)


def linear_cka(X, Y) -> float:
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def cka_matrix(per_layer: Sequence[np.ndarray]) -> np.ndarray:
    n = len(per_layer)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = linear_cka(per_layer[i], per_layer[j])
    return M


def cka_surface(M: np.ndarray) -> go.Figure:
    if M.size == 0:
        return empty("Layer × layer CKA")
    idx = np.arange(M.shape[0])
    fig = go.Figure(go.Surface(z=M, x=idx, y=idx, cmin=0, cmax=1, colorbar=dict(title="CKA"),
                               hovertemplate="layer %{x} vs %{y}: CKA %{z:.3f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="layer", yaxis_title="layer", zaxis=dict(title="linear CKA", range=[0, 1])))
    return style(fig, "Layer × layer linear CKA", height=600)
