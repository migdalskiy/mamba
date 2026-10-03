"""SSM internals (spec §7.1). Inputs are arrays captured by ``probe.capture.InternalsRecorder``
for one example and one layer. Mamba-3 matrices are labelled experimental."""

import numpy as np
import plotly.graph_objects as go

from mambacls.viz.theme import DIVERGING_SCALE, color_map, empty, style


def _sym_range(Z):
    m = float(np.nanmax(np.abs(Z))) if np.size(Z) else 1.0
    return -m, m


def dt_heatmap(dt: np.ndarray, layer: int, tokens=None) -> go.Figure:
    """dt (softplus Δ): (L, H) -> heatmap position × head."""
    dt = np.asarray(dt)
    if dt.size == 0:
        return empty("Δ heatmap")
    fig = go.Figure(go.Heatmap(z=dt.T, x=tokens if tokens is not None else np.arange(dt.shape[0]), colorbar=dict(title="Δ"),
                               hovertemplate="pos %{x}<br>head %{y}<br>Δ %{z:.4f}<extra></extra>"))
    return style(fig, f"Layer {layer}: Δ = softplus(dt) by position and head", xaxis_title="position", yaxis_title="head")


def decay_curves(dt: np.ndarray, A: np.ndarray, layer: int, heads=None) -> go.Figure:
    """Per-head per-step decay exp(Δ·A) over position (Mamba-2: A scalar per head)."""
    dt, A = np.asarray(dt), np.asarray(A)
    decay = np.exp(dt * A[None, :])
    heads = list(range(min(8, decay.shape[1]))) if heads is None else heads
    cmap = color_map(heads, heads)
    fig = go.Figure([go.Scatter(y=decay[:, h], mode="lines", name=f"head {h}", line_color=cmap[h]) for h in heads])
    return style(fig, f"Layer {layer}: per-step decay exp(Δ·A)", xaxis_title="position", yaxis_title="exp(Δ·A)",
                 yaxis_range=[0, 1.02])


def state_norm_lines(state_norm: np.ndarray, layer: int, heads=None) -> go.Figure:
    """state_norm: (L, H) ‖h_t‖ per head."""
    s = np.asarray(state_norm)
    heads = list(range(min(8, s.shape[1]))) if heads is None else heads
    cmap = color_map(heads, heads)
    fig = go.Figure([go.Scatter(y=s[:, h], mode="lines", name=f"head {h}", line_color=cmap[h]) for h in heads])
    return style(fig, f"Layer {layer}: state norm ‖h_t‖", xaxis_title="position", yaxis_title="‖h_t‖")


def hidden_attention_heatmap(M: np.ndarray, layer: int, head: int, experimental: bool = False, tokens=None) -> go.Figure:
    """M (L, L): hidden attention / SSD matrix of one head (rows = output position t)."""
    M = np.asarray(M)
    lo, hi = _sym_range(M)
    ticks = tokens if tokens is not None else np.arange(M.shape[0])
    fig = go.Figure(go.Heatmap(z=M, x=ticks, y=ticks, zmin=lo, zmax=hi, colorscale=DIVERGING_SCALE, colorbar=dict(title="M[t,s]"),
                               hovertemplate="t=%{y} ← s=%{x}<br>%{z:.4f}<extra></extra>"))
    title = f"Layer {layer}, head {head}: hidden attention M = L ∘ CBᵀ" + (" (experimental: Mamba-3)" if experimental else "")
    return style(fig, title, xaxis_title="source position s", yaxis_title="output position t", yaxis_autorange="reversed",
                 height=560)


def dt_surface(dt: np.ndarray, layer: int) -> go.Figure:
    dt = np.asarray(dt)
    fig = go.Figure(go.Surface(z=dt.T, x=np.arange(dt.shape[0]), y=np.arange(dt.shape[1]), colorbar=dict(title="Δ"),
                               hovertemplate="pos %{x}<br>head %{y}<br>Δ %{z:.4f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="position", yaxis_title="head", zaxis_title="Δ"))
    return style(fig, f"Layer {layer}: position × head × Δ", height=600)


def hidden_attention_surface(M: np.ndarray, layer: int, head: int, experimental: bool = False) -> go.Figure:
    M = np.asarray(M)
    lo, hi = _sym_range(M)
    idx = np.arange(M.shape[0])
    fig = go.Figure(go.Surface(z=M, x=idx, y=idx, cmin=lo, cmax=hi, colorscale=DIVERGING_SCALE, colorbar=dict(title="M"),
                               hovertemplate="s=%{x} → t=%{y}<br>%{z:.4f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="source s", yaxis_title="output t", zaxis_title="M[t,s]"))
    return style(fig, f"Layer {layer}, head {head}: position × position × M" + (" (experimental)" if experimental else ""), height=600)


def state_trajectory_3d(states: np.ndarray, layer: int, head: int = None) -> go.Figure:
    """states: (L, ...) flattened per position, projected to 3 PCs; colour runs along position."""
    from mambacls.viz.figures.representations import reduce

    S = np.asarray(states).reshape(np.asarray(states).shape[0], -1)
    if len(S) < 2:
        return empty("State trajectory")
    P = reduce(S, 3, "pca")
    pos = np.arange(len(P))
    fig = go.Figure(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines+markers",
                                 line=dict(color=pos, colorscale="Blues", width=4),
                                 marker=dict(size=3, color=pos, colorscale="Blues", colorbar=dict(title="position")),
                                 hovertemplate="position %{marker.color}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    return style(fig, f"Layer {layer}" + (f", head {head}" if head is not None else "") + ": state trajectory (PCA of h_t)", height=600)
