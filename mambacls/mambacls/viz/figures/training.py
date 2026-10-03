"""Training stage (spec §7.1)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mambacls.viz.theme import CATEGORICAL, SEQUENTIAL_SCALE, color_map, empty, style


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def training_curves(history: pd.DataFrame, metric: str = "loss", split: str = "train", group: str = "adapter") -> go.Figure:
    """Mean over seeds with a 95% band (normal approx.) per ``group``, vs step."""
    df = history[(history.split == split)] if "split" in history else history
    if df.empty or metric not in df:
        return empty(f"{split} {metric} vs step")
    cmap = color_map(df[group].unique())
    fig = go.Figure()
    for g, d in df.groupby(group):
        s = d.groupby("step")[metric].agg(["mean", "std", "count"]).reset_index()
        half = 1.96 * s["std"].fillna(0) / np.sqrt(s["count"].clip(lower=1))
        c = cmap[g]
        fig.add_trace(go.Scatter(x=np.r_[s.step, s.step[::-1]], y=np.r_[s["mean"] + half, (s["mean"] - half)[::-1]],
                                 fill="toself", fillcolor=_rgba(c, 0.15), line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s.step, y=s["mean"], mode="lines", name=str(g), line_color=c,
                                 hovertemplate="step %{x}<br>" + metric + " %{y:.4f}<extra>" + str(g) + "</extra>"))
    return style(fig, f"{split} {metric} (mean over seeds, 95% band)", xaxis_title="optimizer step", yaxis_title=metric,
                 hovermode="x unified")


def lr_schedule(history: pd.DataFrame) -> go.Figure:
    df = history[history.split == "train"] if "split" in history else history
    if df.empty or "lr_backbone" not in df:
        return empty("Learning-rate schedule")
    d = df.groupby("step")[["lr_backbone", "lr_head"]].mean().reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("backbone / adapter", "head"))
    fig.add_trace(go.Scatter(x=d.step, y=d.lr_backbone, mode="lines", line_color=CATEGORICAL[0], name="backbone"), 1, 1)
    fig.add_trace(go.Scatter(x=d.step, y=d.lr_head, mode="lines", line_color=CATEGORICAL[1], name="head"), 1, 2)
    fig.update_xaxes(title_text="step")
    fig.update_yaxes(title_text="learning rate", exponentformat="e")
    return style(fig, "Learning-rate schedule (warm-up + cosine)")


def grad_norm_lines(grad_norms: pd.DataFrame, groups=None) -> go.Figure:
    if grad_norms.empty:
        return empty("Gradient norm per layer group")
    df = grad_norms[~grad_norms.group.str.contains("/")]
    keep = groups or sorted(df.group.unique())
    df = df[df.group.isin(keep)]
    cmap = color_map(keep, keep)
    fig = go.Figure([go.Scatter(x=g.step, y=g.grad_norm, mode="lines", name=k, line_color=cmap[k])
                     for k, g in df.groupby("group")])
    return style(fig, "Gradient norm per layer group", xaxis_title="step", yaxis_title="‖grad‖", yaxis_type="log")


def trainable_params_pie(n_trainable: int, n_total: int, title: str = "Trainable parameters") -> go.Figure:
    fig = go.Figure(go.Pie(labels=["trainable", "frozen"], values=[n_trainable, max(n_total - n_trainable, 0)], hole=0.6,
                           marker=dict(colors=[CATEGORICAL[0], "#e6e5e0"]), sort=False,
                           hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"{100 * n_trainable / max(n_total, 1):.2f}%<br>trainable", showarrow=False, font_size=16)
    return style(fig, title)


def layer_step_gradnorm_surface(grad_norms: pd.DataFrame, suffix: str = None) -> go.Figure:
    """3D surface layer × step × grad-norm; ``suffix`` = "A_log" / "dt_bias" for the SSM-dynamics groups
    (spikes here flag dt / A instability)."""
    df = grad_norms[grad_norms.group.str.startswith("layer_")]
    df = df[df.group.str.endswith("/" + suffix)] if suffix else df[~df.group.str.contains("/")]
    if df.empty:
        return empty("layer × step × grad-norm")
    df = df.assign(layer=df.group.str.extract(r"layer_(\d+)")[0].astype(int))
    piv = df.pivot_table(index="layer", columns="step", values="grad_norm", aggfunc="mean")
    z = np.log10(piv.values.clip(min=1e-12))
    fig = go.Figure(go.Surface(x=piv.columns.values, y=piv.index.values, z=z, colorbar=dict(title="log10 ‖g‖"),
                               hovertemplate="step %{x}<br>layer %{y}<br>log10‖g‖ %{z:.2f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="step", yaxis_title="layer", zaxis_title="log10 ‖grad‖"))
    return style(fig, "Layer × step × gradient norm" + (f" ({suffix})" if suffix else ""), height=600)


def seed_step_ribbons_3d(history: pd.DataFrame, metric: str = "macro_f1") -> go.Figure:
    """One validation curve per seed, placed along the seed axis."""
    df = history[history.split == "val"] if "split" in history else history
    if df.empty or metric not in df:
        return empty(f"seed × step × val {metric}")
    seeds = sorted(df.seed.unique())
    cmap = color_map(seeds, seeds)
    fig = go.Figure([go.Scatter3d(x=g.step, y=[s] * len(g), z=g[metric], mode="lines+markers", name=f"seed {s}",
                                  line=dict(color=cmap[s], width=4), marker=dict(size=3))
                     for s, g in df.sort_values("step").groupby("seed")])
    fig.update_layout(scene=dict(xaxis_title="step", yaxis_title="seed", zaxis_title=f"val {metric}"))
    return style(fig, f"Seed × step × validation {metric}", height=600)
