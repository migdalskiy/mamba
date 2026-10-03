"""Length scaling (spec §7.1, Phase C)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mambacls.viz.theme import color_map, empty, style


def acc_vs_length(df: pd.DataFrame, metric: str = "acc", method_col: str = "method") -> go.Figure:
    """df: method, eval_len, <metric> (rows per seed are averaged)."""
    if df.empty:
        return empty(f"{metric} vs evaluation length")
    cmap = color_map(df[method_col].unique())
    fig = go.Figure()
    for m, g in df.groupby(method_col):
        s = g.groupby("eval_len")[metric].agg(["mean", "std"]).reset_index()
        fig.add_trace(go.Scatter(x=s.eval_len, y=s["mean"], error_y=dict(array=s["std"].fillna(0), thickness=1), mode="lines+markers",
                                 name=m, line_color=cmap[m], hovertemplate="L=%{x}: %{y:.3f}<extra>" + m + "</extra>"))
    return style(fig, f"{metric} vs evaluation length", xaxis_title="L_eval (tokens)", xaxis_type="log", yaxis_title=metric)


def efficiency_vs_length(eff: pd.DataFrame, value: str = "tokens_per_s", method_col: str = "method", batch_size: int = None) -> go.Figure:
    """log-log throughput / memory / latency vs L."""
    df = eff[~eff.get("oom", False).astype(bool)] if "oom" in eff else eff
    if batch_size is not None:
        df = df[df.batch_size == batch_size]
    if df.empty or value not in df:
        return empty(f"{value} vs sequence length")
    cmap = color_map(df[method_col].unique())
    fig = go.Figure([go.Scatter(x=g.seq_len, y=g[value], mode="lines+markers", name=m, line_color=cmap[m])
                     for m, g in df.sort_values("seq_len").groupby(method_col)])
    return style(fig, f"{value} vs sequence length" + (f" (batch {batch_size})" if batch_size else ""), xaxis_title="L (tokens)",
                 yaxis_title=value, xaxis_type="log", yaxis_type="log")


def acc_length_method_surface(df: pd.DataFrame, metric: str = "acc", method_col: str = "method") -> go.Figure:
    """The decisive Phase-C view: accuracy × sequence length × method surface."""
    if df.empty:
        return empty(f"{metric} × length × method")
    piv = df.groupby([method_col, "eval_len"])[metric].mean().unstack("eval_len")
    lens = piv.columns.values
    methods = list(piv.index)
    fig = go.Figure(go.Surface(x=np.log2(lens), y=np.arange(len(methods)), z=piv.values, colorbar=dict(title=metric),
                               customdata=np.broadcast_to(lens, piv.shape),
                               hovertemplate="L=%{customdata}<br>%{z:.3f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis=dict(title="log2 L_eval", tickvals=np.log2(lens), ticktext=[str(int(l)) for l in lens]),
                                 yaxis=dict(title="method", tickvals=list(range(len(methods))), ticktext=methods),
                                 zaxis_title=metric))
    return style(fig, f"{metric} × sequence length × method", height=640)


def latency_surface(eff: pd.DataFrame, value: str = "latency_p50_ms") -> go.Figure:
    df = eff[~eff.get("oom", False).astype(bool)] if "oom" in eff else eff
    if df.empty:
        return empty("latency × L × batch")
    piv = df.groupby(["batch_size", "seq_len"])[value].mean().unstack("seq_len")
    fig = go.Figure(go.Surface(x=np.log2(piv.columns.values), y=np.log2(piv.index.values), z=np.log10(piv.values),
                               colorbar=dict(title=f"log10 {value}")))
    fig.update_layout(scene=dict(xaxis_title="log2 L", yaxis_title="log2 batch", zaxis_title=f"log10 {value}"))
    return style(fig, "Latency × sequence length × batch size", height=600)
