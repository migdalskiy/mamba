"""Tokenisation stage (spec §7.1)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mambacls.viz.theme import color_map, empty, style


def tokens_per_word(summary: pd.DataFrame) -> go.Figure:
    """summary: dataset, tokenizer, split, tokens_per_word (data.stats.summary)."""
    df = summary[summary.split == "train"] if "split" in summary else summary
    if df.empty:
        return empty("Tokens per word")
    cmap = color_map(df.tokenizer.unique())
    fig = go.Figure([go.Bar(x=g.dataset, y=g.tokens_per_word, name=t, marker_color=cmap[t],
                            hovertemplate="%{x}: %{y:.2f} tokens/word<extra>" + t + "</extra>")
                     for t, g in df.groupby("tokenizer")])
    return style(fig, "Tokens per word", yaxis_title="tokens / word", barmode="group")


def pct_truncated(trunc: pd.DataFrame) -> go.Figure:
    """trunc: dataset, tokenizer, max_len, pct_truncated (data.stats.truncation_table)."""
    if trunc.empty:
        return empty("% truncated at each L")
    keys = sorted({f"{d} · {t}" for d, t in zip(trunc.dataset, trunc.tokenizer)})
    cmap = color_map(keys, keys)
    fig = go.Figure()
    for (d, t), g in trunc.groupby(["dataset", "tokenizer"]):
        k = f"{d} · {t}"
        fig.add_trace(go.Scatter(x=g.max_len, y=g.pct_truncated, mode="lines+markers", name=k, line_color=cmap[k],
                                 hovertemplate="L=%{x}: %{y:.1f}% truncated<extra>" + k + "</extra>"))
    return style(fig, "% of examples truncated at max length L", xaxis_title="L (tokens)", xaxis_type="log",
                 yaxis_title="% truncated", yaxis_range=[0, 100])


def tokenizer_length_density_surface(len_a, len_b, name_a="GPT-NeoX", name_b="ModernBERT", bins: int = 30) -> go.Figure:
    """3D surface: length under tokenizer A × length under tokenizer B × example density."""
    a, b = np.log10(np.clip(np.asarray(len_a), 1, None)), np.log10(np.clip(np.asarray(len_b), 1, None))
    if len(a) == 0:
        return empty("Tokenizer length density")
    H, xe, ye = np.histogram2d(a, b, bins=bins)
    xc, yc = 10 ** ((xe[:-1] + xe[1:]) / 2), 10 ** ((ye[:-1] + ye[1:]) / 2)
    fig = go.Figure(go.Surface(x=xc, y=yc, z=H.T, colorbar=dict(title="examples"),
                               hovertemplate=f"{name_a} ≈%{{x:.0f}}<br>{name_b} ≈%{{y:.0f}}<br>%{{z}} examples<extra></extra>"))
    fig.update_layout(scene=dict(xaxis=dict(title=f"{name_a} tokens", type="log"), yaxis=dict(title=f"{name_b} tokens", type="log"),
                                 zaxis=dict(title="examples")))
    return style(fig, f"Length under {name_a} × {name_b}", height=600)
