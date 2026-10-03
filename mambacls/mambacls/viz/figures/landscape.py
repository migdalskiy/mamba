"""Loss landscape (spec §7.1): 1D interpolation, 2D filter-normalised contour, 3D surface."""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mambacls.viz.theme import color_map, empty, style


def interp_1d(df: pd.DataFrame) -> go.Figure:
    """df: alpha, loss, method."""
    if df.empty:
        return empty("Loss along init → final")
    cmap = color_map(df.method.unique())
    fig = go.Figure([go.Scatter(x=g.alpha, y=g.loss, mode="lines+markers", name=m, line_color=cmap[m])
                     for m, g in df.groupby("method")])
    fig.add_vline(x=0, line_dash="dot", line_color="#8a8984", annotation_text="init")
    fig.add_vline(x=1, line_dash="dot", line_color="#8a8984", annotation_text="final")
    return style(fig, "Loss along the init → final interpolation", xaxis_title="α", yaxis_title="loss")


def contour_2d(surface: Dict, title: str = "Loss landscape (filter-normalised)") -> go.Figure:
    fig = go.Figure(go.Contour(x=surface["x"], y=surface["y"], z=np.log(surface["z"]), colorbar=dict(title="log loss"),
                               contours=dict(coloring="heatmap")))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=10, color="white", line=dict(width=2, color="#0b0b0b")),
                             name="solution"))
    return style(fig, title, xaxis_title="direction 1", yaxis_title="direction 2")


def surface_3d(surfaces: Dict[str, Dict]) -> go.Figure:
    """Compare basins: one surface per method (LoRA vs full FT vs State-offset)."""
    if not surfaces:
        return empty("Loss basins")
    fig = go.Figure()
    for i, (name, s) in enumerate(surfaces.items()):
        fig.add_trace(go.Surface(x=s["x"], y=s["y"], z=np.log(s["z"]), name=name, showscale=i == 0, opacity=0.85,
                                 colorbar=dict(title="log loss"), hovertemplate=name + "<br>log loss %{z:.3f}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title="direction 1", yaxis_title="direction 2", zaxis_title="log loss"))
    return style(fig, "Loss basins: " + " vs ".join(surfaces), height=640)
