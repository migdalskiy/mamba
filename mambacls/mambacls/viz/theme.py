"""Shared Plotly theme for all figure factories.

Colours come from a validated reference palette (categorical order checked for colour-vision
deficiency separation). Rules applied everywhere:
* categorical colours follow the *entity* (method, backbone, class), assigned in a fixed slot
  order, never by rank; series past the eighth fold into "Other" (gray);
* sequential = one hue light -> dark (blue); diverging = blue <-> neutral gray <-> red;
* one y-axis per chart (no dual axes); recessive grid; thin marks; hover on every mark.
"""

from typing import Dict, Iterable, List, Sequence

import plotly.graph_objects as go
import plotly.io as pio

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#8a8984"
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL_SCALE = [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]
DIVERGING_SCALE = [[0.0, "#104281"], [0.25, "#5598e7"], [0.5, "#f0efec"], [0.75, "#e66767"], [1.0, "#a52a2a"]]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
STATUS = {"good": "#008300", "warning": "#eda100", "critical": "#c62828"}
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def color_map(entities: Iterable, fixed_order: Sequence = None) -> Dict:
    """Stable entity -> colour. Order is ``fixed_order`` if given, else sorted entity names, so
    filtering never repaints the remaining series. Entities past slot 8 map to gray "Other"."""
    order = list(fixed_order) if fixed_order is not None else sorted({str(e) for e in entities})
    return {e: (CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER) for i, e in enumerate(order)}


def _axis():
    return dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside", tickcolor=GRID,
                title_font=dict(color=TEXT_SECONDARY, size=12), tickfont=dict(color=TEXT_SECONDARY, size=11))


def _scene_axis():
    return dict(backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=GRID, showbackground=True,
                title_font=dict(color=TEXT_SECONDARY, size=11), tickfont=dict(color=TEXT_SECONDARY, size=10))


TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=TEXT, size=12),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        colorscale=dict(sequential=SEQUENTIAL_SCALE, diverging=DIVERGING_SCALE),
        title=dict(font=dict(size=15, color=TEXT), x=0.0, xanchor="left"),
        xaxis=_axis(), yaxis=_axis(),
        scene=dict(xaxis=_scene_axis(), yaxis=_scene_axis(), zaxis=_scene_axis()),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY, size=11), orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=TEXT, family=FONT)),
        hovermode="closest",
        margin=dict(l=60, r=24, t=56, b=60),
        bargap=0.15, bargroupgap=0.05,
    ),
    data=dict(
        scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=1, color=SURFACE)))],
        bar=[go.Bar(marker=dict(line=dict(width=1, color=SURFACE)))],
        heatmap=[go.Heatmap(colorscale=SEQUENTIAL_SCALE)],
        surface=[go.Surface(colorscale=SEQUENTIAL_SCALE)],
    ),
)
pio.templates["mambacls"] = TEMPLATE


def style(fig: go.Figure, title: str = None, **layout) -> go.Figure:
    fig.update_layout(template="mambacls", title=title, **layout)
    return fig


def empty(title: str, message: str = "No data for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=TEXT_SECONDARY, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, title)
