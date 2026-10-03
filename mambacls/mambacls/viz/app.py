"""Dash app serving the workbench pages (spec §7). Run:

    python -m mambacls.viz.app --store results/store --artifacts results/artifacts.zarr --port 8050
"""

import argparse
from typing import Optional

from mambacls.store.results import ResultsStore
from mambacls.viz.pages import PAGES


def _options(values):
    return [{"label": "all", "value": "all"}] + [{"label": str(v), "value": v} for v in values]


def build_app(store_dir: str, artifacts_path: Optional[str] = None):
    import dash
    from dash import Input, Output, dcc, html

    store = ResultsStore(store_dir)

    def artifacts():
        if not artifacts_path:
            return None
        try:
            from mambacls.store.artifacts import ArtifactStore

            return ArtifactStore(artifacts_path)
        except Exception:
            return None

    def distinct(table, col):
        df = store.read(table)
        return sorted(df[col].dropna().unique()) if col in df else []

    app = dash.Dash(__name__, title="mambacls workbench", suppress_callback_exceptions=True)
    label = {"fontSize": 12, "color": "#52514e", "marginBottom": 2}
    filters = html.Div([
        html.Div([html.Div("dataset", style=label), dcc.Dropdown(id="f-dataset", options=_options(distinct("metrics", "dataset")), value="all", clearable=False)], style={"flex": 1}),
        html.Div([html.Div("backbone", style=label), dcc.Dropdown(id="f-backbone", options=_options(distinct("metrics", "backbone")), value="all", clearable=False)], style={"flex": 1}),
        html.Div([html.Div("run", style=label), dcc.Dropdown(id="f-run", options=_options(distinct("metrics", "run_id")), value="all", clearable=False)], style={"flex": 2}),
        html.Div([html.Div("layer", style=label), dcc.Input(id="f-layer", type="number", min=0, placeholder="last", style={"width": "100%"})], style={"flex": 0.5}),
        html.Div([html.Div("head", style=label), dcc.Input(id="f-head", type="number", min=0, value=0, style={"width": "100%"})], style={"flex": 0.5}),
        html.Div([html.Div("reduction", style=label), dcc.Dropdown(id="f-reduction", options=["pca", "umap", "tsne"], value="pca", clearable=False)], style={"flex": 1}),
    ], style={"display": "flex", "gap": 12, "alignItems": "flex-end", "marginBottom": 12})

    app.layout = html.Div([
        html.H2("Mamba-2 / Mamba-3 classifier workbench", style={"fontWeight": 600, "marginBottom": 4}),
        html.Div("Results from the Parquet store; tensors from the Zarr artifacts. Mamba-3 hidden-attention views are experimental.",
                 style={"color": "#52514e", "marginBottom": 12}),
        filters,
        dcc.Tabs(id="tabs", value="Evaluation", children=[dcc.Tab(label=p, value=p) for p in PAGES]),
        dcc.Loading(html.Div(id="page")),
    ], style={"fontFamily": "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif", "maxWidth": 1400,
              "margin": "0 auto", "padding": 16, "background": "#fcfcfb"})

    @app.callback(Output("page", "children"),
                  Input("tabs", "value"), Input("f-dataset", "value"), Input("f-backbone", "value"), Input("f-run", "value"),
                  Input("f-layer", "value"), Input("f-head", "value"), Input("f-reduction", "value"))
    def render(tab, dataset, backbone, run, layer, head, reduction):
        f = {"dataset": dataset, "backbone": backbone, "run_id": run, "layer": layer, "head": head, "reduction": reduction}
        figs = PAGES[tab](store, artifacts(), f)
        return [dcc.Graph(id=f"g-{fid}", figure=fig, style={"marginBottom": 16}) for fid, fig in figs]

    app.render_page = render  # exposed for tests
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results/store")
    ap.add_argument("--artifacts", default="results/artifacts.zarr")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    build_app(a.store, a.artifacts).run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
