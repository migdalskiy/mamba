"""Ingest / data stage (spec §7.1)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mambacls.viz.theme import SEQUENTIAL, color_map, empty, style


def label_balance(label_counts: pd.DataFrame, dataset: str) -> go.Figure:
    """Grouped bars: examples per label for each split."""
    df = label_counts[label_counts.dataset == dataset]
    if df.empty:
        return empty(f"{dataset}: label balance")
    cmap = color_map(df.split.unique(), [s for s in ("train", "val", "test") if s in set(df.split)])
    fig = go.Figure([
        go.Bar(x=g.label.astype(str), y=g["count"], name=split, marker_color=cmap[split],
               hovertemplate="label %{x}<br>%{y} examples<extra>" + split + "</extra>")
        for split, g in df.groupby("split")
    ])
    return style(fig, f"{dataset}: label balance per split", xaxis_title="label", yaxis_title="examples", barmode="group",
                 xaxis_type="category")


def length_hist_ecdf(lengths: pd.DataFrame, dataset: str, column: str = "n_tokens") -> go.Figure:
    """Token-length histogram (top) and ECDF (bottom), log-x, one series per tokenizer."""
    df = lengths[lengths.dataset == dataset]
    if df.empty:
        return empty(f"{dataset}: token lengths")
    cmap = color_map(df.tokenizer.unique())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("histogram", "ECDF"))
    for tok, g in df.groupby("tokenizer"):
        v = np.sort(g[column].clip(lower=1).values)
        bins = np.logspace(0, np.log10(max(v.max(), 2)), 40)
        counts, edges = np.histogram(v, bins=bins)
        centers = np.sqrt(edges[:-1] * edges[1:])
        fig.add_trace(go.Bar(x=centers, y=counts, width=np.diff(edges), name=tok, marker_color=cmap[tok], opacity=0.75,
                             hovertemplate="~%{x:.0f} tokens<br>%{y} examples<extra>" + tok + "</extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=v, y=np.arange(1, len(v) + 1) / len(v), mode="lines", line_color=cmap[tok],
                                 name=tok, showlegend=False,
                                 hovertemplate="%{x} tokens: %{y:.1%} of examples<extra>" + tok + "</extra>"), 2, 1)
    fig.update_xaxes(type="log", title_text="tokens", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="examples", row=1, col=1)
    fig.update_yaxes(title_text="fraction ≤ length", row=2, col=1, tickformat=".0%")
    return style(fig, f"{dataset}: token-length distribution", barmode="overlay", height=560)


def length_by_label_violin(lengths: pd.DataFrame, dataset: str, column: str = "n_tokens") -> go.Figure:
    df = lengths[(lengths.dataset == dataset) & (lengths.split == "train")]
    if df.empty:
        return empty(f"{dataset}: length by label")
    labels = sorted(df.label.unique())
    cmap = color_map(labels, labels)
    fig = go.Figure([go.Violin(x=[str(l)] * int((df.label == l).sum()), y=df[df.label == l][column], name=str(l),
                               line_color=cmap[l], box_visible=True, meanline_visible=True, points=False)
                     for l in labels])
    return style(fig, f"{dataset}: length by label (train)", xaxis_title="label", yaxis_title="tokens", yaxis_type="log",
                 showlegend=len(labels) <= 8)


def dataset_length_label_3d(lengths: pd.DataFrame, buckets=(16, 64, 256, 1024, 4096, 16384, np.inf)) -> go.Figure:
    """3D bars: dataset x length bucket x count, stacked by label (one cube column per cell)."""
    if lengths.empty:
        return empty("datasets × length bucket × count")
    df = lengths[lengths.split == "train"].copy()
    names = [f"≤{int(b)}" if np.isfinite(b) else f">{int(buckets[-2])}" for b in buckets]
    df["bucket"] = pd.cut(df.n_tokens, [0, *buckets], labels=names, include_lowest=True)
    counts = df.groupby(["dataset", "bucket"], observed=False).size().reset_index(name="count")
    datasets = sorted(df.dataset.unique())
    xs, ys, zs, i_, j_, k_, intens = [], [], [], [], [], [], []
    for _, r in counts.iterrows():
        x0, y0, h = datasets.index(r.dataset), names.index(r.bucket), float(r["count"])
        if h <= 0:
            continue
        base = len(xs)
        for dz in (0, h):
            for dx, dy in ((0, 0), (0.7, 0), (0.7, 0.7), (0, 0.7)):
                xs.append(x0 + dx); ys.append(y0 + dy); zs.append(dz); intens.append(h)
        faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                 (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        for a, b, c in faces:
            i_.append(base + a); j_.append(base + b); k_.append(base + c)
    fig = go.Figure(go.Mesh3d(x=xs, y=ys, z=zs, i=i_, j=j_, k=k_, intensity=intens, colorscale=[[0, SEQUENTIAL[3]], [1, SEQUENTIAL[-1]]],
                              flatshading=True, colorbar=dict(title="examples"), hovertemplate="%{z:.0f} examples<extra></extra>"))
    fig.update_layout(scene=dict(xaxis=dict(title="dataset", tickvals=[i + 0.35 for i in range(len(datasets))], ticktext=datasets),
                                 yaxis=dict(title="length bucket", tickvals=[i + 0.35 for i in range(len(names))], ticktext=names),
                                 zaxis=dict(title="examples")))
    return style(fig, "Datasets × length bucket × examples (train)", height=600)


def text_embedding_3d(coords: pd.DataFrame, title: str = "TF-IDF / UMAP of raw texts") -> go.Figure:
    """coords: columns x, y, z, label (and optional text)."""
    if coords.empty:
        return empty(title)
    labels = sorted(coords.label.unique())
    cmap = color_map(labels, labels)
    fig = go.Figure([go.Scatter3d(x=g.x, y=g.y, z=g.z, mode="markers", name=str(l), marker=dict(size=3, color=cmap[l]),
                                  text=g.get("text"), hovertemplate="label " + str(l) + "<br>%{text}<extra></extra>")
                     for l, g in coords.groupby("label")])
    return style(fig, title, height=600)


def tfidf_coords(texts, labels, n_components: int = 3, method: str = "umap", seed: int = 0, max_features: int = 20000) -> pd.DataFrame:
    """TF-IDF -> SVD(50) -> UMAP / PCA to 3D."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=max_features, sublinear_tf=True).fit_transform(texts)
    k = max(2, min(50, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(k, random_state=seed).fit_transform(X)
    from mambacls.viz.figures.representations import reduce

    P = reduce(Z, n_components, method, seed)
    return pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "z": P[:, 2] if n_components > 2 else 0, "label": labels,
                         "text": [t[:120] for t in texts]})
