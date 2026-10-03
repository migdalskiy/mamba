"""Snapshot tests for every figure factory (spec §9: JSON spec diff on fixture dataframes).

A reduced spec (trace types, names, sizes, titles, axis titles/types, colours) is compared with
tests/snapshots/<name>.json. Regenerate after intentional changes with UPDATE_SNAPSHOTS=1.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mambacls.viz.figures import data, evaluation, landscape, length, pareto, pooling, representations, ssm, tokenization, training
from mambacls.viz.theme import CATEGORICAL, color_map

SNAP = Path(__file__).parent / "snapshots"
rng = np.random.default_rng(0)


def _len(v):
    if v is None:
        return None
    if isinstance(v, dict) and "bdata" in v:  # plotly>=6 typed-array encoding
        import base64

        n = len(base64.b64decode(v["bdata"])) // np.dtype(v["dtype"]).itemsize
        return int(n)
    try:
        return int(np.asarray(v).size)
    except Exception:
        return None


def reduce_spec(fig):
    d = fig.to_plotly_json()
    lay = d["layout"]
    traces = []
    for t in d["data"]:
        marker = t.get("marker") or {}
        line = t.get("line") or {}
        color = marker.get("color") if isinstance(marker.get("color"), str) else line.get("color") if isinstance(line.get("color"), str) else None
        traces.append({"type": t.get("type"), "name": t.get("name"), "n_x": _len(t.get("x")), "n_z": _len(t.get("z")), "color": color})

    def ax(name):
        a = lay.get(name) or {}
        return {"title": (a.get("title") or {}).get("text"), "type": a.get("type")}

    scene = lay.get("scene") or {}
    return {
        "title": (lay.get("title") or {}).get("text"),
        "traces": traces,
        "xaxis": ax("xaxis"), "yaxis": ax("yaxis"),
        "scene": {k: ((scene.get(k) or {}).get("title") or {}).get("text") for k in ("xaxis", "yaxis", "zaxis")},
        "n_frames": len(d.get("frames") or []),
    }


def check(name, fig):
    spec = reduce_spec(fig)
    path = SNAP / f"{name}.json"
    if os.environ.get("UPDATE_SNAPSHOTS") or not path.exists():
        SNAP.mkdir(exist_ok=True)
        path.write_text(json.dumps(spec, indent=1, sort_keys=True))
    assert json.loads(path.read_text()) == json.loads(json.dumps(spec, sort_keys=True)), f"snapshot {name} changed"
    fig.to_json()  # serialisable


# ------------------------------- fixtures ---------------------------------------------------
def lengths_df():
    rows = []
    for ds in ("sst2", "imdb"):
        for tok in ("gpt-neox", "modernbert"):
            for split in ("train", "test"):
                for i in range(40):
                    n = int(np.exp(rng.normal(3 if ds == "sst2" else 5.5, 0.6)))
                    rows.append({"dataset": ds, "tokenizer": tok, "split": split, "example_id": i, "label": i % 2,
                                 "n_tokens": n, "length": min(n, 512), "n_words": max(1, int(n / 1.3)), "truncated": n > 512})
    return pd.DataFrame(rows)


def label_counts_df():
    return pd.DataFrame([{"dataset": "sst2", "split": s, "label": l, "count": c}
                         for s, cs in (("train", (30, 50)), ("val", (3, 5)), ("test", (40, 45))) for l, c in enumerate(cs)])


def history_df():
    rows = []
    for adapter in ("lora", "full"):
        for seed in range(3):
            for step in range(1, 21):
                rows.append({"adapter": adapter, "seed": seed, "step": step, "split": "train", "loss": 1 / step + 0.01 * seed,
                             "lr_backbone": 1e-4, "lr_head": 1e-3, "epoch": 0})
                if step % 5 == 0:
                    rows.append({"adapter": adapter, "seed": seed, "step": step, "split": "val", "macro_f1": 0.5 + 0.02 * step, "epoch": 0})
    return pd.DataFrame(rows)


def grad_df():
    return pd.DataFrame([{"step": s, "group": g, "grad_norm": 0.1 * (1 + i) * s}
                         for s in range(10, 60, 10) for i, g in enumerate(["layer_000", "layer_001", "layer_000/dt_bias", "head"])])


def metrics_df():
    return pd.DataFrame([{"method": m, "dataset": d, "macro_f1": 0.5 + 0.1 * i + 0.05 * j, "acc": 0.6 + 0.1 * i,
                          "tokens_per_s": 1e4 * (i + 1), "peak_mem_mb": 1000.0 * (3 - i), "n_trainable": 10 ** (3 + i)}
                         for i, m in enumerate(["last/lora", "mean/full", "attn/probe"]) for j, d in enumerate(["sst2", "imdb"])])


# ------------------------------- factories --------------------------------------------------
CASES = {
    "data_label_balance": lambda: data.label_balance(label_counts_df(), "sst2"),
    "data_length_hist": lambda: data.length_hist_ecdf(lengths_df(), "imdb"),
    "data_length_violin": lambda: data.length_by_label_violin(lengths_df(), "sst2"),
    "data_length_3d": lambda: data.dataset_length_label_3d(lengths_df()),
    "data_text_3d": lambda: data.text_embedding_3d(pd.DataFrame({"x": [0, 1, 2], "y": [1, 2, 3], "z": [0, 0, 1], "label": [0, 1, 1], "text": ["a", "b", "c"]})),
    "tok_tokens_per_word": lambda: tokenization.tokens_per_word(pd.DataFrame({"dataset": ["sst2", "sst2"], "tokenizer": ["a", "b"], "split": ["train"] * 2, "tokens_per_word": [1.3, 1.2]})),
    "tok_pct_truncated": lambda: tokenization.pct_truncated(pd.DataFrame({"dataset": ["imdb"] * 3, "tokenizer": ["a"] * 3, "max_len": [128, 512, 2048], "pct_truncated": [80, 20, 1]})),
    "tok_surface": lambda: tokenization.tokenizer_length_density_surface(np.arange(1, 200), np.arange(1, 200) * 1.1, bins=10),
    "train_loss": lambda: training.training_curves(history_df(), "loss", "train"),
    "train_val_f1": lambda: training.training_curves(history_df(), "macro_f1", "val"),
    "train_lr": lambda: training.lr_schedule(history_df()),
    "train_grad_lines": lambda: training.grad_norm_lines(grad_df()),
    "train_pie": lambda: training.trainable_params_pie(1000, 100000),
    "train_grad_surface": lambda: training.layer_step_gradnorm_surface(grad_df()),
    "train_grad_surface_dt": lambda: training.layer_step_gradnorm_surface(grad_df(), "dt_bias"),
    "train_seed_ribbons": lambda: training.seed_step_ribbons_3d(history_df()),
    "rep_embedding_2d": lambda: representations.embedding_2d(rng.normal(size=(30, 8)), np.arange(30) % 3),
    "rep_embedding_correct": lambda: representations.embedding_2d(np.eye(6), [0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0], color_by="correct"),
    "rep_layer_probe": lambda: representations.layer_probe_accuracy(pd.DataFrame({"backbone": ["a"] * 3, "layer": [0, 1, 2], "acc": [0.5, 0.7, 0.6]})),
    "rep_layer_slider": lambda: representations.embedding_3d_layer_slider([rng.normal(size=(12, 5)) for _ in range(3)], np.arange(12) % 2),
    "rep_norm_surface": lambda: representations.layer_position_norm_surface([np.ones((5, 4)) * i for i in range(3)]),
    "rep_cka": lambda: representations.cka_surface(representations.cka_matrix([rng.normal(size=(20, 4)) for _ in range(3)])),
    "ssm_dt_heatmap": lambda: ssm.dt_heatmap(np.abs(rng.normal(size=(10, 4))), 1),
    "ssm_decay": lambda: ssm.decay_curves(np.abs(rng.normal(size=(10, 4))), -np.ones(4), 1),
    "ssm_state_norm": lambda: ssm.state_norm_lines(np.abs(rng.normal(size=(10, 4))), 1),
    "ssm_hidden_attention": lambda: ssm.hidden_attention_heatmap(np.tril(np.ones((6, 6))), 1, 0),
    "ssm_hidden_attention_m3": lambda: ssm.hidden_attention_heatmap(np.tril(np.ones((6, 6))), 1, 0, experimental=True),
    "ssm_dt_surface": lambda: ssm.dt_surface(np.ones((6, 3)), 0),
    "ssm_ha_surface": lambda: ssm.hidden_attention_surface(np.tril(np.ones((6, 6))), 0, 0),
    "ssm_trajectory": lambda: ssm.state_trajectory_3d(rng.normal(size=(10, 4, 3)), 0, 1),
    "pool_tokens": lambda: pooling.token_weight_heatmap(["the", "movie", "was", "great"], [0.1, 0.2, 0.1, 0.6]),
    "pool_scalar_mix": lambda: pooling.scalar_mix_bar([0.2, 0.5, 0.3]),
    "pool_surface": lambda: pooling.pool_weight_surface(np.ones((3, 5)) / 5, [5, 3, 4]),
    "eval_confusion": lambda: evaluation.confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1]),
    "eval_reliability": lambda: evaluation.reliability_diagram(pd.DataFrame({"confidence": [0.3, 0.6, 0.9], "accuracy": [0.2, 0.6, 0.95], "count": [10, 10, 10], "conf_lo": [0.2, 0.5, 0.8], "conf_hi": [0.4, 0.7, 1.0]})),
    "eval_per_class": lambda: evaluation.per_class_f1({"f1_class_0": 0.5, "f1_class_1": 0.9, "f1_class_2": 0.7}),
    "eval_error_length": lambda: evaluation.error_vs_length(pd.DataFrame({"length": np.arange(1, 101), "correct": np.arange(100) % 3 > 0, "method": ["a", "b"] * 50}), bins=4),
    "eval_towers": lambda: evaluation.confusion_towers_3d([0, 1, 2, 2, 1], [0, 1, 2, 1, 1]),
    "eval_method_dataset": lambda: evaluation.method_dataset_bars_3d(metrics_df()),
    "land_1d": lambda: landscape.interp_1d(pd.DataFrame({"alpha": [0, 0.5, 1] * 2, "loss": [1, 0.6, 0.3, 1, 0.8, 0.4], "method": ["a"] * 3 + ["b"] * 3})),
    "land_contour": lambda: landscape.contour_2d({"x": np.linspace(-1, 1, 3), "y": np.linspace(-1, 1, 3), "z": np.ones((3, 3)) + 0.1}),
    "land_surface": lambda: landscape.surface_3d({"lora": {"x": np.arange(3), "y": np.arange(3), "z": np.ones((3, 3))}}),
    "len_acc": lambda: length.acc_vs_length(pd.DataFrame({"method": ["a", "a", "b", "b"], "eval_len": [512, 1024] * 2, "acc": [0.8, 0.7, 0.9, 0.85]})),
    "len_eff": lambda: length.efficiency_vs_length(pd.DataFrame({"method": ["a", "a"], "seq_len": [128, 512], "tokens_per_s": [1e4, 3e4], "batch_size": [1, 1], "oom": [False, False]})),
    "len_surface": lambda: length.acc_length_method_surface(pd.DataFrame({"method": ["a", "a", "b", "b"], "eval_len": [512, 1024] * 2, "acc": [0.8, 0.7, 0.9, 0.85]})),
    "len_latency": lambda: length.latency_surface(pd.DataFrame({"batch_size": [1, 1, 32, 32], "seq_len": [128, 512] * 2, "latency_p50_ms": [1.0, 2.0, 5.0, 9.0]})),
    "pareto_2d": lambda: pareto.pareto_2d(metrics_df(), "tokens_per_s", "acc"),
    "pareto_3d": lambda: pareto.pareto_3d(metrics_df()),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_figure_snapshot(name):
    check(name, CASES[name]())


def test_pareto_front_mask():
    df = pd.DataFrame({"acc": [0.9, 0.8, 0.7, 0.95], "tokens_per_s": [10, 20, 5, 1]})
    assert pareto.pareto_front(df, maximize=["acc", "tokens_per_s"]).tolist() == [True, True, False, True]


def test_color_follows_entity_not_rank():
    full = color_map(["a", "b", "c"])
    filtered = color_map(["a", "c"], fixed_order=["a", "b", "c"])
    assert full["c"] == filtered["c"] == CATEGORICAL[2]
    many = color_map([f"m{i}" for i in range(10)])
    assert list(many.values()).count("#8a8984") == 2  # past slot 8 folds to "Other" gray


def test_dash_app_and_pages_render(tmp_path):
    """Build the Dash app on a store populated by a real (tiny) run and render every page."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import keyword_dataset

    from mambacls.data.tokenize import WhitespaceTokenizer
    from mambacls.experiment import run_experiment
    from mambacls.store.results import ResultsStore
    from mambacls.viz.app import build_app
    from mambacls.viz.pages import PAGES

    store = ResultsStore(tmp_path / "store")
    cfg = {"seed": 0, "max_len": 32, "precision": "fp32",
           "backbone": {"kind": "mamba", "impl": "reference", "name": "tiny",
                        "config": dict(d_model=32, n_layer=2, vocab_size=101, fused_add_norm=False,
                                       ssm_cfg=dict(layer="Mamba2", d_state=16, headdim=16, chunk_size=8))},
           "pooler": {"name": "attn"}, "adapter": {"name": "lora", "r": 2}, "data": {"name": "sst2"},
           "train": {"epochs": 1, "eval_every": 1.0, "patience": 0, "grad_norm_every": 1},
           "probe": {"enabled": True, "n_examples": 4, "what": ["resid", "dt", "state", "state_norm", "ssd_matrix"], "at": [1.0]},
           "landscape": {"enabled": True, "steps": 3, "n_batches": 1},
           "efficiency": {"enabled": True, "seq_lens": [16], "batch_sizes": [1], "warmup": 1, "iters": 1},
           "tracking": {"artifacts": str(tmp_path / "a.zarr")}}
    run_experiment(cfg, datasets_override=keyword_dataset(n_train=60, n_test=20), tokenizer=WhitespaceTokenizer(vocab_size=101),
                   store=store, device="cpu")
    app = build_app(str(tmp_path / "store"), str(tmp_path / "a.zarr"))
    for tab in PAGES:
        graphs = app.render_page(tab, "all", "all", "all", None, 0, "pca")
        assert graphs, tab
        for g in graphs:
            g.figure.to_json()
    ids = {g.id for tab in PAGES for g in app.render_page(tab, "all", "all", "all", None, 0, "pca")}
    for expected in ("g-confusion", "g-hidden-attention", "g-dt-heatmap", "g-train-loss", "g-pool-surface", "g-interp-1d", "g-pareto-throughput"):
        assert expected in ids
    from export_static_report import export

    out = export(tmp_path / "store", tmp_path / "a.zarr", tmp_path / "report")
    assert (out / "index.html").exists() and len(list(out.glob("*.html"))) == len(PAGES) + 1
