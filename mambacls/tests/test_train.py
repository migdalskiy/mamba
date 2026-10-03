import math

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import keyword_dataset, make_classifier
from mambacls.data.collate import RightPadCollator
from mambacls.data.tokenize import WhitespaceTokenizer, tokenize_dataset
from mambacls.experiment import run_experiment
from mambacls.store.results import ResultsStore
from mambacls.train.distill import kd_loss
from mambacls.train.loop import Trainer, TrainConfig, extract_features, fit_probe_head
from mambacls.train.optim import no_decay, param_groups


def _loaders(n_train=160, seed=0):
    dd = keyword_dataset(n_train=n_train, seed=seed)
    tok = WhitespaceTokenizer(vocab_size=101)
    cols = ["input_ids", "label", "example_id"]
    out = {}
    for k, ds in dd.items():
        ds = tokenize_dataset(ds, tok, 32).with_format(None, columns=cols)
        g = torch.Generator().manual_seed(seed)
        out[k] = DataLoader(ds, batch_size=16, shuffle=k == "train", collate_fn=RightPadCollator(), generator=g)
    return out


ADAPTERS = [
    {"name": "full"}, {"name": "lora", "r": 4}, {"name": "sdlora", "r": 4, "top_frac": 0.5},
    {"name": "state_offset", "variant": "h"}, {"name": "prompt", "n_tokens": 2},
    {"name": "bidir+lora", "bidir": {"mode": "tied_gate"}, "lora": {"r": 4}},
]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=[a["name"] + "-" + str(a.get("variant", "")) for a in ADAPTERS])
@pytest.mark.parametrize("mixer", ["Mamba2", "Mamba3"])
def test_training_smoke(adapter, mixer):
    """Loss decreases, no NaN, for every adapter (spec §9: training smoke)."""
    loaders = _loaders()
    model = make_classifier(mixer, pooler="mean", adapter=adapter, n_classes=2)
    cfg = TrainConfig(epochs=3, lr_backbone=3e-3, lr_head=3e-3, eval_every=1.0, patience=0, precision="fp32", warmup_ratio=0.1)
    res = Trainer(model, cfg, loaders["train"], loaders["val"], 2).fit()
    loss = res.history[res.history.split == "train"].loss.values
    assert res.nan_steps == 0 and all(math.isfinite(v) for v in loss)
    assert loss[-5:].mean() < loss[:5].mean()
    assert res.best_metrics["macro_f1"] >= 0.0


def test_determinism_same_seed():
    def run():
        loaders = _loaders(seed=0)
        model = make_classifier("Mamba2", pooler="mean", adapter={"name": "lora", "r": 4}, n_classes=2, seed=0)
        cfg = TrainConfig(epochs=1, lr_backbone=3e-3, lr_head=3e-3, eval_every=1.0, patience=0, precision="fp32", seed=0)
        return Trainer(model, cfg, loaders["train"], loaders["val"], 2).fit().best_metrics["acc"]

    assert abs(run() - run()) <= 0.001


def test_full_finetune_learns_keyword_task():
    loaders = _loaders(n_train=240)
    model = make_classifier("Mamba2", pooler="attn", adapter={"name": "full"}, n_classes=2)
    cfg = TrainConfig(epochs=6, lr_backbone=3e-3, lr_head=3e-3, eval_every=1.0, patience=0, precision="fp32")
    res = Trainer(model, cfg, loaders["train"], loaders["val"], 2).fit()
    assert res.best_metrics["acc"] > 0.8


def test_early_stopping_and_best_restore():
    loaders = _loaders()
    model = make_classifier("Mamba2", pooler="mean", adapter={"name": "lora", "r": 2}, n_classes=2)
    cfg = TrainConfig(epochs=20, lr_backbone=0.0, lr_head=0.0, eval_every=1.0, patience=2, precision="fp32")
    res = Trainer(model, cfg, loaders["train"], loaders["val"], 2).fit()
    assert res.history.epoch.max() < 19  # stopped early


def test_param_groups_no_decay():
    assert no_decay("backbone.model.layers.0.mixer.A_log") and no_decay("x.norm.weight") and no_decay("head.net.1.bias")
    assert not no_decay("backbone.model.layers.0.mixer.in_proj.parametrizations.weight.0.A.fwd")
    model = make_classifier("Mamba2", adapter={"name": "full"})
    groups = param_groups(model, 1e-5, 1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    for g in groups:
        for p in g["params"]:
            if names[id(p)].endswith(("A_log", "dt_bias", ".D")):
                assert g["weight_decay"] == 0.0
    assert {g["group"] for g in groups} >= {"head_decay", "backbone_decay", "backbone_no_decay"}


def test_kd_loss_reduces_to_ce_and_matches_teacher():
    s = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 0])
    assert torch.allclose(kd_loss(s, s, y, weight=0.0), torch.nn.functional.cross_entropy(s, y))
    assert kd_loss(s, s, y, weight=1.0).abs() < 1e-6  # KL(p || p) = 0


def test_cached_linear_probe_and_scalar_mix():
    loaders = _loaders()
    model = make_classifier("Mamba2", pooler="scalar_mix", n_classes=2)
    xtr, ytr = extract_features(model, loaders["train"], per_layer=True)
    xva, yva = extract_features(model, loaders["val"], per_layer=True)
    assert xtr.shape[1:] == (2, 32)
    fit = fit_probe_head(xtr, ytr, 2, xva, yva, scalar_mix=True, epochs=50)
    assert fit["mix"].shape == (2,) and abs(float(fit["mix"].sum()) - 1) < 1e-5


@pytest.mark.parametrize("adapter,pooler", [({"name": "probe"}, {"name": "scalar_mix"}), ({"name": "lora", "r": 2}, {"name": "attn"})])
def test_run_experiment_end_to_end(tmp_path, adapter, pooler):
    store = ResultsStore(tmp_path / "store")
    cfg = {"seed": 0, "max_len": 32, "precision": "fp32", "phase": "test",
           "backbone": {"kind": "mamba", "impl": "reference", "name": "tiny",
                        "config": dict(d_model=32, n_layer=2, vocab_size=101, fused_add_norm=False,
                                       ssm_cfg=dict(layer="Mamba2", d_state=16, headdim=16, chunk_size=8))},
           "pooler": pooler, "adapter": adapter, "data": {"name": "sst2", "tokenizer": "whitespace"},
           "batching": {"bucket_by_length": True, "tokens_per_batch": 256},
           "train": {"epochs": 1, "eval_every": 1.0, "patience": 0, "probe_epochs": 20, "grad_norm_every": 1},
           "probe": {"enabled": True, "n_examples": 4, "at": [0.0, 1.0]},
           "length_eval": {"eval_lengths": [8, 32], "longmamba": True, "train_len": 8},
           "landscape": {"enabled": True, "steps": 3, "n_batches": 1},
           "efficiency": {"enabled": True, "seq_lens": [16], "batch_sizes": [1], "warmup": 1, "iters": 1},
           "tracking": {"artifacts": str(tmp_path / "a.zarr")}}
    row = run_experiment(cfg, datasets_override=keyword_dataset(), tokenizer=WhitespaceTokenizer(vocab_size=101),
                         store=store, device="cpu")
    assert 0 <= row["acc"] <= 1 and row["ece"] >= 0 and row["n_trainable"] > 0
    tables = set(store.tables())
    assert {"metrics", "predictions", "lengths", "data_stats", "efficiency"} <= tables
    m = store.read("metrics")
    assert set(m.metric_set) >= {"test", "length"}
    if adapter["name"] == "probe":
        assert "layer_probe" in set(m.metric_set) and "scalar_mix_weights" in set(m.metric_set)
    else:
        assert {"history", "grad_norms", "probe_index", "landscape"} <= tables
        assert any(str(x).endswith("+longmamba") for x in m.method.dropna())
