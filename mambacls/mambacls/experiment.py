"""One end-to-end run: ingest -> tokenize -> loaders -> model -> train / probe -> test eval ->
efficiency -> results rows (Parquet) + probe tensors (Zarr). Phases A-D are grids of such runs
(see scripts/sweep.py and conf/experiment/)."""

import logging
from functools import partial
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mambacls.data import stats as data_stats
from mambacls.data.collate import RightPadCollator, VarlenPackCollator, length_bucketed_batches
from mambacls.data.ingest import load_splits
from mambacls.data.registry import get_spec
from mambacls.data.tokenize import load_tokenizer, tokenize_dataset
from mambacls.eval.efficiency import benchmark_efficiency
from mambacls.eval.metrics import compute_metrics, predict
from mambacls.eval.calibration import fit_temperature
from mambacls.models.registry import _to_dict, build_classifier
from mambacls.store.results import ResultsStore, config_hash, provenance
from mambacls.train.callbacks import GradNormLogger, ProbeCaptureCallback
from mambacls.train.loop import Trainer, TrainConfig, extract_features, fit_probe_head, set_seed

log = logging.getLogger(__name__)

PARAMETER_FREE_POOLERS = ("last", "mean", "max")


def _loader(ds, cfg, pad_id, train: bool, seed: int, multilabel: bool):
    batching = cfg.get("batching", {})
    mode = batching.get("mode", "right_pad")
    cols = ["input_ids", "label", "example_id"]
    ds = ds.with_format(None, columns=cols)
    if mode == "varlen":
        coll = VarlenPackCollator(pad_id=pad_id, tokens_per_batch=None, multilabel=multilabel)
    else:
        coll = RightPadCollator(pad_id=pad_id, multilabel=multilabel)
    if batching.get("bucket_by_length", True):
        lengths = [len(x) for x in ds["input_ids"]]
        batches = length_bucketed_batches(lengths, batching.get("tokens_per_batch", 32768), shuffle=train, seed=seed,
                                          max_batch_size=batching.get("max_batch_size"))
        return DataLoader(ds, batch_sampler=batches, collate_fn=coll)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batching.get("batch_size", 32), shuffle=train, collate_fn=coll, generator=g)


def prepare_data(cfg: Dict, datasets_override=None, tokenizer=None):
    dcfg = cfg["data"]
    spec = get_spec(dcfg["name"])
    seed = cfg.get("seed", 0)
    dd = datasets_override if datasets_override is not None else load_splits(
        dcfg["name"], seed=dcfg.get("split_seed", 0), revision=dcfg.get("revision"),
        train_fraction=dcfg.get("train_fraction", 1.0), max_examples=dcfg.get("max_examples"))
    tok_name = dcfg.get("tokenizer") or cfg["backbone"].get("tokenizer", "EleutherAI/gpt-neox-20b")
    if spec.byte_level:
        tok_name = "bytes"
    tokenizer = tokenizer or load_tokenizer(tok_name)
    reserve = int(cfg.get("pooler", {}).get("name") == "eos_cls" and cfg.get("pooler", {}).get("learned", True))
    reserve += int(cfg.get("adapter", {}).get("n_tokens", 0)) if cfg.get("adapter", {}).get("name") == "prompt" else 0
    tokenized = {
        k: tokenize_dataset(v, tokenizer, cfg.get("max_len"), dcfg.get("truncation", "head"),
                            append_eos=dcfg.get("append_eos", False), reserve=reserve)
        for k, v in dd.items()
    }
    return spec, tokenized, tokenizer, tok_name


def run_experiment(cfg, datasets_override=None, tokenizer=None, store: Optional[ResultsStore] = None,
                   device=None, artifacts_path: Optional[str] = None) -> Dict:
    cfg = _to_dict(cfg)
    seed = cfg.get("seed", 0)
    set_seed(seed, cfg.get("train", {}).get("deterministic", False))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tracking = cfg.get("tracking", {})
    store = store or ResultsStore(tracking.get("store_dir", "results/store"))
    artifacts_path = artifacts_path or tracking.get("artifacts", "results/artifacts.zarr")
    prov = provenance(cfg, seed)
    run_id = f"{prov['config_hash']}-s{seed}"

    spec, data, tokenizer, tok_name = prepare_data(cfg, datasets_override, tokenizer)
    n_classes = cfg["data"].get("n_classes") or spec.n_classes
    multilabel = spec.multilabel
    lengths = data_stats.length_frame(data, spec.name, tok_name)
    store.write("lengths", lengths, prov, run_id)
    store.write("data_stats", data_stats.label_counts(data, spec.name), prov, run_id)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else pad_id
    loaders = {k: _loader(v, cfg, pad_id, train=(k == "train"), seed=seed, multilabel=multilabel) for k, v in data.items()}
    model_cfg = dict(cfg, data=dict(cfg["data"], n_classes=n_classes, multilabel=multilabel))
    model = build_classifier(model_cfg, n_classes, device=device)
    tcfg = TrainConfig.from_dict({**cfg.get("optim", {}), **cfg.get("train", {}), "seed": seed,
                                  "precision": cfg.get("precision", "bf16_amp"),
                                  "lr_backbone": cfg.get("optim", {}).get("lr_backbone", 2e-5),
                                  "lr_head": cfg.get("optim", {}).get("lr_head", 1e-3),
                                  "weight_decay": cfg.get("optim", {}).get("wd", 0.01),
                                  "no_decay": tuple(cfg.get("optim", {}).get("no_decay", ("A_log", "D", "dt_bias", "norm", "bias"))),
                                  "clip": cfg.get("optim", {}).get("clip", 1.0)})
    base = {"run_id": run_id, "dataset": spec.name, "backbone": cfg["backbone"].get("hf_id") or cfg["backbone"].get("name", "random"),
            "pooler": cfg.get("pooler", {}).get("name"), "adapter": cfg.get("adapter", {}).get("name"),
            "max_len": cfg.get("max_len"), "truncation": cfg["data"].get("truncation", "head"), "seed": seed,
            "phase": cfg.get("phase"), "n_trainable": getattr(model, "n_trainable", lambda: None)(),
            "n_params": sum(p.numel() for p in model.parameters())}

    probe_cfg = cfg.get("probe", {})
    callbacks = [GradNormLogger(every=cfg.get("train", {}).get("grad_norm_every", 10))]
    if probe_cfg.get("enabled") and hasattr(model, "backbone") and hasattr(model.backbone, "model"):
        from mambacls.probe.capture import InternalsRecorder

        probe_feats = [data["val"][i] for i in range(min(probe_cfg.get("n_examples", 64), len(data["val"])))]
        probe_batch = RightPadCollator(pad_id=pad_id)([{k: f[k] for k in ("input_ids", "label", "example_id")} for f in probe_feats])
        layers = probe_cfg.get("layers", "all")

        def sink(rec, step, frac):
            rec.dump(artifacts_path, run_id, step)
            store.write("probe_index", [{"step": step, "fraction": frac, "artifacts": artifacts_path}], prov, run_id)

        callbacks.append(ProbeCaptureCallback(
            lambda m: InternalsRecorder(m, None if layers == "all" else layers, probe_cfg.get("what", ("resid", "dt", "state_norm", "ssd_matrix")),
                                        probe_cfg.get("position_stride", 1)),
            probe_batch, at=probe_cfg.get("at", (0.0, 0.25, 0.5, 1.0)), sink=sink))

    pooler_name = cfg.get("pooler", {}).get("name")
    cached_probe = cfg.get("adapter", {}).get("name") == "probe" and cfg.get("probe_mode", "cached") == "cached" \
        and pooler_name in PARAMETER_FREE_POOLERS + ("scalar_mix",)
    model.to(device)
    if cached_probe:
        per_layer = pooler_name == "scalar_mix"
        xtr, ytr = extract_features(model, loaders["train"], device, per_layer)
        xva, yva = extract_features(model, loaders["val"], device, per_layer)
        fit = fit_probe_head(xtr, ytr, n_classes, xva, yva, seed=seed, scalar_mix=per_layer,
                             epochs=cfg.get("train", {}).get("probe_epochs", 200), lr=cfg.get("optim", {}).get("lr_head", 1e-2))
        test_split = "test" if "test" in loaders else "val"
        xte, yte, ids_te = extract_features(model, loaders[test_split], device, per_layer, return_ids=True)
        with torch.no_grad():
            test_logits = fit["predict"](xte).numpy()
        temperature = fit_temperature(fit["val_logits"], yva.numpy())
        test_labels, test_ids = yte.numpy(), ids_te.numpy()
        train_info = {"train_time_s": None, "train_tokens_per_s": None, "best_step": None}
        if fit["mix"] is not None:
            store.write("metrics", [{**base, "metric_set": "scalar_mix_weights", **{f"layer_{i}": float(w) for i, w in enumerate(fit["mix"])}}], prov)
            layer_rows = []
            for li in range(xtr.shape[1]):  # linear-probe accuracy of every single layer
                fl = fit_probe_head(xtr[:, li], ytr, n_classes, xva[:, li], yva, seed=seed,
                                    epochs=cfg.get("train", {}).get("probe_epochs", 200))
                layer_rows.append({**base, "metric_set": "layer_probe", "layer": li, **{k: fl["val_metrics"][k] for k in ("acc", "macro_f1")}})
            store.write("metrics", layer_rows, prov)
        test_lengths = None
    else:
        lscfg = cfg.get("landscape", {})
        init_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad} if lscfg.get("enabled") else None
        trainer = Trainer(model, tcfg, loaders["train"], loaders["val"], n_classes, device, callbacks, multilabel=multilabel)
        result = trainer.fit()
        if init_state is not None:
            _landscape(model, init_state, loaders["val"], lscfg, store, prov, run_id, f"{base['pooler']}/{base['adapter']}", device)
        store.write("history", result.history.assign(**{k: base[k] for k in ("dataset", "backbone", "pooler", "adapter", "seed")}), prov, run_id)
        gn = callbacks[0].rows
        if gn:
            store.write("grad_norms", gn, prov, run_id)
        pv = predict(model, loaders["val"], device)
        temperature = fit_temperature(pv.logits, pv.labels) if not multilabel else None
        test_split = "test" if "test" in loaders else "val"
        pt = predict(model, loaders[test_split], device, keep_pooled=True)
        test_logits, test_labels, test_ids, test_lengths = pt.logits, pt.labels, pt.example_ids, pt.lengths
        train_info = {"train_time_s": result.train_time_s, "train_tokens_per_s": result.train_tokens_per_s,
                      "best_step": result.best_step, "nan_steps": result.nan_steps}
        if pt.pooled is not None and cfg.get("tracking", {}).get("save_embeddings", True):
            from mambacls.store.artifacts import ArtifactStore

            arts = ArtifactStore(artifacts_path)
            n_emb = min(len(pt.pooled), cfg.get("tracking", {}).get("n_embeddings", 2000))
            arts.write(run_id, result.total_steps, None, "test_pooled", pt.pooled[:n_emb])
            arts.write(run_id, result.total_steps, None, "test_labels", pt.labels[:n_emb])

    metrics = compute_metrics(test_logits, test_labels, n_classes, multilabel=multilabel, temperature=temperature)
    row = {**base, "metric_set": "test", "split": test_split, **train_info, **metrics}
    store.write("metrics", [row], prov)
    if not multilabel:
        preds = pd.DataFrame({
            "example_id": test_ids if test_ids is not None else np.arange(len(test_labels)),
            "label": test_labels, "pred": test_logits.argmax(1),
            "confidence": (np.exp(test_logits - test_logits.max(1, keepdims=True)) /
                           np.exp(test_logits - test_logits.max(1, keepdims=True)).sum(1, keepdims=True)).max(1),
            "length": test_lengths if test_lengths is not None else np.nan,
        })
        preds["correct"] = (preds["label"] == preds["pred"]).astype(float)
        for k in ("dataset", "backbone", "pooler", "adapter", "seed"):
            preds[k] = base[k]
        preds["method"] = f"{base['pooler']}/{base['adapter']}"
        store.write("predictions", preds, prov, run_id)

    if hasattr(model, "pooler") and len(data["val"]):
        _capture_pool_weights(model, data["val"], pad_id, device, artifacts_path, run_id)

    lcfg = cfg.get("length_eval", {})
    if lcfg.get("eval_lengths"):
        store.write("metrics", _length_eval(model, cfg, data, tokenizer, spec, n_classes, base, device, lcfg), prov)

    eff = cfg.get("efficiency", {})
    if eff.get("enabled"):
        df = benchmark_efficiency(model, eff.get("seq_lens", [128, 512, 2048]), eff.get("batch_sizes", [1, 32]),
                                  vocab_size=getattr(tokenizer, "vocab_size", 50280), n_classes=n_classes,
                                  warmup=eff.get("warmup", 50), iters=eff.get("iters", 200), device=device)
        for k in ("dataset", "backbone", "pooler", "adapter"):
            df[k] = base[k]
        store.write("efficiency", df, prov, run_id)
    log.info("run %s: %s", run_id, {k: row[k] for k in ("acc", "macro_f1") if k in row})
    return row


@torch.no_grad()
def _capture_pool_weights(model, val_ds, pad_id, device, artifacts_path, run_id, n: int = 16):
    from mambacls.store.artifacts import ArtifactStore

    feats = [{k: val_ds[i][k] for k in ("input_ids", "label", "example_id")} for i in range(min(n, len(val_ds)))]
    batch = RightPadCollator(pad_id=pad_id)(feats).to(device)
    model.eval()
    out = model(batch)
    if out.pool_weights is None:
        return
    arts = ArtifactStore(artifacts_path)
    arts.write(run_id, 0, None, "pool_weights", out.pool_weights)
    arts.write(run_id, 0, None, "pool_lengths", batch.lengths)
    arts.write(run_id, 0, None, "pool_input_ids", batch.input_ids)


def _length_eval(model, cfg, data, tokenizer, spec, n_classes, base, device, lcfg):
    """Phase C: evaluate the trained model at several L_eval by truncation; optionally with the
    LongMamba-style filter calibrated at L_train (Mamba-2 backbones)."""
    from datasets import DatasetDict

    from mambacls.models.adapters.longmamba import LongMambaFilter

    test_split = "test" if "test" in data else "val"
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    raw = data[test_split].remove_columns([c for c in data[test_split].column_names if c not in ("text", "label", "example_id")])
    variants = [("", None)]
    layer_types = getattr(getattr(model, "backbone", None), "layer_types", [])
    if lcfg.get("longmamba") and layer_types and all(t == "Mamba2" for t in layer_types):
        lm = LongMambaFilter(train_len=lcfg.get("train_len", cfg.get("max_len") or 2048),
                             global_threshold=lcfg.get("global_threshold", 1e-2))
        lm.attach(model.backbone)
        calib_raw = data["val"].remove_columns([c for c in data["val"].column_names if c not in ("text", "label", "example_id")])
        calib = tokenize_dataset(calib_raw.select(range(min(64, len(calib_raw)))), tokenizer, lm.train_len,
                                 cfg["data"].get("truncation", "head"))
        lm.enabled = True
        lm.calibrate(lambda b: model(b.to(device)), _loader(calib, cfg, pad_id, False, 0, False))
        variants.append(("+longmamba", lm))
    rows = []
    for L in lcfg["eval_lengths"]:
        tok = tokenize_dataset(raw, tokenizer, L, cfg["data"].get("truncation", "head"))
        loader = _loader(tok, cfg, pad_id, False, 0, spec.multilabel)
        for suffix, lm in variants:
            for v in variants:
                if v[1] is not None:
                    v[1].enabled = v[1] is lm
            p = predict(model, loader, device)
            m = compute_metrics(p.logits, p.labels, n_classes, multilabel=spec.multilabel)
            rows.append({**base, "metric_set": "length", "eval_len": L, "method": f"{base['pooler']}/{base['adapter']}{suffix}",
                         **{k: m[k] for k in ("acc", "macro_f1") if k in m}})
    for _, lm in variants:
        if lm is not None:
            lm.enabled = False
    return rows


def _landscape(model, init_state, val_loader, lscfg, store, prov, run_id, method, device):
    """1D init -> final interpolation and a 2D filter-normalised slice over the trained parameters."""
    from itertools import islice

    from mambacls.eval.landscape import interpolate_1d, surface_2d

    batches = list(islice(iter(val_loader), lscfg.get("n_batches", 4)))
    rows = [{"kind": "1d", "method": method, **r} for r in interpolate_1d(model, init_state, batches, device=device)]
    s = surface_2d(model, batches, span=lscfg.get("span", 1.0), steps=lscfg.get("steps", 11), device=device)
    rows += [{"kind": "2d", "method": method, "x": float(x), "y": float(y), "loss": float(s["z"][i, j])}
             for i, x in enumerate(s["x"]) for j, y in enumerate(s["y"])]
    store.write("landscape", rows, prov, run_id)
