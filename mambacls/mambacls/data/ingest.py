"""Load datasets through HF ``datasets`` at a pinned revision and normalise them to
{train, val, test} splits with columns ``text``, ``label``, ``example_id`` (spec §4, §6 controls)."""

import logging
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from mambacls.data.registry import DatasetSpec, get_spec

log = logging.getLogger(__name__)


def stratified_split(labels: List, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (keep, held_out) with ``frac`` of every class held out (at least one per class
    that has two or more examples)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray([str(l) for l in labels])
    held = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        if k == 0 and len(idx) > 1:
            k = 1
        held.extend(idx[:k].tolist())
    held = np.sort(np.asarray(held, dtype=int))
    keep = np.setdiff1d(np.arange(len(labels)), held)
    return keep, held


def kfold_indices(labels: List, k: int, repeats: int = 1, seed: int = 0) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeated stratified k-fold: yields (repeat, fold, train_idx, test_idx)."""
    labels = np.asarray([str(l) for l in labels])
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        folds = [[] for _ in range(k)]
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                folds[j % k].append(int(i))
        for f in range(k):
            test = np.sort(np.asarray(folds[f], dtype=int))
            train = np.setdiff1d(np.arange(len(labels)), test)
            yield r, f, train, test


def _to_columns(ds, spec: DatasetSpec):
    def fmt(ex):
        text = "\n\n".join(str(ex[f]) for f in spec.text_fields if ex.get(f) is not None)
        label = ex[spec.label_field]
        return {"text": text, "label": int(label) if not spec.multilabel else label}

    keep = ["text", "label"]
    ds = ds.map(fmt)
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def generate_listops(n: int, seed: int = 0, max_depth: int = 6, max_args: int = 5, max_len: int = 2000) -> Dict[str, list]:
    """LRA ListOps-style expressions: [MAX 4 3 [MIN 2 3 ] 1 0 [MEDIAN 1 5 8 9 2 ] ] -> 5."""
    rng = random.Random(seed)
    ops = ["MAX", "MIN", "MED", "SM"]

    def build(depth):
        if depth >= max_depth or (depth > 0 and rng.random() < 0.25):
            v = rng.randint(0, 9)
            return str(v), v
        op = rng.choice(ops)
        args = [build(depth + 1) for _ in range(rng.randint(2, max_args))]
        vals = [a[1] for a in args]
        val = {"MAX": max, "MIN": min, "MED": lambda v: sorted(v)[len(v) // 2], "SM": lambda v: sum(v) % 10}[op](vals)
        return f"[{op} " + " ".join(a[0] for a in args) + " ]", val

    texts, labels = [], []
    while len(texts) < n:
        s, v = build(0)
        if len(s.split()) <= max_len:
            texts.append(s)
            labels.append(v)
    return {"text": texts, "label": labels}


def load_splits(name: str, seed: int = 0, val_frac: float = 0.1, revision: Optional[str] = None,
                cache_dir: Optional[str] = None, train_fraction: float = 1.0, max_examples: Optional[int] = None):
    """Returns a ``datasets.DatasetDict`` with train / val / test (test may be absent for
    CV-only datasets). Recomputed split sizes are logged and stored in ``.info.description``."""
    import datasets

    spec = get_spec(name)
    if spec.hf_id is None:  # synthetic
        data = {
            split: datasets.Dataset.from_dict(generate_listops(n, seed=seed + i))
            for i, (split, n) in enumerate({"train": 96000, "val": 2000, "test": 2000}.items())
        }
        dd = datasets.DatasetDict(data)
    else:
        raw = datasets.load_dataset(spec.hf_id, spec.config, revision=revision or spec.revision, cache_dir=cache_dir)
        splits = {"train": _to_columns(raw[spec.train_split], spec)}
        if spec.val_split and spec.val_split in raw:
            splits["val"] = _to_columns(raw[spec.val_split], spec)
        if spec.test_split and spec.test_split in raw:
            splits["test"] = _to_columns(raw[spec.test_split], spec)
        dd = datasets.DatasetDict(splits)
    return finalize_splits(dd, spec, seed=seed, val_frac=val_frac, train_fraction=train_fraction, max_examples=max_examples)


def finalize_splits(dd, spec: DatasetSpec, seed: int = 0, val_frac: float = 0.1, train_fraction: float = 1.0,
                    max_examples: Optional[int] = None):
    """Carve a stratified validation split if needed, subsample, add example ids, log sizes."""
    import datasets

    dd = datasets.DatasetDict(dict(dd))
    if "val" not in dd:
        keep, held = stratified_split(dd["train"]["label"], val_frac, seed)
        dd["val"] = dd["train"].select(held)
        dd["train"] = dd["train"].select(keep)
    if train_fraction < 1.0:
        keep, _ = stratified_split(dd["train"]["label"], 1.0 - train_fraction, seed + 1)
        dd["train"] = dd["train"].select(keep)
    if max_examples:
        for k in dd:
            dd[k] = dd[k].select(range(min(max_examples, len(dd[k]))))
    for k in dd:
        dd[k] = dd[k].add_column("example_id", list(range(len(dd[k]))))
    sizes = {k: len(v) for k, v in dd.items()}
    log.info("%s: recomputed split sizes %s (published: %s)", spec.name, sizes, spec.published_sizes)
    return dd
