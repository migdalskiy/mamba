"""Dataset statistics feeding the ingest / tokenisation dashboard pages."""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def split_sizes(dd, dataset: str) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": dataset, "split": k, "n": len(v)} for k, v in dd.items()])


def label_counts(dd, dataset: str) -> pd.DataFrame:
    rows = []
    for split, ds in dd.items():
        labels, counts = np.unique(np.asarray(ds["label"]), return_counts=True)
        rows += [{"dataset": dataset, "split": split, "label": int(l), "count": int(c)} for l, c in zip(labels, counts)]
    return pd.DataFrame(rows)


def length_frame(dd, dataset: str, tokenizer_name: str) -> pd.DataFrame:
    """One row per example: tokens before truncation, kept length, words, label, split."""
    frames = []
    for split, ds in dd.items():
        cols = {c: ds[c] for c in ("n_tokens", "length", "n_words", "truncated", "label", "example_id") if c in ds.column_names}
        df = pd.DataFrame(cols)
        df["split"], df["dataset"], df["tokenizer"] = split, dataset, tokenizer_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def truncation_table(lengths: pd.DataFrame, max_lens: Iterable[int] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)) -> pd.DataFrame:
    """% of examples truncated at each max length, per dataset / tokenizer."""
    rows = []
    for (ds, tok), g in lengths.groupby(["dataset", "tokenizer"]):
        for L in max_lens:
            rows.append({"dataset": ds, "tokenizer": tok, "max_len": L, "pct_truncated": 100.0 * float((g["n_tokens"] > L).mean())})
    return pd.DataFrame(rows)


def summary(lengths: pd.DataFrame) -> pd.DataFrame:
    g = lengths.groupby(["dataset", "tokenizer", "split"])
    out = g["n_tokens"].describe(percentiles=[0.5, 0.9, 0.99])
    out["tokens_per_word"] = g.apply(lambda d: d["n_tokens"].sum() / max(d["n_words"].sum(), 1))
    return out.reset_index()
