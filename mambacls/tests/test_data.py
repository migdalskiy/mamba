import numpy as np
import pytest
import torch

from conftest import random_features
from mambacls.data.collate import (LeftPadCollator, RightPadCollator, VarlenPackCollator, length_bucketed_batches,
                                   pack_padded, unpack_to_padded)
from mambacls.data.ingest import generate_listops, kfold_indices, stratified_split
from mambacls.data.registry import DATASETS
from mambacls.data.tokenize import ByteTokenizer, WhitespaceTokenizer, tokenize_dataset, truncate


def test_stratified_split_keeps_class_balance():
    labels = [0] * 90 + [1] * 10
    keep, held = stratified_split(labels, 0.1, seed=0)
    assert len(set(keep) & set(held)) == 0 and len(keep) + len(held) == 100
    held_labels = np.asarray(labels)[held]
    assert (held_labels == 0).sum() == 9 and (held_labels == 1).sum() == 1
    assert np.array_equal(held, stratified_split(labels, 0.1, seed=0)[1])  # deterministic


def test_kfold_covers_every_example_once_per_repeat():
    labels = [i % 3 for i in range(30)]
    seen = {}
    for r, f, tr, te in kfold_indices(labels, 5, repeats=2):
        assert len(set(tr) & set(te)) == 0
        seen.setdefault(r, []).extend(te.tolist())
    assert all(sorted(v) == list(range(30)) for v in seen.values())


def test_truncation_policies():
    ids = list(range(10))
    assert truncate(ids, 4, "head") == [0, 1, 2, 3]
    assert truncate(ids, 4, "tail") == [6, 7, 8, 9]
    assert truncate(ids, 5, "head_tail") == [0, 1, 7, 8, 9]
    assert truncate(ids, 20, "head_tail") == ids


def test_tokenize_dataset_reserves_space_and_logs_truncation():
    import datasets

    ds = datasets.Dataset.from_dict({"text": ["a b c d e f g h", "x y"], "label": [0, 1]})
    out = tokenize_dataset(ds, WhitespaceTokenizer(), max_len=5, truncation="head", append_eos=True, reserve=1)
    assert out["length"] == [4, 3] and out["truncated"] == [True, False] and out["n_tokens"] == [8, 2]
    assert out["input_ids"][0][-1] == WhitespaceTokenizer.eos_token_id
    assert ByteTokenizer()("hé")["input_ids"] == [104, 195, 169]


def test_right_and_left_padding():
    feats = random_features([3, 5])
    r = RightPadCollator(pad_id=0)(feats)
    l = LeftPadCollator(pad_id=0)(feats)
    assert r.lengths.tolist() == [3, 5] and r.input_ids.shape == (2, 5)
    assert r.attention_mask[0].tolist() == [1, 1, 1, 0, 0] and l.attention_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert l.input_ids[0, 2:].tolist() == feats[0]["input_ids"]


def test_varlen_pack_covers_every_position():
    """Issue #783: seq_idx must cover the full length; trailing pad goes to a dummy segment."""
    feats = random_features([3, 4])
    b = VarlenPackCollator(tokens_per_batch=10)(feats)
    assert b.seq_idx[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert b.cu_seqlens.tolist() == [0, 3, 7, 10] and b.n_seqs == 2
    exact = VarlenPackCollator(tokens_per_batch=7)(feats)
    assert exact.cu_seqlens.tolist() == [0, 3, 7]
    with pytest.raises(ValueError):
        VarlenPackCollator(tokens_per_batch=5)(feats)
    x = torch.arange(10.0)
    padded, lengths = unpack_to_padded(x, b.cu_seqlens, n_seqs=2)
    assert lengths.tolist() == [3, 4] and torch.equal(pack_padded(padded, lengths, 10)[:7], x[:7])


def test_length_bucketing_respects_token_budget():
    lengths = [5, 50, 7, 48, 6, 52, 8]
    batches = length_bucketed_batches(lengths, tokens_per_batch=100, shuffle=True, seed=1)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert max(lengths[i] for i in b) * len(b) <= 100 or len(b) == 1


def test_listops_generator_is_valid():
    d = generate_listops(50, seed=0)
    assert len(d["text"]) == 50 and all(0 <= y <= 9 for y in d["label"])
    assert all(t.count("[") == t.count("]") for t in d["text"] if t.startswith("["))


def test_registry_covers_spec_datasets():
    for name in ["sst2", "trec", "banking77", "ag_news", "imdb", "yelp", "dbpedia", "arxiv_cls", "hyperpartisan", "lra_text", "lra_listops"]:
        assert name in DATASETS
    assert DATASETS["hyperpartisan"].cv_folds == 5 and DATASETS["banking77"].n_classes == 77
