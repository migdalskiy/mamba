"""Collation (spec §8, collation contract).

* right padding + explicit ``lengths`` (default). In a causal Mamba stack with causal conv1d,
  right-pad tokens cannot influence real positions, so ``last`` (gather at len - 1) and masked
  mean / max pooling are exact.
* varlen packing (throughput mode): sequences are concatenated into a (1, T) row up to a token
  budget. ``seq_idx`` must cover every position (mamba issue #783), so trailing padding is
  assigned to an extra dummy segment, which also appears in ``cu_seqlens``; ``n_seqs`` says how
  many leading segments are real.
* left padding: only for Hugging Face backends (HF Mamba2 is "mostly tested with left-padding").
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, L) or (1, T) when packed
    lengths: torch.Tensor  # (B,) real lengths (number of sequences n when packed)
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None  # (B, L) for HF backends
    seq_idx: Optional[torch.Tensor] = None  # (1, T) int32, packed only
    cu_seqlens: Optional[torch.Tensor] = None  # (n_segments + 1,) int32, packed only
    n_seqs: Optional[int] = None  # real sequences in a packed row (the rest is the dummy segment)
    example_ids: Optional[torch.Tensor] = None
    meta: Dict = field(default_factory=dict)

    @property
    def packed(self) -> bool:
        return self.cu_seqlens is not None

    def to(self, device):
        kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}
        return replace(self, **kw)


def _labels(features, multilabel):
    if "label" not in features[0]:
        return None
    if multilabel:
        return torch.tensor([f["label"] for f in features], dtype=torch.float)
    return torch.tensor([int(f["label"]) for f in features], dtype=torch.long)


def _ids(features):
    if "example_id" in features[0]:
        return torch.tensor([int(f["example_id"]) for f in features], dtype=torch.long)
    return None


class RightPadCollator:
    def __init__(self, pad_id: int = 0, max_len: Optional[int] = None, multilabel: bool = False, pad_to_multiple: int = 1):
        self.pad_id, self.max_len, self.multilabel, self.multiple = pad_id, max_len, multilabel, pad_to_multiple

    def __call__(self, features: Sequence[dict]) -> Batch:
        seqs = [list(f["input_ids"])[: self.max_len] if self.max_len else list(f["input_ids"]) for f in features]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        L = int(lengths.max())
        L = -(-L // self.multiple) * self.multiple
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask = torch.arange(L)[None] < lengths[:, None]
        return Batch(ids, lengths, _labels(features, self.multilabel), mask.long(), example_ids=_ids(features))


class LeftPadCollator(RightPadCollator):
    def __call__(self, features):
        b = super().__call__(features)
        ids = torch.full_like(b.input_ids, self.pad_id)
        mask = torch.zeros_like(b.input_ids)
        L = ids.shape[1]
        for i, n in enumerate(b.lengths.tolist()):
            ids[i, L - n:] = b.input_ids[i, :n]
            mask[i, L - n:] = 1
        return replace(b, input_ids=ids, attention_mask=mask)


class VarlenPackCollator:
    """Packs a list of sequences into one row of exactly ``tokens_per_batch`` tokens (or the
    total length rounded up to ``pad_to_multiple`` when ``tokens_per_batch`` is None)."""

    def __init__(self, pad_id: int = 0, tokens_per_batch: Optional[int] = None, max_len: Optional[int] = None,
                 multilabel: bool = False, pad_to_multiple: int = 8):
        self.pad_id, self.tokens, self.max_len = pad_id, tokens_per_batch, max_len
        self.multilabel, self.multiple = multilabel, pad_to_multiple

    def __call__(self, features):
        seqs = [list(f["input_ids"])[: self.max_len] if self.max_len else list(f["input_ids"]) for f in features]
        lengths = [len(s) for s in seqs]
        used = sum(lengths)
        total = self.tokens or -(-used // self.multiple) * self.multiple
        if used > total:
            raise ValueError(f"{used} tokens do not fit into tokens_per_batch={total}")
        ids = torch.full((1, total), self.pad_id, dtype=torch.long)
        seq_idx = torch.empty((1, total), dtype=torch.int32)
        cu = [0]
        for i, s in enumerate(seqs):
            ids[0, cu[-1]: cu[-1] + len(s)] = torch.tensor(s, dtype=torch.long)
            seq_idx[0, cu[-1]: cu[-1] + len(s)] = i
            cu.append(cu[-1] + len(s))
        if used < total:  # dummy trailing segment so seq_idx / cu_seqlens cover every position
            seq_idx[0, used:] = len(seqs)
            cu.append(total)
        return Batch(
            ids, torch.tensor(lengths, dtype=torch.long), _labels(features, self.multilabel),
            seq_idx=seq_idx, cu_seqlens=torch.tensor(cu, dtype=torch.int32), n_seqs=len(seqs),
            example_ids=_ids(features),
        )


def unpack_to_padded(packed: torch.Tensor, cu_seqlens: torch.Tensor, pad_value=0, n_seqs: Optional[int] = None):
    """(T, ...) packed -> (n, Lmax, ...) right-padded plus lengths. Includes the dummy segment
    unless ``n_seqs`` is given."""
    bounds = cu_seqlens.tolist()
    n = len(bounds) - 1 if n_seqs is None else n_seqs
    lengths = torch.tensor([bounds[i + 1] - bounds[i] for i in range(n)], device=packed.device)
    L = int(lengths.max()) if n else 0
    out = packed.new_full((n, L, *packed.shape[1:]), pad_value)
    for i in range(n):
        out[i, : lengths[i]] = packed[bounds[i]: bounds[i + 1]]
    return out, lengths


def pack_padded(padded: torch.Tensor, lengths: torch.Tensor, total: int):
    """(n, L, ...) right-padded -> (total, ...) packed (zeros after the last sequence)."""
    out = padded.new_zeros((total, *padded.shape[2:]))
    pos = 0
    for i, n in enumerate(lengths.tolist()):
        out[pos: pos + n] = padded[i, :n]
        pos += n
    return out


def length_bucketed_batches(lengths: Sequence[int], tokens_per_batch: int, shuffle: bool = True, seed: int = 0,
                            max_batch_size: Optional[int] = None) -> List[List[int]]:
    """Group indices of similar length so that batch_size * max_len <= tokens_per_batch."""
    g = torch.Generator().manual_seed(seed)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > tokens_per_batch or (max_batch_size and len(cur) >= max_batch_size)):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        perm = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in perm]
    return batches
