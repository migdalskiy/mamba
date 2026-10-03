"""Poolers: (B, L, D) hidden states (or a list of them, one per layer) -> (B, D') vectors (spec §3.1).

All poolers take a (B, L) boolean mask of real tokens (right padding) and must be invariant to
appended padding. ``weights`` (B, L) are returned where the pooler has per-token weights, for
visualisation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoolOutput:
    pooled: torch.Tensor
    weights: Optional[torch.Tensor] = None
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)


def last_index(mask):
    return mask.long().sum(dim=1).clamp_min(1) - 1


def gather_last(h, mask):
    idx = last_index(mask)
    return h[torch.arange(h.shape[0], device=h.device), idx]


class Pooler(nn.Module):
    needs_all_layers = False
    appends_token = False

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model

    @property
    def out_dim(self) -> int:
        return self.d_model

    def forward(self, h: Union[torch.Tensor, List[torch.Tensor]], mask: torch.Tensor) -> PoolOutput:
        raise NotImplementedError


class LastPooler(Pooler):
    """Hidden state at index len - 1 of every sequence."""

    def forward(self, h, mask):
        weights = F.one_hot(last_index(mask), h.shape[1]).to(h.dtype)
        return PoolOutput(gather_last(h, mask), weights)


class EOSClsPooler(LastPooler):
    """Reads the state of a token appended after the last real token.

    learned=True: the appended token is a new trainable embedding (one extra vector), passed to
    the backbone as ``append_embeds``. learned=False: the collator appends the tokenizer's EOS id
    (``append_eos=True`` in the data config), and this pooler is ``last``.
    """

    def __init__(self, d_model, learned: bool = True, init_std: float = 0.02, **kwargs):
        super().__init__(d_model)
        self.learned = learned
        self.appends_token = learned
        self.embedding = nn.Parameter(torch.randn(d_model) * init_std) if learned else None


class MeanPooler(Pooler):
    def forward(self, h, mask):
        w = mask.to(h.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1)
        return PoolOutput(torch.einsum("bl,bld->bd", w, h), w)


class MaxPooler(Pooler):
    def forward(self, h, mask):
        filled = h.masked_fill(~mask[..., None], torch.finfo(h.dtype).min)
        pooled, arg = filled.max(dim=1)
        # weight of a token = fraction of channels whose max it attains
        weights = F.one_hot(arg, h.shape[1]).to(h.dtype).mean(dim=1)
        return PoolOutput(pooled, weights)


class AttnPooler(Pooler):
    """softmax_t(w^T tanh(W h_t)) weighted sum."""

    def __init__(self, d_model, hidden: Optional[int] = None, **kwargs):
        super().__init__(d_model)
        hidden = hidden or d_model
        self.proj = nn.Linear(d_model, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, h, mask):
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1).float()
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).to(h.dtype)
        return PoolOutput(torch.einsum("bl,bld->bd", w, h), w)


class LatentQueryPooler(Pooler):
    """K learned queries cross-attend to all h_t, then MLP and mean (NV-Embed-style)."""

    def __init__(self, d_model, n_queries: int = 16, n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.0, **kwargs):
        super().__init__(d_model)
        while d_model % n_heads:
            n_heads -= 1
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model)
        )

    def forward(self, h, mask):
        q = self.queries[None].expand(h.shape[0], -1, -1).to(h.dtype)
        out, attn = self.attn(q, h, h, key_padding_mask=~mask, need_weights=True, average_attn_weights=True)
        out = out + self.mlp(out)
        return PoolOutput(out.mean(dim=1), attn.mean(dim=1), {"query_attention": attn})


class ScalarMixPooler(Pooler):
    """Softmax-weighted sum over layers of per-layer pooled vectors (ELMo-style).

    Expects the list of per-layer streams (normalised by ``norm_f`` in the backbone, or by a
    per-layer LayerNorm here with ``layer_norm=True``)."""

    needs_all_layers = True

    def __init__(self, d_model, n_layers: int, base: str = "mean", layer_norm: bool = False, **kwargs):
        super().__init__(d_model)
        self.base = POOLERS[base](d_model)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(()))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]) if layer_norm else None

    def forward(self, h, mask):
        if torch.is_tensor(h):
            raise ValueError("scalar_mix needs the list of per-layer hidden states")
        if len(h) != self.layer_logits.numel():
            raise ValueError(f"scalar_mix built for {self.layer_logits.numel()} layers, got {len(h)}")
        pooled = []
        for i, hi in enumerate(h):
            if self.norms is not None:
                hi = self.norms[i](hi)
            pooled.append(self.base(hi, mask).pooled)
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = self.gamma * torch.einsum("k,kbd->bd", w.to(pooled[0].dtype), torch.stack(pooled))
        return PoolOutput(mixed, None, {"layer_weights": w.detach()})


POOLERS = {
    "last": LastPooler,
    "eos_cls": EOSClsPooler,
    "mean": MeanPooler,
    "max": MaxPooler,
    "attn": AttnPooler,
    "latent_query": LatentQueryPooler,
    "scalar_mix": ScalarMixPooler,
}


def build_pooler(name: str, d_model: int, n_layers: int, **kwargs) -> Pooler:
    if name not in POOLERS:
        raise KeyError(f"unknown pooler {name!r}; choose from {sorted(POOLERS)}")
    return POOLERS[name](d_model, n_layers=n_layers, **kwargs)
