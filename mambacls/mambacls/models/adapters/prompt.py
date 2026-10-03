"""Prompt (affix) tuning: k learned embeddings prepended to every sequence (spec §3.4; the
trivially portable part of MambaPEFT's Affix-tuning). Prefix positions are removed from the
backbone output, so poolers only see real tokens."""

import torch
import torch.nn as nn

from mambacls.models.adapters.base import Adapter


class PromptAdapter(Adapter):
    name = "prompt"

    def __init__(self, n_tokens: int = 16, init: str = "vocab", seed: int = 0):
        super().__init__()
        self.n_tokens, self.init, self.seed = n_tokens, init, seed
        self.prefix = None

    def attach(self, backbone):
        super().attach(backbone)
        emb = backbone.embedding.weight
        g = torch.Generator().manual_seed(self.seed)
        if self.init == "vocab":  # start from embeddings of random vocabulary tokens
            idx = torch.randint(0, emb.shape[0], (self.n_tokens,), generator=g)
            init = emb.detach()[idx.to(emb.device)].float().clone()
        else:
            init = torch.randn(self.n_tokens, emb.shape[1], generator=g).to(emb.device) * 0.02
        self.prefix = nn.Parameter(init)
        backbone.prefix_embeds = self.prefix
