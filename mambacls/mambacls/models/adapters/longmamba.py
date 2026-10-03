"""LongMamba-style training-free long-context extension for Mamba-2 (spec §3.6; Ye et al., ICLR 2025,
arXiv:2504.16053). This is a simplified port, not the reference implementation.

Calibration at the training length L_train measures, for every layer and head, the total decay
budget S_h = E[sum_t dt_t |A_h|] over a sequence. A head is "global" if its memory survives the
training length, exp(-S_h) > ``global_threshold``; the other heads are local. At inference on
longer inputs, local heads are unchanged. For global heads, tokens are filtered: only the tokens
with the largest dt (the most important updates) are kept, while their cumulative decay stays
within the calibrated budget S_h. Filtered tokens get dt = 0, so they neither write to the state
nor decay it. Their receptive field then matches what the head saw during training. Token selection
looks at the whole input, which is fine for classification (the full document is available).
"""

from typing import Dict, Optional

import torch

from mambacls.models.adapters.base import Adapter
from mambacls.models.mixers import MixerMods, mixer_kind


class LongMambaFilter(Adapter):
    name = "longmamba"

    def __init__(self, train_len: int, global_threshold: float = 1e-2, min_keep: int = 1):
        super().__init__()
        self.train_len, self.global_threshold, self.min_keep = train_len, global_threshold, min_keep
        self.register_buffer("budget", torch.empty(0))  # (n_layers, nheads)
        self.register_buffer("is_global", torch.empty(0, dtype=torch.bool))
        self._calibrating: Optional[Dict[int, list]] = None
        self.enabled = True

    def attach(self, backbone):
        super().attach(backbone)
        if any(k in ("Mamba1", "Mamba3") for k in backbone.layer_types):
            raise NotImplementedError("the LongMamba port supports Mamba2 layers (Mamba-1/3 ports are v2 work)")
        backbone.mods_providers.append(self.mods)

    # -- calibration -------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, run_batch, batches):
        """``run_batch(batch)`` runs the model on calibration batches truncated to L_train."""
        self._calibrating = {}
        for b in batches:
            run_batch(b)
        stats = self._calibrating
        self._calibrating = None
        n_layers = self.backbone.n_layers
        nheads = max(v[0].shape[-1] for v in stats.values())
        budget = torch.zeros(n_layers, nheads)
        for i, vals in stats.items():
            budget[i] = torch.cat(vals).mean(0)
        self.budget = budget.to(self.backbone.embedding.weight.device)
        self.is_global = torch.exp(-self.budget) > self.global_threshold
        return self.budget, self.is_global

    # -- filter ------------------------------------------------------------------------
    def mods(self, i):
        if mixer_kind(self.backbone.model.layers[i].mixer) != "Mamba2" or not self.enabled:
            return None

        def dt_filter(dt, A):  # dt: (b, l, h); A: (h,)
            decay = dt * A.abs()
            if self._calibrating is not None:
                self._calibrating.setdefault(i, []).append(decay[:, : self.train_len].sum(1).float().cpu())
                return torch.ones_like(dt)
            if self.budget.numel() == 0 or dt.shape[1] <= self.train_len:
                return torch.ones_like(dt)
            keep = torch.ones_like(dt, dtype=torch.bool)
            glob = self.is_global[i]
            if glob.any():
                d = decay[..., glob]  # (b, l, g)
                order = d.argsort(dim=1, descending=True)
                csum = torch.gather(d, 1, order).cumsum(1)
                ok_sorted = csum <= self.budget[i, glob]
                ok_sorted[:, : self.min_keep] = True
                ok = torch.zeros_like(ok_sorted).scatter(1, order, ok_sorted)
                keep[..., glob] = ok
            return keep.to(dt.dtype)

        return MixerMods(dt_filter=dt_filter)
