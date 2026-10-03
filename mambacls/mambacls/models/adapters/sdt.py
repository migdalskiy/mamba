"""SDLoRA: LoRA on projections + Sparse Dimension Tuning of the SSM parameters (spec §3.3;
Galim et al., ICML 2025, arXiv:2410.09016).

Warm-up (``warmup_epochs``): LoRA and all SSM parameters (A_log, dt_bias, D, and for Mamba-3 also
B_bias / C_bias) train. At the end of warm-up, units are ranked per layer by how much their SSM
parameters moved; the top ``top_frac`` stay trainable, the rest are reset to their initial values
and frozen. A unit is a channel for Mamba (S6, A is (d_inner, d_state)) and a head for Mamba-2 /
Mamba-3 (A is a scalar per head). Head-level selection for Mamba-2/3 is this workbench's
adaptation and has not been tested in the paper.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from mambacls.models.adapters.lora import LoRAAdapter
from mambacls.models.mixers import mixer_kind

_SSM_PARAMS = {
    "Mamba1": ("A_log", "D", "dt_proj.bias"),
    "Mamba2": ("A_log", "dt_bias", "D"),
    "Mamba3": ("dt_bias", "D", "B_bias", "C_bias"),
}


def _unit_view(param: torch.Tensor, kind: str, mixer) -> torch.Tensor:
    """Reshape so dim 0 indexes the tuning unit (channel or head)."""
    if kind == "Mamba2" and param.dim() == 1 and param.numel() == mixer.d_ssm and mixer.D_has_hdim:
        return param.view(mixer.nheads, -1)
    return param.reshape(param.shape[0], -1)


class SDLoRAAdapter(LoRAAdapter):
    name = "sdlora"

    def __init__(self, top_frac: float = 0.1, warmup_epochs: int = 1, **lora_kwargs):
        super().__init__(**lora_kwargs)
        self.top_frac, self.warmup_epochs = top_frac, warmup_epochs
        self.selected = False
        self._init: Dict[str, torch.Tensor] = {}
        self._masks: Dict[str, torch.Tensor] = {}
        self._params: Dict[str, nn.Parameter] = {}
        self._kind: Dict[str, tuple] = {}

    def attach(self, backbone):
        super().attach(backbone)
        for i, block in enumerate(backbone.model.layers):
            kind = mixer_kind(block.mixer)
            for pname in _SSM_PARAMS.get(kind, ()):
                p = block.mixer.get_parameter(pname)
                p.data = p.data.float()
                p.requires_grad_(True)
                key = f"{i}.{pname}"
                self._params[key] = p
                self._init[key] = p.detach().clone()
                self._kind[key] = (i, kind)

    def trainable_parameters(self):
        return super().trainable_parameters() + [p for p in self._params.values() if p.requires_grad]

    def on_epoch_end(self, epoch, trainer=None):
        if not self.selected and epoch + 1 >= self.warmup_epochs:
            self.select()

    @torch.no_grad()
    def select(self) -> Dict[int, torch.Tensor]:
        """Rank units by update magnitude, keep the top fraction per layer, reset the rest."""
        layers = self.backbone.model.layers
        by_layer: Dict[int, List[str]] = {}
        for key, (i, _) in self._kind.items():
            by_layer.setdefault(i, []).append(key)
        selection = {}
        for i, keys in by_layer.items():
            mixer = layers[i].mixer
            kind = self._kind[keys[0]][1]
            score = None
            for key in keys:
                delta = _unit_view(self._params[key] - self._init[key], kind, mixer).abs().sum(dim=1)
                score = delta if score is None else score + delta
            k = max(1, int(round(self.top_frac * score.numel())))
            keep = torch.zeros_like(score, dtype=torch.bool)
            keep[score.topk(k).indices] = True
            selection[i] = keep
            for key in keys:
                p = self._params[key]
                mask = keep.view(-1, *([1] * (_unit_view(p, kind, mixer).dim() - 1))).expand_as(_unit_view(p, kind, mixer))
                mask = mask.reshape(p.shape)
                self._masks[key] = mask
                p.data = torch.where(mask, p.data, self._init[key].to(p.dtype))
                p.register_hook(lambda g, m=mask: g * m.to(g.dtype))
        self.selected = True
        self.selection = selection
        return selection

    @torch.no_grad()
    def after_optimizer_step(self):
        # Adam moments from the warm-up would keep moving frozen entries: restore them.
        for key, mask in self._masks.items():
            p = self._params[key]
            p.data = torch.where(mask, p.data, self._init[key].to(p.dtype))
