"""State-offset Tuning (spec §3.4; arXiv:2503.03499) and initial-state tuning.

variant "y":  y_t += y'          (learned offset of the pre-gate SSM output, per layer)
variant "h":  y_t += C_t^T h'    (a learned state offset read out by the current C / q; linear in
              C_t, so it is computed outside the kernel from the decomposed forward)
variant "h0": learned initial state h_0 per layer (cheap ablation; Mamba2, and Mamba3 with the
              reference implementation)
All offsets are zero-initialised, so step 0 reproduces the pretrained model.
"""

import torch
import torch.nn as nn

from mambacls.models.adapters.base import Adapter
from mambacls.models.mixers import MixerMods, mixer_kind


def offset_shape(mixer, variant: str):
    kind = mixer_kind(mixer)
    if kind == "Mamba1":
        return {"y": (mixer.d_inner,), "h": (mixer.d_inner, mixer.d_state)}.get(variant)
    if kind == "Mamba2":
        return {"y": (mixer.nheads, mixer.headdim), "h": (mixer.nheads, mixer.headdim, mixer.d_state),
                "h0": (mixer.nheads, mixer.headdim, mixer.d_state)}[variant]
    if kind == "Mamba3":
        return {"y": (mixer.mimo_rank, mixer.nheads, mixer.headdim), "h": (mixer.nheads, mixer.headdim, mixer.d_state),
                "h0": (mixer.nheads, mixer.headdim, mixer.d_state)}[variant]
    return None


class StateOffsetAdapter(Adapter):
    name = "state_offset"

    def __init__(self, variant: str = "h", layers=None):
        super().__init__()
        if variant not in ("y", "h", "h0"):
            raise ValueError("variant must be 'y', 'h' or 'h0'")
        self.variant, self.layers = variant, layers
        self.offsets = nn.ParameterDict()

    def attach(self, backbone):
        super().attach(backbone)
        for i, block in enumerate(backbone.model.layers):
            if self.layers is not None and i not in self.layers:
                continue
            shape = offset_shape(block.mixer, self.variant)
            if shape is None:
                if mixer_kind(block.mixer) == "Mamba1" and self.variant == "h0":
                    raise NotImplementedError("h0 tuning is not available for Mamba (S6) layers")
                continue  # attention layers of hybrid models
            self.offsets[str(i)] = nn.Parameter(torch.zeros(shape, device=block.mixer.out_proj.weight.device))
        backbone.mods_providers.append(self.mods)

    def mods(self, i):
        p = self.offsets[str(i)] if str(i) in self.offsets else None
        if p is None:
            return None
        return MixerMods(**{{"y": "y_offset", "h": "h_offset", "h0": "h0"}[self.variant]: p})
