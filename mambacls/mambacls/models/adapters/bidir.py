"""Weight-tied bidirectional retrofit of a causal checkpoint (spec §3.5; Caduceus BiMamba style).

Per layer: y = mixer(x) + g * rev(mixer(rev(x))), where ``rev`` flips every sequence within its
real length (right padding) or within its varlen segment.

modes:
  tied_add    g = 1, no new parameters
  tied_gate   g a learned scalar per layer, initialised at 0, so step 0 reproduces the causal model
              exactly (recommended default)
  untied_lora the backward branch uses its own LoRA factors (the LoRA adapter is created with
              branches ("fwd", "bwd") and switched by the backbone); gate as in tied_gate
Pair with LoRA or full fine-tuning: a frozen backbone can only use the backward signal through g.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from mambacls.models.adapters.base import Adapter
from mambacls.models.adapters.lora import LoRAAdapter


class BidirectionalConfig:
    def __init__(self, gates: nn.ParameterDict, fixed_gate: Optional[float], layers):
        self.gates, self.fixed_gate, self.layers = gates, fixed_gate, layers

    def applies(self, i: int) -> bool:
        return self.layers is None or i in self.layers

    def gate(self, i: int) -> torch.Tensor:
        if self.fixed_gate is not None:
            return torch.tensor(self.fixed_gate)
        return self.gates[str(i)]


class BidirAdapter(Adapter):
    def __init__(self, mode: str = "tied_gate", layers: Optional[Sequence[int]] = None, lora_kwargs: Optional[dict] = None):
        super().__init__()
        if mode not in ("tied_add", "tied_gate", "untied_lora"):
            raise ValueError("mode must be tied_add, tied_gate or untied_lora")
        self.mode, self.layers = mode, layers
        self.name = f"bidir_{mode}"
        self.gates = nn.ParameterDict()
        self.lora = LoRAAdapter(branches=("fwd", "bwd"), **(lora_kwargs or {})) if mode == "untied_lora" else None
        self.train_backbone = False

    def attach(self, backbone):
        super().attach(backbone)
        device = backbone.embedding.weight.device
        if self.mode != "tied_add":
            for i in range(backbone.n_layers):
                if self.layers is None or i in self.layers:
                    self.gates[str(i)] = nn.Parameter(torch.zeros((), device=device))
        backbone.bidir = BidirectionalConfig(self.gates, 1.0 if self.mode == "tied_add" else None, self.layers)
        if self.lora is not None:
            self.lora.attach(backbone)
            backbone.branch_hooks.append(self.lora.set_branch)

    def trainable_parameters(self):
        params = list(self.gates.values())
        if self.lora is not None:
            params += self.lora.trainable_parameters()
        return params
