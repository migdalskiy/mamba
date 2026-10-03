"""LoRA on linear projections (spec §3.3), implemented as a weight parametrization.

The fused Mamba-2 kernel reads ``out_proj.weight`` directly instead of calling the module, so a
forward-hook LoRA would silently be skipped on GPU. A parametrization makes ``.weight`` itself
W + scale * B A, which every code path sees. Because of that, dropout is applied to the factor
A (DropConnect) rather than to the layer input.
"""

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from mambacls.models.adapters.base import Adapter

DEFAULT_TARGETS = ("in_proj", "out_proj")  # never conv1d, A_log, D, dt_bias


class LoRAParametrization(nn.Module):
    def __init__(self, out_features: int, in_features: int, r: int, alpha: float, dropout: float = 0.0,
                 branches: Sequence[str] = ("fwd",)):
        super().__init__()
        self.r, self.scale, self.dropout = r, alpha / r, dropout
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for b in branches:
            A = torch.empty(r, in_features)
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            self.A[b] = nn.Parameter(A)
            self.B[b] = nn.Parameter(torch.zeros(out_features, r))  # zero: starts at the pretrained W
        self.active = branches[0]
        self.merged = False

    def delta(self, branch=None, use_dropout: bool = True) -> torch.Tensor:
        branch = branch or self.active
        A = self.A[branch]
        if use_dropout and self.training and self.dropout > 0:
            A = nn.functional.dropout(A, self.dropout)
        return (self.B[branch] @ A) * self.scale

    def forward(self, W):
        if self.merged:
            return W
        return W + self.delta().to(W.dtype)


class LoRAAdapter(Adapter):
    name = "lora"

    def __init__(self, r: int = 16, alpha: float = None, dropout: float = 0.05,
                 targets: Iterable[str] = DEFAULT_TARGETS, layers=None, branches: Sequence[str] = ("fwd",)):
        super().__init__()
        self.r, self.alpha, self.dropout = r, alpha if alpha is not None else 2 * r, dropout
        self.targets, self.layers, self.branches = tuple(targets), layers, tuple(branches)
        self.loras = nn.ModuleDict()

    def _targets(self, backbone):
        for i, block in enumerate(backbone.model.layers):
            if self.layers is not None and i not in self.layers:
                continue
            for name, mod in block.mixer.named_modules():
                if isinstance(mod, nn.Linear) and name.split(".")[-1] in self.targets:
                    yield f"{i}.{name}", mod

    def attach(self, backbone):
        super().attach(backbone)
        for key, linear in self._targets(backbone):
            p = LoRAParametrization(linear.out_features, linear.in_features, self.r, self.alpha, self.dropout,
                                    self.branches).to(linear.weight.device)
            parametrize.register_parametrization(linear, "weight", p)
            self.loras[key.replace(".", "__")] = p
        if not len(self.loras):
            raise ValueError(f"LoRA found no targets {self.targets} in the backbone")

    def trainable_parameters(self):
        return [p for p in self.loras.parameters() if p.requires_grad]

    def set_branch(self, branch: str):
        for p in self.loras.values():
            p.active = branch

    def _linears(self):
        return list(self._targets(self.backbone))

    def merge(self):
        """Fold W + scale B A into the stored weight (single-branch LoRA only)."""
        if len(self.branches) > 1:
            raise ValueError("cannot merge a multi-branch (untied bidirectional) LoRA")
        with torch.no_grad():
            for key, linear in self._linears():
                p = self.loras[key.replace(".", "__")]
                if not p.merged:
                    orig = linear.parametrizations.weight.original
                    orig.add_(p.delta(self.branches[0], use_dropout=False).to(orig.dtype))
                    p.merged = True

    def unmerge(self):
        with torch.no_grad():
            for key, linear in self._linears():
                p = self.loras[key.replace(".", "__")]
                if p.merged:
                    orig = linear.parametrizations.weight.original
                    orig.sub_(p.delta(self.branches[0], use_dropout=False).to(orig.dtype))
                    p.merged = False

    def remove(self, merge: bool = True):
        """Remove the parametrizations, keeping the merged weights if ``merge``."""
        for key, linear in self._linears():
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=merge)
        self.loras = nn.ModuleDict()
