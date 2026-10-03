"""Forward hooks on a plain ``MixerModel`` (spec §7.2), for code that uses the model directly.

A ``Block`` returns ``(hidden_states, residual)``; the true residual stream after block i is
``hidden_states + residual``. ``MambaBackbone(return_all_layers=True)`` returns the same
quantity without hooks."""

from typing import Dict, List, Optional

import torch


class ResidualStreamHooks:
    def __init__(self, mixer_model, layers: Optional[List[int]] = None, capture_mixer_io: bool = False):
        self.model = mixer_model
        n = len(mixer_model.layers)
        self.layers = list(range(n)) if layers is None else layers
        self.capture_mixer_io = capture_mixer_io
        self.streams: Dict[int, torch.Tensor] = {}
        self.mixer_io: Dict[int, Dict[str, torch.Tensor]] = {}
        self._handles = []

    def __enter__(self):
        for i in self.layers:
            block = self.model.layers[i]
            self._handles.append(block.register_forward_hook(self._block_hook(i)))
            if self.capture_mixer_io:
                self._handles.append(block.mixer.in_proj.register_forward_hook(self._io_hook(i, "in_proj")))
                self._handles.append(block.mixer.register_forward_hook(self._io_hook(i, "mixer")))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def _block_hook(self, i):
        def hook(module, inputs, output):
            hidden, residual = output
            self.streams[i] = (hidden + residual).detach()
        return hook

    def _io_hook(self, i, name):
        def hook(module, inputs, output):
            self.mixer_io.setdefault(i, {})[f"{name}_in"] = inputs[0].detach()
            self.mixer_io[i][f"{name}_out"] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
