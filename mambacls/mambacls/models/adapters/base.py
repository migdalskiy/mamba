"""Adapter protocol (spec §8). An adapter decides which parameters train and may modify the
backbone (wrap weights, add offsets, prefixes, a backward branch)."""

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

# Parameters of the SSM dynamics: never weight-decayed, kept in fp32.
SSM_PARAM_NAMES = ("A_log", "dt_bias", "D", "B_bias", "C_bias")


class Adapter(nn.Module):
    name = "base"
    train_backbone = False

    def apply(self, backbone) -> None:
        """Freeze (or unfreeze) the backbone, then attach this adapter's modifications.

        ``nn.Module.apply(fn)`` (recursive function application) keeps working: it is used
        when the argument is not a module."""
        if not isinstance(backbone, nn.Module):
            return super().apply(backbone)
        self.freeze(backbone, self.train_backbone)
        self.attach(backbone)

    @staticmethod
    def freeze(backbone, train_backbone: bool) -> None:
        for p in backbone.parameters():
            p.requires_grad_(train_backbone)

    def attach(self, backbone) -> None:
        """Mutate / wrap modules. Subclasses call ``super().attach(backbone)`` first."""
        self._backbone = [backbone]  # list: do not register the backbone as a submodule

    @property
    def backbone(self):
        return self._backbone[0]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_delta(self) -> Dict[str, torch.Tensor]:
        """Adapter-only checkpoint."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # training-loop hooks
    def on_epoch_end(self, epoch: int, trainer=None) -> None:
        pass

    def after_optimizer_step(self) -> None:
        pass


class ProbeAdapter(Adapter):
    """Frozen backbone (linear probe; features can be cached)."""

    name = "probe"


class FullFinetune(Adapter):
    """All backbone parameters train. dt / A / D are kept in fp32."""

    name = "full"
    train_backbone = True

    def attach(self, backbone):
        super().attach(backbone)
        for n, p in backbone.named_parameters():
            if n.split(".")[-1] in SSM_PARAM_NAMES:
                p.data = p.data.float()

    def trainable_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def state_dict_delta(self):
        return {k: v.detach().cpu().clone() for k, v in self.backbone.state_dict().items()}


class CompositeAdapter(Adapter):
    """Applies several adapters in order (e.g. bidirectional retrofit + LoRA)."""

    def __init__(self, adapters: List[Adapter]):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
        self.name = "+".join(a.name for a in adapters)

    @property
    def train_backbone(self):
        return any(a.train_backbone for a in self.adapters)

    def attach(self, backbone):
        super().attach(backbone)
        for a in self.adapters:
            a.attach(backbone)

    def trainable_parameters(self):
        seen, out = set(), []
        for a in self.adapters:
            for p in a.trainable_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def state_dict_delta(self):
        out = {}
        for a in self.adapters:
            out.update({f"{a.name}.{k}": v for k, v in a.state_dict_delta().items()})
        return out

    def on_epoch_end(self, epoch, trainer=None):
        for a in self.adapters:
            a.on_epoch_end(epoch, trainer)

    def after_optimizer_step(self):
        for a in self.adapters:
            a.after_optimizer_step()
