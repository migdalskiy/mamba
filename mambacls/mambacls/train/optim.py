"""AdamW with separate head / backbone learning rates, no weight decay on SSM dynamics and norms,
linear warm-up and cosine decay (spec §3.2)."""

import math
from typing import Iterable, Sequence

import torch

NO_DECAY_DEFAULT = ("A_log", "D", "dt_bias", "norm", "bias")


def no_decay(name: str, tokens: Sequence[str] = NO_DECAY_DEFAULT) -> bool:
    parts = name.split(".")
    last = parts[-1]
    for tok in tokens:
        if last == tok or (tok == "bias" and last.endswith("bias")) or (tok == "norm" and any("norm" in p for p in parts)):
            return True
    return False


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float = 0.01,
                 no_decay_tokens: Sequence[str] = NO_DECAY_DEFAULT):
    names = {id(p): n for n, p in model.named_parameters()}
    head, body = model.trainable_parameter_groups()
    groups = []
    for params, lr, tag in ((head, lr_head, "head"), (body, lr_backbone, "backbone")):
        dec = [p for p in params if not no_decay(names.get(id(p), ""), no_decay_tokens) and p.dim() > 0]
        nodec = [p for p in params if no_decay(names.get(id(p), ""), no_decay_tokens) or p.dim() == 0]
        if dec:
            groups.append({"params": dec, "lr": lr, "weight_decay": weight_decay, "group": f"{tag}_decay"})
        if nodec:
            groups.append({"params": nodec, "lr": lr, "weight_decay": 0.0, "group": f"{tag}_no_decay"})
    return groups


def build_optimizer(model, lr_backbone=2e-5, lr_head=1e-3, weight_decay=0.01, no_decay_tokens=NO_DECAY_DEFAULT,
                    betas=(0.9, 0.999), eps=1e-8):
    groups = param_groups(model, lr_backbone, lr_head, weight_decay, no_decay_tokens)
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(groups, betas=betas, eps=eps)


def warmup_cosine(optimizer, total_steps: int, warmup_ratio: float = 0.06, schedule: str = "cosine", min_ratio: float = 0.0):
    warmup = max(1, int(round(warmup_ratio * total_steps)))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        if schedule == "linear":
            return max(min_ratio, 1.0 - progress)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
