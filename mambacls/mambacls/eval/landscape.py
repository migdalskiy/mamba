"""Loss landscape around the trained head + adapter (spec §7.1): 1D interpolation from init to the
final solution and a 2D filter-normalised slice (Li et al., 2018)."""

from typing import Callable, Dict, List

import numpy as np
import torch


def _params(model) -> Dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _loss(model, batches, device):
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        b = b.to(device)
        out = model(b)
        tot += float(out.loss) * len(b.lengths)
        n += len(b.lengths)
    return tot / max(n, 1)


@torch.no_grad()
def interpolate_1d(model, init_state: Dict[str, torch.Tensor], batches, alphas=None, device=None):
    """loss((1 - a) * theta_init + a * theta_final) for a in ``alphas``."""
    device = device or next(model.parameters()).device
    alphas = np.linspace(-0.25, 1.25, 31) if alphas is None else alphas
    params = _params(model)
    final = {n: p.detach().clone() for n, p in params.items()}
    rows = []
    for a in alphas:
        for n, p in params.items():
            p.copy_((1 - a) * init_state[n].to(p) + a * final[n])
        rows.append({"alpha": float(a), "loss": _loss(model, batches, device)})
    for n, p in params.items():
        p.copy_(final[n])
    return rows


def _filter_normalised_direction(params, generator):
    d = {}
    for n, p in params.items():
        r = torch.randn(p.shape, generator=generator).to(p)
        if p.dim() <= 1:
            d[n] = r / r.norm().clamp_min(1e-12) * p.norm()
        else:  # per filter (row) normalisation
            pr = p.reshape(p.shape[0], -1)
            rr = r.reshape(r.shape[0], -1)
            rr = rr / rr.norm(dim=1, keepdim=True).clamp_min(1e-12) * pr.norm(dim=1, keepdim=True)
            d[n] = rr.reshape(p.shape)
    return d


@torch.no_grad()
def surface_2d(model, batches, span: float = 1.0, steps: int = 21, seed: int = 0, device=None):
    device = device or next(model.parameters()).device
    params = _params(model)
    g = torch.Generator().manual_seed(seed)
    d1, d2 = _filter_normalised_direction(params, g), _filter_normalised_direction(params, g)
    center = {n: p.detach().clone() for n, p in params.items()}
    grid = np.linspace(-span, span, steps)
    Z = np.zeros((steps, steps))
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            for n, p in params.items():
                p.copy_(center[n] + a * d1[n] + b * d2[n])
            Z[i, j] = _loss(model, batches, device)
    for n, p in params.items():
        p.copy_(center[n])
    return {"x": grid, "y": grid, "z": Z}
