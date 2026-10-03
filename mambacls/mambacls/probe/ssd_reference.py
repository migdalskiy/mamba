"""Reference SSM computations on captured internals (spec §7.4, workaround 2), in float32.

Given the selection parameters recorded by the decomposed mixer forward, compute the states h_t,
the per-head decay matrix L and the "hidden attention" matrix M (Ali, Zimerman & Wolf,
arXiv:2403.01590), with  y = M x + D x  (pre-gate). The tests check that reconstruction against
the mixer output.

* Mamba-2: A is a scalar per head, so M = L o (C B^T) o dt is exactly the SSD matrix.
* Mamba (S6): one matrix per channel, M[d] = sum_n C_t[n] exp(A[d,n] sum dt) dt_s B_s[n].
* Mamba-3 (experimental): the trapezoidal rule couples adjacent inputs, so M is no longer a pure
  1-semiseparable C B^T mask: M_{t,s} = (q_t . k_s) exp(sum_{u=s+1..t} dt_u A_u) c_{t,s} with
  c_{t,s} = gamma_s + (1 - lam_{s+1}) dt_{s+1} (s < t), c_{t,t} = gamma_t. Label these plots
  experimental until validated against kernel outputs.
"""

from typing import Dict

import torch
from einops import rearrange, repeat

from mamba_ssm.modules.ssd_minimal import segsum


def _heads(t, nheads):
    """(b, l, g, n) -> (b, l, h, n)."""
    return t.repeat_interleave(nheads // t.shape[2], dim=2)


def decay_matrix(log_decay: torch.Tensor) -> torch.Tensor:
    """log_decay (b, l, h) -> L (b, h, l, l) with L[t, s] = exp(sum_{u=s+1..t} log_decay_u), 0 for s > t."""
    return torch.exp(segsum(rearrange(log_decay.float(), "b l h -> b h l")))


def mamba2_reference(x, dt, A, B, C, D=None, return_states: bool = True) -> Dict[str, torch.Tensor]:
    """x (b,l,h,p), dt (b,l,h) [softplus applied], A (h,), B / C (b,l,g,n), D (h,) or (h,p)."""
    x, dt, A = x.float(), dt.float(), A.float()
    nheads = x.shape[2]
    Bh, Ch = _heads(B.float(), nheads), _heads(C.float(), nheads)
    L = decay_matrix(dt * A)
    M = torch.einsum("blhn,bshn->bhls", Ch, Bh) * L * rearrange(dt, "b s h -> b h 1 s")
    y = torch.einsum("bhls,bshp->blhp", M, x)
    if D is not None:
        y = y + x * (D.float() if D.dim() == 2 else D.float()[:, None])
    out = {"y": y, "decay": L, "ssd_matrix": M, "dt": dt}
    if return_states:
        h = torch.zeros(x.shape[0], nheads, x.shape[3], Bh.shape[-1], device=x.device)
        states = []
        for t in range(x.shape[1]):
            h = torch.exp(dt[:, t] * A)[..., None, None] * h + torch.einsum("bh,bhp,bhn->bhpn", dt[:, t], x[:, t], Bh[:, t])
            states.append(h)
        out["state"] = torch.stack(states, dim=1)  # (b, l, h, p, n)
        out["state_norm"] = out["state"].flatten(3).norm(dim=-1)  # (b, l, h)
    return out


def mamba1_reference(x, dt, A, B, C, D=None, channels=None, return_states: bool = True) -> Dict[str, torch.Tensor]:
    """x, dt (b,l,d); A (d,n); B, C (b,l,n). ``channels`` limits the (b,d,l,l) matrix to a subset."""
    x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()
    ch = torch.arange(x.shape[-1], device=x.device) if channels is None else torch.as_tensor(channels, device=x.device)
    cum = torch.cumsum(dt[..., ch], dim=1)  # (b, l, c)
    seg = cum[:, :, None, :] - cum[:, None, :, :]  # (b, t, s, c) = sum_{u=s+1..t} dt_u
    causal = torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device))[None, :, :, None]
    expo = torch.exp(torch.einsum("btsc,cn->btscn", seg, A[ch]).masked_fill(~causal[..., None], float("-inf")))
    M = torch.einsum("btn,bsn,btscn,bsc->bcts", C, B, expo, dt[..., ch])
    y = torch.einsum("bcts,bsc->btc", M, x[..., ch])
    if D is not None:
        y = y + x[..., ch] * D.float()[ch]
    out = {"y": y, "ssd_matrix": M, "dt": dt}
    if return_states:
        h = torch.zeros(x.shape[0], x.shape[2], A.shape[1], device=x.device)
        states = []
        for t in range(x.shape[1]):
            h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * x[:, t])[..., None] * B[:, t, None, :]
            states.append(h)
        out["state"] = torch.stack(states, dim=1)  # (b, l, d, n)
        out["state_norm"] = out["state"].norm(dim=-1)  # (b, l, d)
    return out


def mamba3_reference(record: Dict[str, torch.Tensor], D=None, return_states: bool = True) -> Dict[str, torch.Tensor]:
    """Experimental hidden attention of Mamba-3 from the routing recorded by the decomposed forward
    (q, k, k_scaled, diag, log_decay, dt, lam; v (b,l,r,h,p)). M: (b, h, R, r, l, l)."""
    q, k, ks = record["q"].float(), record["k"].float(), record["k_scaled"].float()
    v, ld, dt, lam = record["v"].float(), record["log_decay"].float(), record["dt"].float(), record["lam"].float()
    L = decay_matrix(ld)  # (b, h, l, l)
    M = torch.einsum("btRhn,bsrhn->bhRrts", q, ks) * L[:, :, None, None]
    eye = torch.eye(q.shape[1], device=q.device)
    M = M - torch.einsum("btRrh,ts->bhRrts", record["diag"].float(), eye)
    y = torch.einsum("bhRrts,bsrhp->btRhp", M, v)
    if D is not None:
        y = y + v * D.float()[:, None]
    out = {"y": y, "decay": L, "ssd_matrix": M, "dt": dt, "experimental": torch.tensor(True)}
    if return_states:
        b, l, R, H, P = v.shape
        alpha = torch.exp(ld)
        gamma = lam * dt
        beta = (1 - lam) * dt * alpha
        h = torch.zeros(b, H, P, q.shape[-1], device=q.device)
        prev = torch.zeros(b, H, P, q.shape[-1], device=q.device)
        states = []
        for t in range(l):
            cur = torch.einsum("brhn,brhp->bhpn", k[:, t], v[:, t])
            h = alpha[:, t, :, None, None] * h + beta[:, t, :, None, None] * prev + gamma[:, t, :, None, None] * cur
            prev = cur
            states.append(h)
        out["state"] = torch.stack(states, dim=1)
        out["state_norm"] = out["state"].flatten(3).norm(dim=-1)
    return out
