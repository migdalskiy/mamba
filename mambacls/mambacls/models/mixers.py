"""Decomposed forward passes of the Mamba, Mamba-2 and Mamba-3 mixers.

The fused training kernels hide the SSM internals: post-conv x, B, C, softplus(dt) and the
states are never materialised as Python tensors (spec §7.3). ``run_mixer`` computes a mixer in
stages using the module's own parameters (spec §7.4, workaround 1). Adapters and the internals
recorder hook in between the stages:

    projection -> conv -> selection parameters (dt, A, B, C, ...) -> scan -> offsets -> gate -> out_proj

The scan itself runs either through the repository kernels (``impl="kernel"``, GPU) or through
a pure-PyTorch chunked scan (``impl="reference"``, any device, fp32), so the reference is also
the CPU execution path and the oracle for kernel parity tests. When nothing needs to be
intercepted and ``impl="kernel"``, ``run_mixer`` simply calls the module (fused fast path).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.explain.mamba_lrp import (
    _causal_conv1d,
    _chunked_linear_attention,
    _mamba3_routing,
    _rms_norm_ref,
)
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DEFAULT_CHUNK = 64


@dataclass
class MixerMods:
    """Per-layer modifications of the SSM computation (all optional).

    y_offset: added to the pre-gate SSM output (State-offset Tuning, y variant).
        Mamba: (d_inner,); Mamba2: (nheads, headdim); Mamba3: (mimo_rank, nheads, headdim).
    h_offset: a state offset h' read out by the current C / q at every step,
        y_t += C_t^T h' (State-offset Tuning, h variant).
        Mamba: (d_inner, d_state); Mamba2 / Mamba3: (nheads, headdim, d_state).
    h0: learned initial state (initial-state tuning). Mamba2 (and Mamba3 reference path):
        (nheads, headdim, d_state).
    dt_filter: callable(dt, A) -> multiplicative mask on softplus(dt), shape (batch, seqlen,
        nheads); used by the LongMamba-style token filter. Mamba2 only.
    """

    y_offset: Optional[torch.Tensor] = None
    h_offset: Optional[torch.Tensor] = None
    h0: Optional[torch.Tensor] = None
    dt_filter: Optional[Callable] = None

    def is_empty(self):
        return self.y_offset is None and self.h_offset is None and self.h0 is None and self.dt_filter is None


def mixer_kind(mixer) -> str:
    if isinstance(mixer, Mamba3):
        return "Mamba3"
    if isinstance(mixer, Mamba2):
        return "Mamba2"
    if isinstance(mixer, Mamba):
        return "Mamba1"
    return type(mixer).__name__


def _silu_gate(y, z):
    return y * F.silu(z)


def _gated_rmsnorm(y, z, norm):
    """mamba_ssm.ops.triton.layernorm_gated.RMSNorm in plain PyTorch."""
    gs = norm.group_size or y.shape[-1]
    if not norm.norm_before_gate:
        y = y * F.silu(z)
    yg = rearrange(y, "... (g d) -> ... g d", d=gs)
    y = rearrange(yg * torch.rsqrt(yg.square().mean(-1, keepdim=True) + norm.eps), "... g d -> ... (g d)")
    y = y * norm.weight.to(y.dtype)
    if norm.bias is not None:
        y = y + norm.bias.to(y.dtype)
    if norm.norm_before_gate:
        y = y * F.silu(z)
    return y


def _float_linear(x, layer):
    return F.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype) if layer.bias is not None else None)


# ----------------------------------------------------------------------------------------------
# Mamba (S6)
# ----------------------------------------------------------------------------------------------

def _mamba1_forward(m: Mamba, u, impl, mods: MixerMods, record, seq_kwargs):
    if seq_kwargs:
        raise NotImplementedError("Mamba (S6) layers do not support varlen (seq_idx / cu_seqlens) inputs")
    if mods.h0 is not None or mods.dt_filter is not None:
        raise NotImplementedError("h0 / dt_filter mods are only implemented for Mamba2")
    x, z = _float_linear(u, m.in_proj).chunk(2, dim=-1)
    x = F.silu(_causal_conv1d(x, m.conv1d).to(x.dtype))
    dt_low, B, C = torch.split(_float_linear(x, m.x_proj), [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = F.linear(dt_low, m.dt_proj.weight.to(x.dtype))  # bias added inside the scan
    A = -torch.exp(m.A_log.float())
    args = (x.transpose(1, 2), dt.transpose(1, 2), A, B.transpose(1, 2), C.transpose(1, 2), m.D.float())
    kw = dict(z=None, delta_bias=m.dt_proj.bias.float(), delta_softplus=True)
    if impl == "kernel" and x.is_cuda:
        y = selective_scan_fn(*args, **kw)
    else:
        y = selective_scan_ref(*args, **kw)
    y = y.transpose(1, 2).to(x.dtype)  # (b l d)
    if record is not None:
        record.update(x=x, z=z, B=B, C=C, A=A, dt=F.softplus(dt + m.dt_proj.bias.float()), y=y)
    if mods.h_offset is not None:
        y = y + torch.einsum("bln,dn->bld", C, mods.h_offset.to(C.dtype))
    if mods.y_offset is not None:
        y = y + mods.y_offset.to(y.dtype)
    return _float_linear(_silu_gate(y, z), m.out_proj)


# ----------------------------------------------------------------------------------------------
# Mamba-2 (SSD)
# ----------------------------------------------------------------------------------------------

def _mamba2_forward(m: Mamba2, u, impl, mods: MixerMods, record, seq_kwargs):
    zxbcdt = _float_linear(u, m.in_proj)
    d_mlp = (zxbcdt.shape[-1] - 2 * m.d_ssm - 2 * m.ngroups * m.d_state - m.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1
    )
    seq_idx = seq_kwargs.get("seq_idx")
    if seq_idx is not None:
        from causal_conv1d import causal_conv1d_fn  # varlen conv needs the causal_conv1d package

        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2), rearrange(m.conv1d.weight, "d 1 w -> d w"), m.conv1d.bias,
            activation="silu", seq_idx=seq_idx,
        ).transpose(1, 2)
    else:
        xBC = F.silu(_causal_conv1d(xBC, m.conv1d).to(xBC.dtype))
    x, B, C = torch.split(xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
    A = -torch.exp(m.A_log.float())
    dt = F.softplus(dt.float() + m.dt_bias.float())
    if m.dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=m.dt_limit[0], max=m.dt_limit[1])
    if mods.dt_filter is not None:
        dt = dt * mods.dt_filter(dt, A)
    x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=m.ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=m.ngroups)
    D = rearrange(m.D.float(), "(h p) -> h p", p=m.headdim) if m.D_has_hdim else m.D.float()
    h0 = None if mods.h0 is None else mods.h0.float().expand(x.shape[0], *mods.h0.shape)
    if impl == "kernel":
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        y = mamba_chunk_scan_combined(
            x, dt.to(x.dtype), A, B, C, m.chunk_size, D=D, z=None, initial_states=h0,
            seq_idx=seq_idx, cu_seqlens=seq_kwargs.get("cu_seqlens"), dt_softplus=False,
        )
    else:
        heads_per_group = m.nheads // m.ngroups
        Bh = repeat(B, "b l g n -> b l 1 (g j) n", j=heads_per_group).float()
        Ch = repeat(C, "b l g n -> b l 1 (g j) n", j=heads_per_group).float()
        y = _chunked_linear_attention(
            Ch, Bh * dt[:, :, None, :, None], x.float().unsqueeze(2), dt * A, min(m.chunk_size, DEFAULT_CHUNK),
            detach=False, initial_state=h0,
        ).squeeze(2)
        y = y + x.float() * (D if D.dim() == 2 else D[:, None])
    y = y.to(x.dtype)
    if record is not None:
        record.update(x=x, z=z, B=B, C=C, A=A, dt=dt, D=D, y=y)
    if mods.h_offset is not None:
        Ch = C.repeat_interleave(m.nheads // m.ngroups, dim=2)
        y = y + torch.einsum("blhn,hpn->blhp", Ch, mods.h_offset.to(Ch.dtype))
    if mods.y_offset is not None:
        y = y + mods.y_offset.to(y.dtype)
    y = rearrange(y, "b l h p -> b l (h p)")
    y = _gated_rmsnorm(y, z, m.norm) if m.rmsnorm else _silu_gate(y, z)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    return _float_linear(y, m.out_proj)


# ----------------------------------------------------------------------------------------------
# Mamba-3 (trapezoidal, rotary, SISO / MIMO)
# ----------------------------------------------------------------------------------------------

def _mamba3_forward(m: Mamba3, u, impl, mods: MixerMods, record, seq_kwargs):
    if mods.dt_filter is not None:
        raise NotImplementedError("dt_filter (LongMamba) is only implemented for Mamba2")
    H, P, N, R, G = m.nheads, m.headdim, m.d_state, m.mimo_rank, m.num_bc_heads
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        _float_linear(u, m.in_proj),
        [m.d_inner, m.d_inner, N * G * R, N * G * R, H, H, H, m.num_rope_angles],
        dim=-1,
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=P)
    x = rearrange(x, "b l (h p) -> b l h p", p=P)
    routing = _mamba3_routing(m, B, C, dd_dt.float(), dd_A.float(), trap.float(), angles.float())
    if impl == "kernel":
        if mods.h0 is not None:
            raise NotImplementedError("h0 for Mamba3 is only available with impl='reference'")
        Bn = _rms_norm_ref(rearrange(B, "b l (r g n) -> b l r g n", r=R, g=G), m.B_norm.weight, None, m.B_norm.eps)
        Cn = _rms_norm_ref(rearrange(C, "b l (r g n) -> b l r g n", r=R, g=G), m.C_norm.weight, None, m.C_norm.eps)
        ADT = rearrange(routing["log_decay"], "b l h -> b h l")
        DT = rearrange(routing["dt"], "b l h -> b h l")
        Trap = rearrange(trap, "b l h -> b h l")
        Angles = angles.unsqueeze(-2).expand(-1, -1, H, -1).float()
        if m.is_mimo:
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo

            y = mamba3_mimo(
                Cn.to(x.dtype), Bn.to(x.dtype), x, ADT, DT, Trap, m.C_bias, m.B_bias, m.mimo_x, m.mimo_z,
                None, Angles, m.D, None, m.chunk_size, m.rotary_dim_divisor, x.dtype,
                cu_seqlens=seq_kwargs.get("cu_seqlens"),
            )  # (b, l, r, h, p): un-gated, not reduced over ranks
        else:
            from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

            y = mamba3_siso_combined(
                Cn.squeeze(2).to(x.dtype), Bn.squeeze(2).to(x.dtype), x, ADT, DT, Trap,
                m.C_bias.squeeze(1), m.B_bias.squeeze(1), Angles, D=m.D, Z=None,
                chunk_size=m.chunk_size, cu_seqlens=seq_kwargs.get("cu_seqlens"),
            ).unsqueeze(2)
        v = torch.einsum("blhp,hrp->blrhp", x, m.mimo_x.to(x.dtype)) if m.is_mimo else x.unsqueeze(2)
    else:
        v = torch.einsum("blhp,hrp->blrhp", x.float(), m.mimo_x.float()) if m.is_mimo else x.float().unsqueeze(2)
        y = _chunked_linear_attention(
            routing["q"], routing["k_scaled"], v, routing["log_decay"], max(DEFAULT_CHUNK // R, 1),
            detach=False, initial_state=None if mods.h0 is None else mods.h0.float().expand(x.shape[0], H, P, N),
        )
        y = y - torch.einsum("blRrh,blrhp->blRhp", routing["diag"], v) + v * m.D.float()[:, None]
    y = y.to(x.dtype)
    if record is not None:
        record.update(x=x, z=z, v=v, y=y, **routing)
    if mods.h_offset is not None:
        y = y + torch.einsum("blrhn,hpn->blrhp", routing["q"].to(y.dtype), mods.h_offset.to(y.dtype))
    if mods.y_offset is not None:
        y = y + mods.y_offset.to(y.dtype)
    zr = torch.einsum("blhp,hrp->blrhp", z, m.mimo_z.to(z.dtype)) if m.is_mimo else z.unsqueeze(2)
    if m.is_outproj_norm:
        y = y * torch.rsqrt(y.float().square().mean(-1, keepdim=True) + m.norm.eps).to(y.dtype)
        y = y * m.norm.weight.view(H, P).to(y.dtype)
    y = y * F.silu(zr)
    y = torch.einsum("blrhp,hrp->blhp", y, m.mimo_o.to(y.dtype)) if m.is_mimo else y.squeeze(2)
    return _float_linear(rearrange(y, "b l h p -> b l (h p)"), m.out_proj)


_FORWARDS = {"Mamba1": _mamba1_forward, "Mamba2": _mamba2_forward, "Mamba3": _mamba3_forward}


def run_mixer(
    mixer,
    u,
    impl: str = "kernel",
    seq_kwargs: Optional[Dict] = None,
    mods: Optional[MixerMods] = None,
    record: Optional[dict] = None,
):
    """Apply ``mixer`` to ``u`` (batch, seqlen, d_model).

    impl: "kernel" uses the module's fused kernels (fast path) unless ``mods`` or ``record``
        require the decomposed computation, whose scan then still runs through the kernels.
        "reference" runs everything in plain PyTorch in float32 (any device).
    seq_kwargs: ``seq_idx`` / ``cu_seqlens`` for varlen-packed batches (kernel impl only).
    record: if a dict is given, the internals (x, z, B, C, dt, A, ... ) are stored in it.
    """
    seq_kwargs = {k: v for k, v in (seq_kwargs or {}).items() if v is not None}
    mods = mods or MixerMods()
    kind = mixer_kind(mixer)
    if kind not in _FORWARDS:  # attention layers in hybrid models
        if not mods.is_empty():
            raise NotImplementedError(f"mixer mods are not supported for {kind} layers")
        return mixer(u) if not seq_kwargs else mixer(u, **seq_kwargs)
    if impl == "kernel" and mods.is_empty() and record is None:
        return mixer(u, **seq_kwargs) if kind != "Mamba1" else mixer(u)
    if impl == "reference":
        if seq_kwargs:
            raise ValueError("the reference implementation takes padded batches; unpack varlen inputs first")
        dtype = u.dtype
        return _FORWARDS[kind](mixer, u.float(), impl, mods, record, seq_kwargs).to(dtype)
    return _FORWARDS[kind](mixer, u, impl, mods, record, seq_kwargs)
