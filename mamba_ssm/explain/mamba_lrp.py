"""MambaLRP: Layer-wise Relevance Propagation for Mamba, Mamba-2 and Mamba-3 language models.

Builds on "MambaLRP: Explaining Selective State Space Sequence Models",
Jafari, Montavon, Müller, Eberle (NeurIPS 2024), https://arxiv.org/abs/2406.07592,
and extends its rules to Mamba-2 and Mamba-3.

Principle
---------
Relevance is computed as Gradient x Input on a *modified* forward pass: the modified pass
computes exactly the same values as the model, but some tensors are detached, so its gradient
(written d~) differs. For the explained logit f and a representation a (e.g. the token
embeddings), R(a) = a * d~f/da.

Relevance is conserved through an operation b = g(a), sum_i R(a_i) = sum_j R(b_j), if the
modified operation is degree-1 homogeneous: sum_i a_i d~g_j/da_i = g_j (Euler's theorem).
Bias-free linear maps (embeddings, in/out projections, depthwise conv1d, residual additions)
satisfy this as they are. For every other operation, the rule below picks what to detach so
that the modified operation is linear in the inputs that carry token content:

1. Element-wise activations, f(x) = x * [f(x) / x]: the ratio is detached (identity rule).
   Plain Gradient x Input would give x * f'(x), which is not f(x).
2. Products of two content-carrying factors a * b (output gates y * silu(z), gated MLPs,
   per-rank MIMO gates): a * b is degree 2, so plain Gradient x Input returns 2ab. Using
   0.5 * ab + 0.5 * (ab).detach() gives each factor half the relevance (half rule).
3. Normalisation x / sigma(x) (LayerNorm, RMSNorm, the gated / head-wise norms inside the
   mixers): this is degree 0 in x, so plain Gradient x Input gives zero total relevance.
   Detaching sigma makes it linear (LN rule).
4. Selective state-space mixing. Every mixer computes, per channel or head,
       y_t = sum_{s <= t} W_{t,s}(theta) v_s + D v_t,
   where v is the content ("value") stream and the mixing weights W depend on the input
   through the selection parameters theta. W involves exponentials of cumulative sums,
   products of dt, B and C and, for Mamba-3, rotations and a trapezoidal rule, so it is not
   homogeneous in theta. Theta is also computed from the same tokens as v, so gradients
   through theta break conservation and double count. Detaching theta (treating W as fixed)
   makes the mixer a linear map of v: all relevance flows along the content path, and W
   decides how much of token s's content reaches position t. The derivation of W for each
   architecture is in the docstrings of ``_mamba1_lrp``, ``_mamba2_lrp`` and ``_mamba3_lrp``.

Rules 1 to 3 cover everything else, so relevance is conserved through the whole network. In
a model without biases in the content path, the token relevances at every layer sum exactly
to the explained logit. Such biases are the conv1d bias of Mamba and Mamba-2 (on by default),
optional projection biases and LayerNorm biases. Mamba-3 has no convolution, so it conserves
exactly with default settings.

The modified forward is a pure PyTorch re-implementation of ``MambaLMHeadModel``. It does not
need the Triton, TileLang or CUDA kernels and also runs on CPU.

Example:
    >>> from mamba_ssm.explain import mamba_lrp
    >>> out = model.generate(input_ids, max_length=input_ids.shape[1] + 20)
    >>> output_ids = out[:, input_ids.shape[1]:]
    >>> attr = mamba_lrp(model, input_ids, output_ids, output_index=3)
    >>> attr.input_relevance  # (batch, len(input_ids)) relative importance of each input token
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.ssd_minimal import segsum


@dataclass
class MambaLRPAttribution:
    """Result of :func:`mamba_lrp`.

    Token positions cover the context the explained token was predicted from: all input
    tokens followed by the output tokens generated before the token of interest.

    Attributes:
        relevance: (batch, n_context) relevance of each context token, for the first layer
            requested (the embedding layer by default).
        input_relevance: (batch, n_input) the part of ``relevance`` on the input tokens.
        output_relevance: (batch, n_prev_output) the part of ``relevance`` on the previously
            generated output tokens (empty when explaining the first output token).
        layer_relevance: {layer: (batch, n_context)} relevance at the input of every requested
            layer. Layer 0 is the token embedding, layer ``k`` is the residual stream entering
            block ``k`` and layer ``n_layer`` is the residual stream entering the final norm.
        raw_total_relevance: {layer: (batch,)} sum of the unnormalised relevance at each
            requested layer. Equals ``target_logit`` up to the relevance absorbed by biases.
        target_logit: (batch,) the logit being explained.
        target_token_ids: (batch,) the token whose logit is explained.
        position: sequence position (in the concatenated input + output sequence) whose
            output distribution predicts the target token.
        layers: the resolved (non-negative) layer indices, in the requested order.
    """

    relevance: torch.Tensor
    input_relevance: torch.Tensor
    output_relevance: torch.Tensor
    layer_relevance: Dict[int, torch.Tensor]
    raw_total_relevance: Dict[int, torch.Tensor]
    target_logit: torch.Tensor
    target_token_ids: torch.Tensor
    position: int
    layers: List[int] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------
# LRP building blocks. Each one returns the same value as the original op; only the gradient
# (and hence Gradient x Input) differs.
# ----------------------------------------------------------------------------------------------

def _act_lrp(x, act_fn):
    """Identity rule for element-wise activations: all relevance passes to the input."""
    y = act_fn(x)
    ratio = torch.where(x == 0, torch.zeros_like(x), y / torch.where(x == 0, torch.ones_like(x), x))
    return x * ratio.detach()


def _silu_lrp(x):
    return x * torch.sigmoid(x).detach()


def _gate_lrp(a, b):
    """Half rule for a product of two relevance-carrying tensors."""
    out = a * b
    return 0.5 * out + 0.5 * out.detach()


def _linear(x, layer):
    bias = layer.bias.float() if layer.bias is not None else None
    return F.linear(x, layer.weight.float(), bias)


def _causal_conv1d(x, conv):
    """Depthwise causal conv1d on (batch, seqlen, channels), as in Mamba / Mamba2."""
    seqlen = x.shape[1]
    out = F.conv1d(
        x.transpose(1, 2),
        conv.weight.float(),
        conv.bias.float() if conv.bias is not None else None,
        padding=conv.weight.shape[-1] - 1,
        groups=conv.weight.shape[0],
    )
    return out[..., :seqlen].transpose(1, 2)


def _rms_norm_lrp(x, weight, bias, eps, group_size=None):
    """RMSNorm with the (group-wise) normaliser detached."""
    if group_size is not None and group_size != x.shape[-1]:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = torch.rsqrt(x_group.square().mean(dim=-1, keepdim=True) + eps).detach()
        out = rearrange(x_group * rstd, "... g d -> ... (g d)")
    else:
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps).detach()
        out = x * rstd
    out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out


def _norm_lrp(x, norm):
    """Block / final norm (nn.LayerNorm or RMSNorm) with 1/std detached."""
    if isinstance(norm, nn.LayerNorm):
        xc = x - x.mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(xc.square().mean(dim=-1, keepdim=True) + norm.eps).detach()
        out = xc * rstd
        if norm.weight is not None:
            out = out * norm.weight.float()
        if norm.bias is not None:
            out = out + norm.bias.float()
        return out
    if type(norm).__name__ == "RMSNorm":
        return _rms_norm_lrp(x, norm.weight, getattr(norm, "bias", None), norm.eps)
    raise NotImplementedError(f"MambaLRP: unsupported normalisation layer {type(norm).__name__}")


# ----------------------------------------------------------------------------------------------
# Linear sequence mixers with fixed (detached) mixing weights
# ----------------------------------------------------------------------------------------------

def _chunked_linear_attention(q, k, v, log_decay, chunk_size):
    """Causal linear attention with scalar per-head decay and fixed q, k and decay.

        y[t, R] = sum_{s <= t} exp(sum_{u=s+1..t} log_decay[u]) sum_r (q[t, R] . k[s, r]) v[s, r]

    This is the SSD algorithm of Mamba-2 (``ssd_minimal_discrete``), generalised to the MIMO
    rank dimension of Mamba-3, where all ranks at one time step read the same state.
    q, k: (batch, seqlen, rank, nheads, dstate); v: (batch, seqlen, rank, nheads, headdim);
    log_decay: (batch, seqlen, nheads). q, k and log_decay are detached, so y is linear in v.
    Returns (batch, seqlen, rank, nheads, headdim).
    """
    q, k, log_decay = q.detach(), k.detach(), log_decay.detach()
    seqlen = v.shape[1]
    chunk_size = min(chunk_size, seqlen)
    pad = (-seqlen) % chunk_size
    if pad > 0:
        q, k, v = [F.pad(t, (0, 0, 0, 0, 0, 0, 0, pad)) for t in (q, k, v)]
        log_decay = F.pad(log_decay, (0, 0, 0, pad))
    q, k, v = [rearrange(t, "b (c l) r h n -> b c l r h n", l=chunk_size) for t in (q, k, v)]
    a = rearrange(log_decay, "b (c l) h -> b h c l", l=chunk_size)
    a_cumsum = torch.cumsum(a, dim=-1)

    # 1. Within each chunk: materialise the (fixed) mixing weights.
    weights = torch.einsum("bclRhn,bcsrhn->bhclsRr", q, k) * torch.exp(segsum(a))[..., None, None]
    y = torch.einsum("bhclsRr,bcsrhp->bclRhp", weights, v)
    # 2. State at the end of each chunk.
    decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
    states = torch.einsum("bcsrhn,bhcs,bcsrhp->bchpn", k, decay_states, v)
    # 3. Pass states between chunks.
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cumsum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # 4. Contribution of earlier chunks.
    y = y + torch.einsum("bclRhn,bchpn,bhcl->bclRhp", q, states, torch.exp(a_cumsum))
    return rearrange(y, "b c l r h p -> b (c l) r h p")[:, :seqlen]


class _SelectiveScanLRP(torch.autograd.Function):
    """Mamba (S6) selective scan with fixed dt, A, B, C: a linear map of x.

        h_t[d, n] = exp(dt_t[d] A[d, n]) h_{t-1}[d, n] + dt_t[d] B_t[n] x_t[d]
        y_t[d]    = sum_n C_t[n] h_t[d, n]

    The state has a separate decay per (channel, state) pair, so the chunked form would need a
    (seqlen x seqlen) decay matrix per (channel, state). Instead the scan is sequential and the
    states are not stored. Because the map is linear in x, the backward pass is the adjoint
    scan, run backwards in time:
        g_t = C_t gy_t + exp(dt_{t+1} A) g_{t+1},   gx_t[d] = dt_t[d] sum_n B_t[n] g_t[d, n].
    Memory is O(batch * d_inner * d_state) on top of the inputs.
    x, dt: (batch, seqlen, d_inner); A: (d_inner, d_state); B, C: (batch, seqlen, d_state).
    """

    @staticmethod
    def forward(ctx, x, dt, A, B, C):
        ctx.save_for_backward(dt, A, B, C)
        batch, seqlen, dim = x.shape
        h = x.new_zeros(batch, dim, A.shape[1])
        ys = []
        for t in range(seqlen):
            h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * x[:, t])[..., None] * B[:, t, None, :]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1)

    @staticmethod
    def backward(ctx, grad_y):
        dt, A, B, C = ctx.saved_tensors
        batch, seqlen, dim = dt.shape
        g = grad_y.new_zeros(batch, dim, A.shape[1])
        grads = [None] * seqlen
        for t in reversed(range(seqlen)):
            g = g + grad_y[:, t, :, None] * C[:, t, None, :]
            grads[t] = dt[:, t] * torch.einsum("bdn,bn->bd", g, B[:, t])
            g = g * torch.exp(dt[:, t, :, None] * A)
        return torch.stack(grads, dim=1), None, None, None, None


# ----------------------------------------------------------------------------------------------
# Mixers
# ----------------------------------------------------------------------------------------------

def _mamba1_lrp(mixer: Mamba, u, chunk_size=None):
    """Modified ``Mamba.forward`` (selective SSM, S6).

    Forward:  [x, z] = in_proj(u);  x = silu(conv1d(x));  [dt_low, B, C] = x_proj(x);
              dt = softplus(dt_proj(dt_low));  A = -exp(A_log)   (d_inner x d_state)
              y = S6(x; dt, A, B, C) + D * x;   out = out_proj(y * silu(z))
    Unrolling the recurrence (``_SelectiveScanLRP``) gives
              y_t[d] = sum_{s<=t} W_{t,s}[d] x_s[d] + D[d] x_t[d],
              W_{t,s}[d] = sum_n C_t[n] exp(A[d, n] sum_{u=s+1..t} dt_u[d]) dt_s[d] B_s[n].
    In Mamba the selection parameters dt, B and C are computed from x itself (x_proj on the
    conv output). Without detaching, Gradient x Input would count x's relevance twice: once as
    content and once through W. Detaching dt, B and C (the MambaLRP S6 rule) makes y linear in
    x. The conv1d is linear (LRP-0), silu uses the identity rule and the output gate the half
    rule.
    """
    xz = _linear(u, mixer.in_proj)
    x, z = xz.chunk(2, dim=-1)
    x = _silu_lrp(_causal_conv1d(x, mixer.conv1d))
    dt, B, C = torch.split(
        _linear(x, mixer.x_proj), [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1
    )
    dt = F.softplus(_linear(dt, mixer.dt_proj))
    A = -torch.exp(mixer.A_log.float())
    y = _SelectiveScanLRP.apply(x, dt.detach(), A.detach(), B.detach(), C.detach())
    y = y + x * mixer.D.float()
    y = _gate_lrp(y, _silu_lrp(z))
    return _linear(y, mixer.out_proj)


def _mamba2_lrp(mixer: Mamba2, u, chunk_size=None):
    """Modified ``Mamba2.forward`` (SSD, training / prefill path).

    Forward:  [z0, x0, z, xBC, dt] = in_proj(u);  [x, B, C] = silu(conv1d(xBC));
              dt = softplus(dt + dt_bias);  A = -exp(A_log)   (one scalar per head)
              y_t = sum_{s<=t} W_{t,s} x_s + D x_t,
              W_{t,s} = (C_t . B_s) exp(A sum_{u=s+1..t} dt_u) dt_s   (per head)
              y = norm(y, z) or y * silu(z);   out = out_proj([silu(z0) * x0, y])
    Here B and C come from the same conv as x but are separate channels, and W is
    detached as in the S6 rule. The gated RMSNorm detaches its normaliser and uses the half
    rule for the gate. The optional MLP part, silu(z0) * x0, uses the half rule.
    """
    if mixer.process_group is not None:
        raise NotImplementedError("MambaLRP does not support tensor-parallel Mamba2 layers")
    chunk_size = 64 if chunk_size is None else chunk_size
    zxbcdt = _linear(u, mixer.in_proj)
    d_ssm, ngroups, d_state, nheads = mixer.d_ssm, mixer.ngroups, mixer.d_state, mixer.nheads
    d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads], dim=-1
    )
    xBC = _silu_lrp(_causal_conv1d(xBC, mixer.conv1d))
    x, B, C = torch.split(xBC, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1)

    A = -torch.exp(mixer.A_log.float())
    dt = F.softplus(dt + mixer.dt_bias.float())
    if mixer.dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=mixer.dt_limit[0], max=mixer.dt_limit[1])
    heads_per_group = nheads // ngroups
    B = repeat(B, "b l (g n) -> b l 1 (g j) n", g=ngroups, j=heads_per_group)
    C = repeat(C, "b l (g n) -> b l 1 (g j) n", g=ngroups, j=heads_per_group)
    x = rearrange(x, "b l (h p) -> b l 1 h p", p=mixer.headdim)
    y = _chunked_linear_attention(C, B * dt[:, :, None, :, None], x, dt * A, chunk_size)
    D = mixer.D.float()
    D = rearrange(D, "(h p) -> h p", p=mixer.headdim) if mixer.D_has_hdim else D.unsqueeze(-1)
    y = y + x * D
    y = rearrange(y, "b l 1 h p -> b l (h p)")

    gate = _silu_lrp(z)
    if mixer.rmsnorm:
        norm = mixer.norm
        if norm.norm_before_gate:
            y = _rms_norm_lrp(y, norm.weight, norm.bias, norm.eps, norm.group_size)
            y = _gate_lrp(y, gate)
        else:
            y = _gate_lrp(y, gate)
            y = _rms_norm_lrp(y, norm.weight, norm.bias, norm.eps, norm.group_size)
    else:
        y = _gate_lrp(y, gate)
    if d_mlp > 0:
        y = torch.cat([_gate_lrp(_silu_lrp(z0), x0), y], dim=-1)
    return _linear(y, mixer.out_proj)


def _rotate(x, cos, sin, pairwise):
    """Rotate pairs of entries of the last dim of x. cos, sin: (..., dstate // 2).

    pairwise=True rotates (x[2i], x[2i+1]) (Mamba-3 SISO kernels); otherwise (x[i], x[i + n/2])
    (Mamba-3 MIMO kernels).
    """
    if pairwise:
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
    half = x.shape[-1] // 2
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _mamba3_lrp(mixer: Mamba3, u, chunk_size=None):
    """Modified ``Mamba3.forward`` (SISO and MIMO).

    Forward, per head (R = mimo_rank, R = 1 for SISO; no conv1d):
        [z, x, B, C, dd_dt, dd_A, trap, phi] = in_proj(u)
        dt_t = softplus(dd_dt + dt_bias),  A_t = -max(heavy_tail(dd_A), A_floor)   (data-dependent)
        lam_t = sigmoid(trap),  theta_t = sum_{u<=t} pi tanh(phi_u) dt_u          (rotation angles)
        q_{t,r} = Rot(theta_t)(rmsnorm(C_{t,r}) + C_bias_r)
        k_{t,r} = Rot(theta_t)(rmsnorm(B_{t,r}) + B_bias_r)
        v_{t,r} = mimo_x_r * x_t,  z_{t,r} = mimo_z_r * z_t        (identity maps for SISO)
    Trapezoidal recurrence with alpha_t = exp(dt_t A_t), beta_t = (1 - lam_t) dt_t alpha_t,
    gamma_t = lam_t dt_t:
        h_t = alpha_t h_{t-1} + beta_t sum_r v_{t-1,r} k_{t-1,r}^T + gamma_t sum_r v_{t,r} k_{t,r}^T
        y_{t,R} = h_t q_{t,R} + D v_{t,R}
        out = out_proj(sum_R mimo_o_R * gate(y_{t,R}, z_{t,R}))
    where gate(y, z) = y * silu(z), or rmsnorm_headwise(y) * silu(z) if ``is_outproj_norm``.

    Unrolling the recurrence: v_s enters h_t (s < t) once through gamma_s and once through
    beta_{s+1}, and both terms decay by exp(sum_{u=s+1..t} dt_u A_u). This gives
        y_{t,R} = sum_{s<=t} sum_r W_{t,s,R,r} v_{s,r} + D v_{t,R},
        W_{t,s,R,r} = (q_{t,R} . k_{s,r}) exp(sum_{u=s+1..t} dt_u A_u) c_{t,s},
        c_{t,s} = gamma_s + (1 - lam_{s+1}) dt_{s+1}  for s < t,   c_{t,t} = gamma_t.
    Since Rot is orthogonal, q_t . k_s depends on theta_t - theta_s only. This is the
    data-dependent relative position rotation. Computationally, W is a chunked linear
    attention with keys k_s * (gamma_s + (1 - lam_{s+1}) dt_{s+1}), minus a diagonal correction
    for the look-ahead term at s = t.

    Relevance rules derived from this form:
    * The selection parameters of Mamba-3 are q, k (including their norms, biases and the
      rotation), dt, the data-dependent A, lam and theta. All of them only enter W, so they are
      detached as in the S6 rule, and y is linear in v. The trapezoidal rule spreads each
      token's content over two state updates, but with W fixed this is still linear, so the
      beta/gamma split needs no rule of its own. The look-ahead coefficient (1 - lam_{s+1})
      dt_{s+1} depends only on tokens up to t, so causality holds.
    * Mamba-3 has no short convolution, so v_s depends on token s only and the mixer is a
      direct token-to-token linear map of the content stream.
    * The MIMO up/down projections mimo_x and mimo_o are fixed element-wise scalings (LRP-0).
      The per-rank gate y_R * silu(z_R) uses the half rule, and silu(z_R) the identity rule.
    * The head-wise RMSNorm (``is_outproj_norm``) detaches its normaliser.
    No relevance is absorbed: dt_bias, B_bias and C_bias only enter W, and Mamba-3 layers
    have no other biases, so relevance is exactly conserved.
    """
    nheads, headdim, d_state = mixer.nheads, mixer.headdim, mixer.d_state
    rank, ngroups = mixer.mimo_rank, mixer.num_bc_heads
    if chunk_size is None:
        chunk_size = max(64 // rank, 1)
    zxBCdtAtrap = _linear(u, mixer.in_proj)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtAtrap,
        [
            mixer.d_inner, mixer.d_inner,
            d_state * ngroups * rank, d_state * ngroups * rank,
            nheads, nheads, nheads,
            mixer.num_rope_angles,
        ],
        dim=-1,
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)

    # Mixing weights W (selection parameters): computed exactly as the model does, then fixed.
    with torch.no_grad():
        B = rearrange(B, "b l (r g n) -> b l r g n", r=rank, g=ngroups)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=rank, g=ngroups)
        B = _rms_norm_lrp(B, mixer.B_norm.weight, mixer.B_norm.bias, mixer.B_norm.eps)
        C = _rms_norm_lrp(C, mixer.C_norm.weight, mixer.C_norm.bias, mixer.C_norm.eps)
        B = repeat(B, "b l r g n -> b l r (g j) n", j=nheads // ngroups)
        C = repeat(C, "b l r g n -> b l r (g j) n", j=nheads // ngroups)
        B = B + rearrange(mixer.B_bias.float(), "h r n -> r h n")
        C = C + rearrange(mixer.C_bias.float(), "h r n -> r h n")

        A = torch.clamp(-heavy_tail_activation(dd_A), max=-mixer.A_floor)  # (b, l, h)
        dt = F.softplus(dd_dt + mixer.dt_bias.float())  # (b, l, h)
        lam = torch.sigmoid(trap)  # (b, l, h)
        theta = torch.cumsum(
            torch.tanh(angles)[:, :, None, :].double() * math.pi * dt[..., None].double(), dim=1
        ).remainder(2 * math.pi).float()  # (b, l, h, num_rope_angles)
        pad = d_state // 2 - theta.shape[-1]
        cos = F.pad(torch.cos(theta), (0, pad), value=1.0)[:, :, None]
        sin = F.pad(torch.sin(theta), (0, pad), value=0.0)[:, :, None]
        q = _rotate(C, cos, sin, pairwise=not mixer.is_mimo)
        k = _rotate(B, cos, sin, pairwise=not mixer.is_mimo)

        gamma = lam * dt
        lookahead = F.pad((dt * (1 - lam))[:, 1:], (0, 0, 0, 1))  # (1 - lam_{t+1}) dt_{t+1}
        k_scaled = k * (gamma + lookahead)[:, :, None, :, None]
        # At s = t only gamma_t applies: remove the look-ahead part from the diagonal.
        diag = torch.einsum("blRhn,blrhn->blRrh", q, k) * lookahead[:, :, None, None, :]

    if mixer.is_mimo:
        v = torch.einsum("blhp,hrp->blrhp", x, mixer.mimo_x.float())
        zr = torch.einsum("blhp,hrp->blrhp", z, mixer.mimo_z.float())
    else:
        v, zr = x.unsqueeze(2), z.unsqueeze(2)
    y = _chunked_linear_attention(q, k_scaled, v, A * dt, chunk_size)
    y = y - torch.einsum("blRrh,blrhp->blRhp", diag, v)
    y = y + v * mixer.D.float()[:, None]

    gate = _silu_lrp(zr)
    if mixer.is_outproj_norm:
        y = rearrange(y, "b l r h p -> b l r (h p)")
        y = _rms_norm_lrp(y, mixer.norm.weight, mixer.norm.bias, mixer.norm.eps, group_size=headdim)
        y = rearrange(y, "b l r (h p) -> b l r h p", p=headdim)
    y = _gate_lrp(y, gate)
    if mixer.is_mimo:
        y = torch.einsum("blrhp,hrp->blhp", y, mixer.mimo_o.float())
    else:
        y = y.squeeze(2)
    return _linear(rearrange(y, "b l h p -> b l (h p)"), mixer.out_proj)


_MIXER_LRP = {Mamba: _mamba1_lrp, Mamba2: _mamba2_lrp, Mamba3: _mamba3_lrp}


def _gated_mlp_lrp(mlp: GatedMLP, x):
    y = _linear(x, mlp.fc1)
    y, gate = y.chunk(2, dim=-1)
    y = _gate_lrp(y, _act_lrp(gate, mlp.activation))
    return _linear(y, mlp.fc2)


def _mlp_lrp(mlp, x):
    if isinstance(mlp, GatedMLP):
        return _gated_mlp_lrp(mlp, x)
    raise NotImplementedError(f"MambaLRP: unsupported MLP layer {type(mlp).__name__}")


# ----------------------------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------------------------

def _as_2d_long(ids, device, name):
    if ids is None:
        return None
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError(f"{name} must be 1D (seqlen,) or 2D (batch, seqlen), got shape {tuple(ids.shape)}")
    return ids.to(device=device, dtype=torch.long)


def _resolve_layers(layer, n_layer):
    if layer is None:
        layers = [0]
    elif isinstance(layer, str):
        if layer != "all":
            raise ValueError(f"layer must be an int, a sequence of ints, 'all' or None, got {layer!r}")
        layers = list(range(n_layer + 1))
    elif isinstance(layer, int):
        layers = [layer]
    else:
        layers = list(layer) if len(layer) > 0 else [0]
    resolved = []
    for l in layers:
        l = 0 if l is None else int(l)
        if l < 0:
            l += n_layer + 1
        if not 0 <= l <= n_layer:
            raise ValueError(f"layer index out of range: {l} (valid: 0..{n_layer} or negative)")
        if l not in resolved:
            resolved.append(l)
    return resolved


def _normalize(rel, normalize):
    if normalize is None or normalize is False:
        return rel
    if normalize is True or normalize == "abs_sum":
        denom = rel.abs().sum(dim=-1, keepdim=True)
    elif normalize == "max":
        denom = rel.abs().amax(dim=-1, keepdim=True)
    else:
        raise ValueError(f"normalize must be None, True, False, 'abs_sum' or 'max', got {normalize!r}")
    return rel / denom.clamp_min(torch.finfo(rel.dtype).tiny)


def mamba_lrp(
    model,
    input_ids,
    output_ids=None,
    output_index: Optional[int] = None,
    layer: Optional[Union[int, Sequence[int], str]] = None,
    target_token_id: Optional[Union[int, torch.Tensor]] = None,
    normalize: Optional[Union[bool, str]] = True,
    chunk_size: Optional[int] = None,
) -> MambaLRPAttribution:
    """Explain one output token of a ``MambaLMHeadModel`` with MambaLRP.

    Args:
        model: a ``MambaLMHeadModel`` whose mixers are ``Mamba``, ``Mamba2`` or ``Mamba3``
            layers (e.g. built with ``ssm_cfg={"layer": "Mamba3"}`` or loaded with
            ``from_pretrained``).
        input_ids: prompt tokens, (seqlen,) or (batch, seqlen). A list of ints is accepted.
        output_ids: generated tokens *without* the prompt, (n_out,) or (batch, n_out),
            e.g. ``model.generate(...)[:, input_ids.shape[1]:]``. If None, the model's
            next-token prediction after ``input_ids`` is explained.
        output_index: index into ``output_ids`` of the token to explain; negative values count
            from the end. None means the last output token (or the next-token prediction when
            ``output_ids`` is None).
        layer: where to read relevance. None / 0 = token embeddings (token-level attribution
            of the input, the usual choice). ``k`` = residual stream entering block ``k``;
            ``n_layer`` (or -1) = residual stream entering the final norm. Negative indices
            count from ``n_layer``. A sequence of layers or ``"all"`` computes several at once
            in a single backward pass (see ``MambaLRPAttribution.layer_relevance``).
        target_token_id: explain the logit of this token instead of the one in ``output_ids``
            (e.g. an alternative continuation). None = the actual output token (or the argmax
            prediction when ``output_ids`` is None). An int or a (batch,) tensor.
        normalize: True / "abs_sum" (default) divides relevance by the sum of its absolute
            values over the context, giving signed relative importances whose magnitudes sum
            to 1. "max" divides by the largest magnitude. None / False returns raw relevance,
            which sums (approximately) to the explained logit.
        chunk_size: chunk length of the chunked scans of Mamba2 and Mamba3. Only affects speed
            and memory. None = 64 (64 // mimo_rank for Mamba3 MIMO). Unused by Mamba, whose
            scan is sequential.

    Returns:
        MambaLRPAttribution. ``input_relevance[b, i]`` is the (signed) contribution of input
        token ``i`` to the target logit; positive values support the prediction.
    """
    backbone = getattr(model, "backbone", None)
    lm_head = getattr(model, "lm_head", None)
    if backbone is None or lm_head is None:
        raise TypeError("mamba_lrp expects a MambaLMHeadModel (with .backbone and .lm_head)")
    mixer_lrp = []
    for i, block in enumerate(backbone.layers):
        fn = next((f for cls, f in _MIXER_LRP.items() if isinstance(block.mixer, cls)), None)
        if fn is None:
            raise NotImplementedError(
                f"MambaLRP supports Mamba, Mamba2 and Mamba3 mixers; layer {i} is {type(block.mixer).__name__}"
            )
        mixer_lrp.append(fn)
    n_layer = len(backbone.layers)
    device = backbone.embedding.weight.device

    input_ids = _as_2d_long(input_ids, device, "input_ids")
    output_ids = _as_2d_long(output_ids, device, "output_ids")
    batch, n_in = input_ids.shape
    if n_in == 0:
        raise ValueError("input_ids must contain at least one token")

    if output_ids is None or output_ids.shape[1] == 0:
        if output_index not in (None, 0, -1):
            raise ValueError("output_index must be None when output_ids is not given")
        context = input_ids
        target = None
    else:
        if output_ids.shape[0] != batch:
            raise ValueError("input_ids and output_ids must have the same batch size")
        n_out = output_ids.shape[1]
        idx = n_out - 1 if output_index is None else int(output_index)
        if idx < 0:
            idx += n_out
        if not 0 <= idx < n_out:
            raise IndexError(f"output_index {output_index} out of range for {n_out} output tokens")
        context = torch.cat([input_ids, output_ids[:, :idx]], dim=1)
        target = output_ids[:, idx]
    position = context.shape[1] - 1
    layers = _resolve_layers(layer, n_layer)

    with torch.enable_grad():
        hidden = F.embedding(context, backbone.embedding.weight.float()).detach().requires_grad_(True)
        streams = {0: hidden}
        residual = None
        for i, (block, fn) in enumerate(zip(backbone.layers, mixer_lrp)):
            residual = hidden if residual is None else hidden + residual
            if i > 0:
                streams[i] = residual
            hidden = fn(block.mixer, _norm_lrp(residual, block.norm), chunk_size)
            if block.mlp is not None:
                residual = hidden + residual
                hidden = _mlp_lrp(block.mlp, _norm_lrp(residual, block.norm2))
        residual = hidden if residual is None else hidden + residual
        streams[n_layer] = residual
        final = _norm_lrp(residual[:, -1], backbone.norm_f)
        logits = _linear(final, lm_head)  # (batch, vocab)

        if target_token_id is not None:
            target = torch.as_tensor(target_token_id, device=device, dtype=torch.long).expand(batch)
        elif target is None:
            target = logits.argmax(dim=-1)
        target_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        grads = torch.autograd.grad(target_logit.sum(), [streams[l] for l in layers])

    layer_relevance, raw_total = {}, {}
    for l, g in zip(layers, grads):
        rel = (streams[l] * g).sum(dim=-1).detach()  # (batch, n_context)
        raw_total[l] = rel.sum(dim=-1)
        layer_relevance[l] = _normalize(rel, normalize)
    relevance = layer_relevance[layers[0]]
    return MambaLRPAttribution(
        relevance=relevance,
        input_relevance=relevance[:, :n_in],
        output_relevance=relevance[:, n_in:],
        layer_relevance=layer_relevance,
        raw_total_relevance=raw_total,
        target_logit=target_logit.detach(),
        target_token_ids=target,
        position=position,
        layers=layers,
    )
