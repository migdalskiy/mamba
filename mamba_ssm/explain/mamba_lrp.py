"""MambaLRP: Layer-wise Relevance Propagation for Mamba-2 language models.

Implements the propagation rules of
    "MambaLRP: Explaining Selective State Space Sequence Models"
    Jafari, Montavon, Müller, Eberle (NeurIPS 2024), https://arxiv.org/abs/2406.07592

LRP is realised as Gradient x Input on a *modified* forward pass. The modified forward
computes exactly the same values as the model, but some tensors are detached from the
autograd graph so that the gradient implements the MambaLRP rules:

* SiLU (and other activations): identity rule, ``act(x) = x * [act(x) / x].detach()``.
* Selective SSM: the input-dependent parameters (dt, A, B, C) are detached, so relevance
  flows only through the SSM input ``x`` (and the ``D`` skip connection).
* Multiplicative gates (``y * silu(z)``, gated MLP, ``silu(z0) * x0``): half rule,
  relevance is split equally between the two factors.
* LayerNorm / RMSNorm (block norms, final norm and the gated norm inside Mamba2):
  the normalisation factor (1 / std) is detached.
* Linear layers, depthwise conv1d, embeddings and residual additions: LRP-0 (plain
  Gradient x Input).

With no biases in the model (Mamba2 defaults, except ``conv_bias=True``), relevance is
conserved: at every layer the token relevances sum to the explained logit.

The modified forward is a pure PyTorch re-implementation of ``MambaLMHeadModel`` with
Mamba2 mixers; it does not need the Triton / CUDA kernels and runs on CPU as well.

Example:
    >>> from mamba_ssm.explain import mamba_lrp
    >>> out = model.generate(input_ids, max_length=input_ids.shape[1] + 20)
    >>> output_ids = out[:, input_ids.shape[1]:]
    >>> attr = mamba_lrp(model, input_ids, output_ids, output_index=3)
    >>> attr.input_relevance  # (batch, len(input_ids)) relative importance of each input token
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete


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


def _ssd_lrp(x, dt, A, B, C, chunk_size):
    """Mamba2 SSD scan with dt, A, B, C detached: a linear map of x.

    x: (batch, seqlen, nheads, headdim); dt: (batch, seqlen, nheads); A: (nheads,)
    B, C: (batch, seqlen, ngroups, dstate)
    """
    batch, seqlen, nheads, _ = x.shape
    ngroups = B.shape[2]
    dt, A, B, C = dt.detach(), A.detach(), B.detach(), C.detach()
    B = repeat(B, "b l g n -> b l (g h) n", h=nheads // ngroups)
    C = repeat(C, "b l g n -> b l (g h) n", h=nheads // ngroups)
    chunk_size = min(chunk_size, seqlen)
    pad = (-seqlen) % chunk_size
    X = x * dt.unsqueeze(-1)
    dA = dt * A
    if pad > 0:
        X, B, C = [F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (X, B, C)]
        dA = F.pad(dA, (0, 0, 0, pad))
    y, _ = ssd_minimal_discrete(X, dA, B, C, chunk_size)
    return y[:, :seqlen]


def _mamba2_lrp(mixer: Mamba2, u, chunk_size):
    """Modified Mamba2.forward (training / prefill path) implementing the MambaLRP rules."""
    if mixer.process_group is not None:
        raise NotImplementedError("MambaLRP does not support tensor-parallel Mamba2 layers")
    zxbcdt = _linear(u, mixer.in_proj)
    d_ssm, ngroups, d_state, nheads = mixer.d_ssm, mixer.ngroups, mixer.d_state, mixer.nheads
    d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads], dim=-1
    )
    seqlen = xBC.shape[1]
    conv = mixer.conv1d
    xBC = F.conv1d(
        xBC.transpose(1, 2),
        conv.weight.float(),
        conv.bias.float() if conv.bias is not None else None,
        padding=mixer.d_conv - 1,
        groups=conv.weight.shape[0],
    )[..., :seqlen].transpose(1, 2)
    xBC = _silu_lrp(xBC)
    x, B, C = torch.split(xBC, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1)

    A = -torch.exp(mixer.A_log.float())
    dt = F.softplus(dt + mixer.dt_bias.float())
    if mixer.dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=mixer.dt_limit[0], max=mixer.dt_limit[1])
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
    y = _ssd_lrp(
        x,
        dt,
        A,
        rearrange(B, "b l (g n) -> b l g n", g=ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=ngroups),
        chunk_size,
    )
    D = mixer.D.float()
    D = rearrange(D, "(h p) -> h p", p=mixer.headdim) if mixer.D_has_hdim else D.unsqueeze(-1)
    y = y + x * D
    y = rearrange(y, "b l h p -> b l (h p)")

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
    """Explain one output token of a Mamba2 ``MambaLMHeadModel`` with MambaLRP.

    Args:
        model: a ``MambaLMHeadModel`` whose mixers are ``Mamba2`` layers (e.g. built with
            ``ssm_cfg={"layer": "Mamba2"}`` or loaded with ``from_pretrained``).
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
        chunk_size: chunk length of the pure PyTorch SSD scan. Only affects speed and memory.
            None = 64.

    Returns:
        MambaLRPAttribution. ``input_relevance[b, i]`` is the (signed) contribution of input
        token ``i`` to the target logit; positive values support the prediction.
    """
    backbone = getattr(model, "backbone", None)
    lm_head = getattr(model, "lm_head", None)
    if backbone is None or lm_head is None:
        raise TypeError("mamba_lrp expects a MambaLMHeadModel (with .backbone and .lm_head)")
    for i, block in enumerate(backbone.layers):
        if not isinstance(block.mixer, Mamba2):
            raise NotImplementedError(
                f"MambaLRP is implemented for Mamba2 mixers; layer {i} is {type(block.mixer).__name__}"
            )
    n_layer = len(backbone.layers)
    device = backbone.embedding.weight.device
    chunk_size = 64 if chunk_size is None else chunk_size

    input_ids = _as_2d_long(input_ids, device, "input_ids")
    output_ids = _as_2d_long(output_ids, device, "output_ids")
    batch, n_in = input_ids.shape
    if n_in == 0:
        raise ValueError("input_ids must contain at least one token")

    if output_ids is None or output_ids.shape[1] == 0:
        if output_index not in (None, 0, -1):
            raise ValueError("output_index must be None when output_ids is not given")
        n_prev = 0
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
        n_prev = idx
        context = torch.cat([input_ids, output_ids[:, :idx]], dim=1)
        target = output_ids[:, idx]
    position = context.shape[1] - 1
    layers = _resolve_layers(layer, n_layer)

    with torch.enable_grad():
        hidden = F.embedding(context, backbone.embedding.weight.float()).detach().requires_grad_(True)
        streams = {0: hidden}
        residual = None
        for i, block in enumerate(backbone.layers):
            residual = hidden if residual is None else hidden + residual
            if i > 0:
                streams[i] = residual
            hidden = _mamba2_lrp(block.mixer, _norm_lrp(residual, block.norm), chunk_size)
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
