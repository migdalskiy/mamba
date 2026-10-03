"""Backbone wrapper around ``mamba_ssm``'s ``MixerModel`` (no LM head)."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MixerModel

from mambacls.models.mixers import MixerMods, mixer_kind, run_mixer


@dataclass
class BackboneOutput:
    """last_hidden: (B, L, D) after norm_f. layers: per-layer residual streams (B, L, D), the
    stream *after* layer i, i.e. ``hidden_states + residual`` (spec §2.2); with
    ``normalize_layers`` they are passed through ``norm_f``. lengths: (B,) real lengths; mask:
    (B, L) bool. Prefix (prompt) positions are removed, an appended readout token is kept."""

    last_hidden: torch.Tensor
    layers: Optional[List[torch.Tensor]]
    lengths: torch.Tensor
    mask: torch.Tensor


def lengths_to_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def reverse_index(lengths=None, cu_seqlens=None, seqlen=None, device=None):
    """Index that reverses every sequence within its real length.

    Right-padded batch: returns (B, L) with idx[b, t] = len_b - 1 - t for t < len_b and t
    otherwise (padding stays at the end). Varlen-packed: returns (T,) reversing every segment
    of ``cu_seqlens``. The map is an involution, so applying it twice is the identity.
    """
    if cu_seqlens is not None:
        total = int(cu_seqlens[-1]) if seqlen is None else seqlen
        idx = torch.arange(total, device=cu_seqlens.device)
        for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            idx[s:e] = torch.arange(e - 1, s - 1, -1, device=cu_seqlens.device)
        return idx
    t = torch.arange(seqlen, device=device)[None, :]
    lens = lengths[:, None]
    return torch.where(t < lens, lens - 1 - t, t)


def _ref_norm(x, norm):
    x = x.float()
    if isinstance(norm, nn.LayerNorm):
        return F.layer_norm(x, norm.normalized_shape, norm.weight.float(), norm.bias.float() if norm.bias is not None else None, norm.eps)
    out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps) * norm.weight.float()
    return out + norm.bias.float() if getattr(norm, "bias", None) is not None else out


class MambaBackbone(nn.Module):
    """Wraps ``mamba_ssm``'s MixerModel; no LM head.

    impl: "kernel" (fused kernels, GPU) or "reference" (plain PyTorch, any device; also the
        oracle used by the parity tests). Adapters plug in through:
          * ``mods_providers``: callables (layer_idx) -> MixerMods | None (state offsets, LongMamba);
          * ``bidir``: a ``BidirectionalConfig`` (weight-tied backward pass per layer);
          * ``prefix_embeds``: learned prompt embeddings prepended to every sequence;
          * ``branch_hooks``: callables(name) invoked before the forward / backward branch
            (used by untied LoRA adapters).
    """

    def __init__(
        self,
        mixer_model: MixerModel,
        config: Optional[MambaConfig] = None,
        impl: str = "kernel",
        grad_checkpointing: bool = False,
        capture_layers: Optional[List[int]] = None,
        name: str = "custom",
    ):
        super().__init__()
        self.model = mixer_model
        self.config = config
        self.impl = impl
        self.grad_checkpointing = grad_checkpointing
        self.capture_layers = capture_layers
        self.name = name
        self.mods_providers: List[Callable[[int], Optional[MixerMods]]] = []
        self.bidir = None
        self.prefix_embeds: Optional[nn.Parameter] = None
        self.branch_hooks: List[Callable[[str], None]] = []
        self.recorder = None  # set by probe.capture.InternalsRecorder

    # -- construction ----------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: MambaConfig, device=None, dtype=None, **kw):
        lm = MambaLMHeadModel(config, device=device, dtype=dtype)
        return cls(lm.backbone, config=config, **kw)

    @classmethod
    def from_pretrained(cls, hf_id: str, device=None, dtype=None, **kw):
        lm = MambaLMHeadModel.from_pretrained(hf_id, device=device, dtype=dtype)
        backbone = lm.backbone
        del lm.lm_head
        return cls(backbone, config=lm.config, name=hf_id, **kw)

    # -- properties ------------------------------------------------------------------------
    @property
    def d_model(self) -> int:
        return self.model.embedding.weight.shape[1]

    @property
    def n_layers(self) -> int:
        return len(self.model.layers)

    @property
    def layer_types(self) -> List[str]:
        return [mixer_kind(b.mixer) for b in self.model.layers]

    @property
    def embedding(self) -> nn.Embedding:
        return self.model.embedding

    # -- forward ---------------------------------------------------------------------------
    def _norm(self, x, norm):
        if self.impl == "reference":
            return _ref_norm(x, norm)
        return norm(x.to(dtype=norm.weight.dtype))

    def _mods(self, i):
        mods = MixerMods()
        for provider in self.mods_providers:
            m = provider(i)
            if m is None:
                continue
            for k in ("y_offset", "h_offset", "h0", "dt_filter"):
                if getattr(m, k) is not None:
                    if getattr(mods, k) is not None:
                        raise ValueError(f"two adapters set {k} on layer {i}")
                    setattr(mods, k, getattr(m, k))
        return mods

    def _mixer(self, i, mixer, h, lengths, seq_kwargs):
        mods = self._mods(i)
        record = self.recorder.slot(i) if self.recorder is not None else None

        def branch(u, rec):
            return run_mixer(mixer, u, impl=self.impl, seq_kwargs=seq_kwargs, mods=mods, record=rec)

        y = branch(h, record)
        if self.bidir is not None and self.bidir.applies(i):
            if seq_kwargs.get("cu_seqlens") is not None:
                idx = reverse_index(cu_seqlens=seq_kwargs["cu_seqlens"], seqlen=h.shape[1])
                rev = lambda t: t[:, idx]  # noqa: E731
            else:
                idx = reverse_index(lengths=lengths, seqlen=h.shape[1])[..., None].expand_as(h)
                rev = lambda t: torch.gather(t, 1, idx)  # noqa: E731
            for hook in self.branch_hooks:
                hook("bwd")
            try:
                y_bwd = rev(branch(rev(h), None))
            finally:
                for hook in self.branch_hooks:
                    hook("fwd")
            y = y + self.bidir.gate(i).to(y.dtype) * y_bwd
        return y

    def _block(self, i, block, hidden, residual, lengths, seq_kwargs):
        residual = hidden if residual is None else hidden + residual
        h = self._norm(residual, block.norm)
        if block.residual_in_fp32:
            residual = residual.float()
        h = self._mixer(i, block.mixer, h, lengths, seq_kwargs)
        if block.mlp is not None:
            residual = h + residual
            h = block.mlp(self._norm(residual, block.norm2).to(h.dtype))
            if block.residual_in_fp32:
                residual = residual.float()
        return h, residual

    def forward(
        self,
        input_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
        seq_idx: Optional[torch.IntTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        return_all_layers: bool = False,
        normalize_layers: bool = True,
        append_embeds: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """input_ids: (B, L) right-padded, with ``lengths``; or (1, T) varlen-packed with
        ``seq_idx`` (1, T) and ``cu_seqlens`` (n + 1,). append_embeds: (D,) a learned readout
        embedding inserted after the last real token of every sequence (``eos_cls`` pooler)."""
        varlen = cu_seqlens is not None
        x = self.model.embedding(input_ids)
        B, L = input_ids.shape
        if lengths is None:
            lengths = torch.full((B,), L, device=input_ids.device, dtype=torch.long)
        if varlen and (append_embeds is not None or self.prefix_embeds is not None):
            raise NotImplementedError("prompt tuning / appended readout tokens need right-padded batches")
        if varlen and self.impl == "reference":
            return self._varlen_via_padding(input_ids, cu_seqlens, return_all_layers, normalize_layers)

        if append_embeds is not None:
            x = F.pad(x, (0, 0, 0, 1))
            x = x.scatter(1, lengths.view(B, 1, 1).expand(B, 1, x.shape[-1]),
                          append_embeds.to(x.dtype).view(1, 1, -1).expand(B, 1, -1))
            lengths = lengths + 1
        n_prefix = 0
        if self.prefix_embeds is not None:
            n_prefix = self.prefix_embeds.shape[0]
            x = torch.cat([self.prefix_embeds.to(x.dtype)[None].expand(B, -1, -1), x], dim=1)
            lengths = lengths + n_prefix

        seq_kwargs = {"seq_idx": seq_idx, "cu_seqlens": cu_seqlens} if varlen else {}
        hidden, residual = x, None
        streams = []
        for i, block in enumerate(self.model.layers):
            if self.grad_checkpointing and self.training:
                hidden, residual = checkpoint(
                    self._block, i, block, hidden, residual, lengths, seq_kwargs, use_reentrant=False
                )
            else:
                hidden, residual = self._block(i, block, hidden, residual, lengths, seq_kwargs)
            if return_all_layers:
                streams.append(hidden + residual)
        residual = hidden + residual
        last = self._norm(residual, self.model.norm_f)
        layers = None
        if return_all_layers:
            layers = [self._norm(s, self.model.norm_f) if normalize_layers else s for s in streams]
        if n_prefix:
            last = last[:, n_prefix:]
            layers = None if layers is None else [s[:, n_prefix:] for s in layers]
            lengths = lengths - n_prefix
        mask = lengths_to_mask(lengths, last.shape[1])
        return BackboneOutput(last_hidden=last, layers=layers, lengths=lengths, mask=mask)

    def _varlen_via_padding(self, input_ids, cu_seqlens, return_all_layers, normalize_layers):
        """Reference path for packed input: unpack, run padded, re-pack."""
        from mambacls.data.collate import pack_padded, unpack_to_padded

        ids, lengths = unpack_to_padded(input_ids[0], cu_seqlens, pad_value=0)
        out = self.forward(ids, lengths, return_all_layers=return_all_layers, normalize_layers=normalize_layers)
        total = input_ids.shape[1]
        last = pack_padded(out.last_hidden, out.lengths, total)[None]
        layers = None if out.layers is None else [pack_padded(s, out.lengths, total)[None] for s in out.layers]
        mask = torch.zeros(1, total, dtype=torch.bool, device=input_ids.device)
        mask[0, : int(cu_seqlens[-1])] = True
        return BackboneOutput(last_hidden=last, layers=layers, lengths=out.lengths, mask=mask)


def load_backbone(hf_id: str, allowed_layers=("Mamba2", "Mamba3"), **kw) -> MambaBackbone:
    """MambaLMHeadModel.from_pretrained(hf_id) -> .backbone; the LM head is dropped. Asserts that
    the config's ssm layer is one of ``allowed_layers`` (the Mamba-1 control arm passes
    ``allowed_layers=("Mamba1",)``)."""
    backbone = MambaBackbone.from_pretrained(hf_id, **kw)
    layer = (backbone.config.ssm_cfg or {}).get("layer", "Mamba1")
    if layer not in allowed_layers:
        raise ValueError(f"{hf_id}: ssm_cfg.layer={layer!r}, expected one of {allowed_layers}")
    return backbone
