"""SequenceClassifier = backbone + pooler + head + adapter, and the config-driven builder."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig

from mambacls.data.collate import Batch, unpack_to_padded
from mambacls.models.adapters import ADAPTERS, Adapter, CompositeAdapter, ProbeAdapter, build_adapter
from mambacls.models.backbone import MambaBackbone, load_backbone
from mambacls.models.heads import ClassificationHead
from mambacls.models.poolers import Pooler, build_pooler


@dataclass
class ClsOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None
    pool_weights: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class SequenceClassifier(nn.Module):
    def __init__(self, backbone: MambaBackbone, pooler: Pooler, head: ClassificationHead, adapter: Optional[Adapter] = None):
        super().__init__()
        self.backbone, self.pooler, self.head = backbone, pooler, head
        self.adapter = adapter if adapter is not None else ProbeAdapter()
        self.adapter.apply(backbone)

    @property
    def multilabel(self):
        return self.head.multilabel

    def encode(self, batch: Batch):
        """Backbone + pooler: returns (PoolOutput, mask)."""
        append = self.pooler.embedding if getattr(self.pooler, "appends_token", False) else None
        out = self.backbone(
            batch.input_ids,
            lengths=None if batch.packed else batch.lengths,
            seq_idx=batch.seq_idx,
            cu_seqlens=batch.cu_seqlens,
            return_all_layers=self.pooler.needs_all_layers,
            append_embeds=append,
        )
        h, layers, mask = out.last_hidden, out.layers, out.mask
        if batch.packed:  # back to a padded (n_seqs, L, D) view for pooling
            h, lengths = unpack_to_padded(h[0], batch.cu_seqlens, n_seqs=batch.n_seqs)
            layers = None if layers is None else [unpack_to_padded(s[0], batch.cu_seqlens, n_seqs=batch.n_seqs)[0] for s in layers]
            mask = torch.arange(h.shape[1], device=h.device)[None] < lengths[:, None]
        return self.pooler(layers if self.pooler.needs_all_layers else h, mask), mask

    def forward(self, batch: Batch) -> ClsOutput:
        pool, _ = self.encode(batch)
        logits = self.head(pool.pooled.float() if pool.pooled.dtype != torch.float32 else pool.pooled)
        loss = None
        if batch.labels is not None:
            loss = classification_loss(logits, batch.labels, self.multilabel)
        return ClsOutput(logits=logits, loss=loss, pooled=pool.pooled, pool_weights=pool.weights, extra=pool.extra)

    def trainable_parameter_groups(self):
        """(head params, backbone/adapter params) for separate learning rates."""
        head = [p for p in list(self.head.parameters()) + list(self.pooler.parameters()) if p.requires_grad]
        ids = {id(p) for p in head}
        adapter = [p for p in self.adapter.trainable_parameters() if p.requires_grad and id(p) not in ids]
        return head, adapter

    def n_trainable(self) -> int:
        head, adapter = self.trainable_parameter_groups()
        return sum(p.numel() for p in head + adapter)


def classification_loss(logits, labels, multilabel=False):
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels)


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default) if not hasattr(cfg, "get") else cfg.get(key, default)


def _to_dict(cfg):
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


DTYPES = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float16": torch.float16}


def build_backbone(bcfg, device=None):
    bcfg = _to_dict(bcfg)
    kind = bcfg.get("kind", "mamba")
    dtype = DTYPES[bcfg.get("param_dtype", "float32")]  # fp32 master weights; bf16 via autocast
    impl = bcfg.get("impl", "kernel")
    if kind == "mamba":
        kw = dict(impl=impl, grad_checkpointing=bcfg.get("grad_checkpointing", False))
        if bcfg.get("hf_id"):
            return load_backbone(bcfg["hf_id"], allowed_layers=tuple(bcfg.get("allowed_layers", ("Mamba1", "Mamba2", "Mamba3"))),
                                 device=device, dtype=dtype, **kw)
        config = MambaConfig(**bcfg["config"])
        torch.manual_seed(bcfg.get("init_seed", 0))
        return MambaBackbone.from_config(config, device=device, dtype=dtype, name=bcfg.get("name", "random"), **kw)
    if kind == "hf_causal":
        from mambacls.models.baselines import HFCausalBackbone

        return HFCausalBackbone(bcfg["hf_id"], device=device, dtype=dtype)
    raise ValueError(f"unknown backbone kind {kind!r}")


def build_classifier(cfg, n_classes: Optional[int] = None, device=None) -> nn.Module:
    """cfg: {backbone: {...}, pooler: {name, ...}, head: {...}, adapter: {name, ...}, data: {n_classes, multilabel}}."""
    cfg = _to_dict(cfg)
    bcfg = cfg["backbone"]
    data = cfg.get("data", {})
    n_classes = n_classes or data.get("n_classes")
    multilabel = data.get("multilabel", False)
    if bcfg.get("kind") == "hf_seqcls":
        from mambacls.models.baselines import HFSequenceClassifier

        return HFSequenceClassifier(bcfg["hf_id"], n_classes, multilabel=multilabel, device=device,
                                    train_backbone=cfg.get("adapter", {}).get("name", "full") != "probe")
    backbone = build_backbone(bcfg, device=device)
    pcfg = dict(cfg.get("pooler", {"name": "last"}))
    pooler = build_pooler(pcfg.pop("name"), backbone.d_model, backbone.n_layers, **pcfg)
    hcfg = cfg.get("head", {})
    head = ClassificationHead(pooler.out_dim, n_classes, hidden=hcfg.get("hidden"), dropout=hcfg.get("dropout", 0.1),
                              multilabel=multilabel)
    acfg = dict(cfg.get("adapter", {"name": "probe"}))
    adapter = build_adapter(acfg.pop("name"), **acfg)
    model = SequenceClassifier(backbone, pooler, head, adapter)
    return model.to(device) if device is not None else model
