"""Transformer baselines (spec §3.9).

* ``HFSequenceClassifier``: ModernBERT / DeBERTa-v3 through ``AutoModelForSequenceClassification``.
* ``HFCausalBackbone``: a causal Transformer (Pythia, trained on the Pile like the Mamba
  checkpoints) behind the same backbone interface, so the same poolers and heads apply.
Both use attention masks; the truncation policy is a data-config field and is logged per run.
"""

from typing import Optional

import torch
import torch.nn as nn

from mambacls.data.collate import Batch
from mambacls.models.backbone import BackboneOutput, lengths_to_mask


class HFSequenceClassifier(nn.Module):
    def __init__(self, hf_id: str, n_classes: int, multilabel: bool = False, device=None, train_backbone: bool = True,
                 dtype=torch.float32):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=n_classes, torch_dtype=dtype,
            problem_type="multi_label_classification" if multilabel else "single_label_classification",
        )
        self.multilabel = multilabel
        self.name = hf_id
        base = getattr(self.model, self.model.base_model_prefix)
        for p in base.parameters():
            p.requires_grad_(train_backbone)
        if device is not None:
            self.to(device)

    def forward(self, batch: Batch):
        from mambacls.models.registry import ClsOutput, classification_loss

        mask = batch.attention_mask
        if mask is None:
            mask = lengths_to_mask(batch.lengths, batch.input_ids.shape[1]).long()
        out = self.model(input_ids=batch.input_ids, attention_mask=mask, output_hidden_states=True)
        loss = None if batch.labels is None else classification_loss(out.logits, batch.labels, self.multilabel)
        return ClsOutput(logits=out.logits, loss=loss, pooled=out.hidden_states[-1][:, 0])

    def trainable_parameter_groups(self):
        head_names = ("classifier", "score", "pooler", "head")
        head, body = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                (head if n.split(".")[0] in head_names else body).append(p)
        return head, body

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HFCausalBackbone(nn.Module):
    """Duck-types ``MambaBackbone`` for causal HF models (right padding + attention mask)."""

    def __init__(self, hf_id: str, device=None, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype)
        self.name = hf_id
        self.mods_providers, self.branch_hooks, self.bidir, self.prefix_embeds, self.recorder = [], [], None, None, None
        if device is not None:
            self.to(device)

    @property
    def d_model(self):
        return self.model.config.hidden_size

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def layer_types(self):
        return ["Attention"] * self.n_layers

    @property
    def embedding(self):
        return self.model.get_input_embeddings()

    def forward(self, input_ids, lengths=None, seq_idx=None, cu_seqlens=None, return_all_layers=False,
                normalize_layers=True, append_embeds=None):
        if cu_seqlens is not None or append_embeds is not None:
            raise NotImplementedError("HF baselines take right-padded batches without appended embeddings")
        B, L = input_ids.shape
        if lengths is None:
            lengths = torch.full((B,), L, device=input_ids.device)
        mask = lengths_to_mask(lengths, L)
        out = self.model(input_ids=input_ids, attention_mask=mask.long(), output_hidden_states=return_all_layers)
        layers = list(out.hidden_states[1:]) if return_all_layers else None
        return BackboneOutput(last_hidden=out.last_hidden_state, layers=layers, lengths=lengths, mask=mask)
