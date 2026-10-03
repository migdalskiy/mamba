"""Task-level logit distillation from a Transformer teacher (spec §3.8):
loss = (1 - w) * CE + w * tau^2 * KL(p_T^tau || p_S^tau), tau = 2, w = 0.5."""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, tau: float = 2.0, weight: float = 0.5, multilabel: bool = False):
    if multilabel:
        ce = F.binary_cross_entropy_with_logits(student_logits, labels.float())
        soft = F.binary_cross_entropy_with_logits(student_logits / tau, torch.sigmoid(teacher_logits / tau))
        return (1 - weight) * ce + weight * tau ** 2 * soft
    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(F.log_softmax(student_logits / tau, -1), F.log_softmax(teacher_logits / tau, -1),
                  log_target=True, reduction="batchmean")
    return (1 - weight) * ce + weight * tau ** 2 * kl


def hidden_attention_alignment(student_matrix, teacher_attention_row):
    """Optional: align the Mamba-2 SSD matrix row of the pooled token with the teacher's attention
    row (both (B, L), normalised to distributions over positions)."""
    s = student_matrix.abs()
    s = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    t = teacher_attention_row / teacher_attention_row.sum(-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(s.clamp_min(1e-12).log(), t, reduction="batchmean")
