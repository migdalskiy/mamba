import pytest
import torch

from conftest import make_classifier, random_features
from mambacls.data.collate import RightPadCollator
from mambacls.models.poolers import POOLERS, build_pooler


@pytest.mark.parametrize("name", sorted(POOLERS))
def test_pad_invariance_pooler_only(name):
    """Appending pad positions (any values) leaves every pooler's output unchanged."""
    torch.manual_seed(0)
    B, L, D, n_layers = 3, 7, 16, 2
    pooler = build_pooler(name, D, n_layers).eval()
    lengths = torch.tensor([7, 4, 1])
    mask = torch.arange(L)[None] < lengths[:, None]
    h = [torch.randn(B, L, D) for _ in range(n_layers)]
    h_pad = [torch.cat([x, 100 * torch.randn(B, 5, D)], dim=1) for x in h]
    mask_pad = torch.cat([mask, torch.zeros(B, 5, dtype=torch.bool)], dim=1)
    arg, arg_pad = (h, h_pad) if pooler.needs_all_layers else (h[-1], h_pad[-1])
    a = pooler(arg, mask)
    b = pooler(arg_pad, mask_pad)
    torch.testing.assert_close(a.pooled, b.pooled, rtol=1e-5, atol=1e-5)
    if a.weights is not None:
        torch.testing.assert_close(b.weights[:, :L], a.weights, rtol=1e-5, atol=1e-6)
        assert b.weights[:, L:].abs().max() < 1e-6


@pytest.mark.parametrize("name", sorted(POOLERS))
def test_pad_invariance_end_to_end(name):
    """Right padding cannot change any real position of a causal stack, so a whole classifier is
    invariant to extra padding in the batch."""
    model = make_classifier("Mamba2", pooler=name).eval()
    feats = random_features([6, 9, 3])
    a = model(RightPadCollator()(feats)).logits
    b = model(RightPadCollator(pad_to_multiple=16)(feats)).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_last_equals_manual_index():
    pooler = build_pooler("last", 8, 1)
    h = torch.randn(3, 6, 8)
    lengths = torch.tensor([6, 2, 4])
    mask = torch.arange(6)[None] < lengths[:, None]
    out = pooler(h, mask)
    manual = torch.stack([h[0, 5], h[1, 1], h[2, 3]])
    torch.testing.assert_close(out.pooled, manual)


def test_mean_and_max_are_masked():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert build_pooler("mean", 1, 1)(h, mask).pooled.item() == 2.0
    assert build_pooler("max", 1, 1)(h, mask).pooled.item() == 3.0


def test_attention_weights_sum_to_one():
    p = build_pooler("attn", 8, 1)
    mask = torch.tensor([[True] * 5 + [False] * 2])
    w = p(torch.randn(1, 7, 8), mask).weights
    assert torch.allclose(w.sum(), torch.tensor(1.0)) and w[0, 5:].abs().max() == 0


def test_scalar_mix_reports_layer_weights_and_needs_layers():
    p = build_pooler("scalar_mix", 8, 3)
    out = p([torch.randn(2, 4, 8) for _ in range(3)], torch.ones(2, 4, dtype=torch.bool))
    assert out.extra["layer_weights"].shape == (3,)
    with pytest.raises(ValueError):
        p(torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool))


def test_eos_cls_learned_reads_appended_position():
    model = make_classifier("Mamba2", pooler="eos_cls").eval()
    batch = RightPadCollator()(random_features([5, 3]))
    pool, mask = model.encode(batch)
    assert mask.sum(1).tolist() == [6, 4]  # one appended readout token per sequence
    assert pool.weights.argmax(1).tolist() == [5, 3]
