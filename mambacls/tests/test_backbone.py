import pytest
import torch

from conftest import MIXERS, VOCAB, backbone_cfg, make_classifier, random_features
from mamba_ssm.explain.mamba_lrp import _MIXER_LRP
from mambacls.data.collate import RightPadCollator, VarlenPackCollator
from mambacls.models.adapters import BidirAdapter, PromptAdapter
from mambacls.models.backbone import reverse_index
from mambacls.models.mixers import MixerMods, run_mixer
from mambacls.models.registry import build_backbone


@pytest.mark.parametrize("mixer", sorted(MIXERS))
def test_reference_mixer_matches_validated_forward(mixer):
    """The reference mixers compute the same function as the MambaLRP modified forward, which is
    validated against independent step-by-step recurrences and the repo's kernel references."""
    bb = build_backbone(backbone_cfg(mixer))
    u = torch.randn(2, 13, 32)
    for block in bb.model.layers:
        ours = run_mixer(block.mixer, u, impl="reference")
        lrp = next(f for cls, f in _MIXER_LRP.items() if isinstance(block.mixer, cls))(block.mixer, u, 4)
        torch.testing.assert_close(ours, lrp, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mixer", ["Mamba1", "Mamba2", "Mamba3", "Mamba3-mimo"])
def test_reference_path_trains_ssm_parameters(mixer):
    """Unlike the LRP forward, the reference path is fully differentiable (dt / A / B / C get grads)."""
    bb = build_backbone(backbone_cfg(mixer))
    out = bb(torch.randint(0, VOCAB, (2, 9)))
    out.last_hidden.sum().backward()
    m = bb.model.layers[0].mixer
    names = {"Mamba1": ["A_log", "dt_proj.bias"], "Mamba2": ["A_log", "dt_bias"], "Mamba3": ["dt_bias", "B_bias", "C_bias"]}
    for n in names["Mamba1" if mixer == "Mamba1" else "Mamba2" if mixer.startswith("Mamba2") else "Mamba3"]:
        g = m.get_parameter(n).grad
        assert g is not None and g.abs().sum() > 0, n


def test_reverse_index_right_pad_and_varlen():
    idx = reverse_index(lengths=torch.tensor([3, 5]), seqlen=5)
    assert idx.tolist() == [[2, 1, 0, 3, 4], [4, 3, 2, 1, 0]]
    v = reverse_index(cu_seqlens=torch.tensor([0, 2, 5, 6]))
    assert v.tolist() == [1, 0, 4, 3, 2, 5]
    assert torch.equal(idx.gather(1, idx), torch.arange(5).expand(2, 5))  # involution


@pytest.mark.parametrize("mixer", ["Mamba2", "Mamba3"])
def test_bidir_reverse_respects_lengths(mixer):
    """The backward branch of sequence b must not see padding: padding a batch changes nothing."""
    model = make_classifier(mixer, pooler="mean", adapter={"name": "bidir", "mode": "tied_add"}).eval()
    feats = random_features([7, 3, 5])
    a = model(RightPadCollator()(feats)).logits
    b = model(RightPadCollator(pad_to_multiple=16)(feats)).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    single = torch.cat([model(RightPadCollator()([f])).logits for f in feats])
    torch.testing.assert_close(a, single, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mixer", ["Mamba1", "Mamba2", "Mamba3-mimo"])
def test_tied_gate_init_reproduces_causal(mixer):
    causal = make_classifier(mixer, pooler="mean").eval()
    bidir = make_classifier(mixer, pooler="mean", adapter={"name": "bidir", "mode": "tied_gate"}).eval()
    bidir.load_state_dict(causal.state_dict(), strict=False)
    batch = RightPadCollator()(random_features([6, 4]))
    torch.testing.assert_close(bidir(batch).logits, causal(batch).logits, rtol=1e-5, atol=1e-5)
    added = make_classifier(mixer, pooler="mean", adapter={"name": "bidir", "mode": "tied_add"}).eval()
    added.load_state_dict(causal.state_dict(), strict=False)
    assert not torch.allclose(added(batch).logits, causal(batch).logits)  # the backward branch matters


def test_untied_lora_backward_branch_has_own_factors():
    m = make_classifier("Mamba2", pooler="mean", adapter={"name": "bidir", "mode": "untied_lora", "lora_kwargs": {"r": 2}})
    loras = m.adapter.lora.loras
    p = next(iter(loras.values()))
    assert set(p.A.keys()) == {"fwd", "bwd"}
    with torch.no_grad():
        for g in m.adapter.gates.values():
            g.fill_(1.0)
    out = m(RightPadCollator()(random_features([5, 4])))
    out.loss.backward()
    assert p.B["bwd"].grad is not None and p.B["fwd"].grad is not None
    assert p.active == "fwd"


@pytest.mark.parametrize("mixer", ["Mamba2", "Mamba3"])
def test_varlen_packed_matches_per_sequence(mixer):
    model = make_classifier(mixer, pooler="mean").eval()
    feats = random_features([5, 8, 3])
    padded = model(RightPadCollator()(feats)).logits
    packed_batch = VarlenPackCollator(tokens_per_batch=24)(feats)
    assert packed_batch.seq_idx.shape == (1, 24) and int(packed_batch.cu_seqlens[-1]) == 24
    packed = model(packed_batch).logits
    torch.testing.assert_close(packed, padded, rtol=1e-4, atol=1e-5)


def test_prompt_prefix_is_stripped():
    model = make_classifier("Mamba2", pooler="mean", adapter={"name": "prompt", "n_tokens": 4})
    batch = RightPadCollator()(random_features([5, 3]))
    out = model.backbone(batch.input_ids, batch.lengths)
    assert out.last_hidden.shape[1] == 5 and out.lengths.tolist() == [5, 3]
    model.zero_grad()
    model(batch).loss.backward()
    assert model.adapter.prefix.grad.abs().sum() > 0


def test_return_all_layers_is_residual_stream():
    bb = build_backbone(backbone_cfg("Mamba2", n_layer=3))
    out = bb(torch.randint(0, VOCAB, (2, 6)), return_all_layers=True, normalize_layers=False)
    assert len(out.layers) == 3
    torch.testing.assert_close(bb._norm(out.layers[-1], bb.model.norm_f), out.last_hidden)


def test_zero_mods_equal_plain_reference():
    bb = build_backbone(backbone_cfg("Mamba2"))
    u = torch.randn(2, 7, 32)
    m = bb.model.layers[0].mixer
    zero = MixerMods(y_offset=torch.zeros(m.nheads, m.headdim), h_offset=torch.zeros(m.nheads, m.headdim, m.d_state),
                     h0=torch.zeros(m.nheads, m.headdim, m.d_state))
    torch.testing.assert_close(run_mixer(m, u, "reference", mods=zero), run_mixer(m, u, "reference"))
