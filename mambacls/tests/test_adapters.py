import pytest
import torch
import torch.nn.functional as F

from conftest import VOCAB, backbone_cfg, make_classifier, random_features
from mambacls.data.collate import RightPadCollator
from mambacls.models.adapters import ADAPTERS, LongMambaFilter, build_adapter
from mambacls.models.mixers import MixerMods, run_mixer
from mambacls.models.registry import build_backbone


def _batch():
    return RightPadCollator()(random_features([6, 4, 5]))


def _perturb_lora(adapter, scale=0.05):
    with torch.no_grad():
        for p in adapter.loras.values():
            for B in p.B.values():
                B.normal_(0, scale)


@pytest.mark.parametrize("mixer", ["Mamba1", "Mamba2", "Mamba3-mimo"])
def test_lora_merge_unmerge_round_trip(mixer):
    model = make_classifier(mixer, pooler="mean", adapter={"name": "lora", "r": 4, "dropout": 0.0}).eval()
    lora = model.adapter
    _perturb_lora(lora)
    batch = _batch()
    linear = next(m for n, m in model.backbone.named_modules() if n.endswith("mixer.out_proj"))
    W0 = linear.parametrizations.weight.original.detach().clone()
    before = model(batch).logits
    lora.merge()
    torch.testing.assert_close(model(batch).logits, before, rtol=1e-5, atol=1e-5)  # same function, merged
    assert not torch.allclose(linear.parametrizations.weight.original, W0)
    lora.unmerge()
    torch.testing.assert_close(linear.parametrizations.weight.original, W0, rtol=0, atol=1e-6)
    torch.testing.assert_close(model(batch).logits, before, rtol=1e-5, atol=1e-5)


def test_lora_is_visible_through_weight_attribute():
    """Fused kernels read ``.weight`` directly; the parametrization makes LoRA visible there."""
    model = make_classifier("Mamba2", adapter={"name": "lora", "r": 2, "dropout": 0.0})
    _perturb_lora(model.adapter)
    linear = model.backbone.model.layers[0].mixer.in_proj
    p = next(v for k, v in model.adapter.loras.items() if k.startswith("0__in_proj"))
    expected = linear.parametrizations.weight.original + p.delta(use_dropout=False)
    torch.testing.assert_close(linear.weight, expected)


def test_lora_only_trains_factors_and_head():
    model = make_classifier("Mamba2", adapter={"name": "lora", "r": 2})
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all(("parametrizations" in n and (".A." in n or ".B." in n)) or n.startswith(("head", "pooler", "adapter")) for n in names)
    assert not any(n.endswith(("A_log", "dt_bias", "conv1d.weight")) for n in names)


@pytest.mark.parametrize("variant", ["y", "h", "h0"])
@pytest.mark.parametrize("mixer", ["Mamba1", "Mamba2", "Mamba3"])
def test_state_offset_zero_init_and_effect(mixer, variant):
    if mixer == "Mamba1" and variant == "h0":
        with pytest.raises(NotImplementedError):
            make_classifier(mixer, adapter={"name": "state_offset", "variant": variant})
        return
    base = make_classifier(mixer, pooler="mean").eval()
    tuned = make_classifier(mixer, pooler="mean", adapter={"name": "state_offset", "variant": variant}).eval()
    tuned.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    torch.testing.assert_close(tuned(batch).logits, base(batch).logits, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        for p in tuned.adapter.offsets.values():
            p.normal_(0, 0.5)
    assert not torch.allclose(tuned(batch).logits, base(batch).logits)
    tuned.train()
    tuned(batch).loss.backward()
    assert all(p.grad is not None for p in tuned.adapter.offsets.values())


def test_h_offset_is_c_projection_mamba2():
    """h variant: y_t += C_t^T h'. Adding C_t^T h' to the recorded pre-gate output by hand and
    finishing the mixer (gate, out_proj) must give the same result as the h-offset mod."""
    from einops import rearrange

    from mambacls.models import mixers

    bb = build_backbone(backbone_cfg("Mamba2"))
    m = bb.model.layers[0].mixer
    u = torch.randn(1, 5, 32)
    h_off = torch.randn(m.nheads, m.headdim, m.d_state)
    rec = {}
    run_mixer(m, u, "reference", record=rec)
    y_off = run_mixer(m, u, "reference", mods=MixerMods(h_offset=h_off))
    C = rec["C"].repeat_interleave(m.nheads // m.ngroups, dim=2)
    y = rearrange(rec["y"] + torch.einsum("blhn,hpn->blhp", C, h_off), "b l h p -> b l (h p)")
    y = mixers._gated_rmsnorm(y, rec["z"], m.norm) if m.rmsnorm else y * F.silu(rec["z"])
    torch.testing.assert_close(mixers._float_linear(y, m.out_proj), y_off, rtol=1e-5, atol=1e-5)


def test_sdlora_selects_top_heads_and_freezes_rest():
    model = make_classifier("Mamba2", adapter={"name": "sdlora", "r": 2, "top_frac": 0.5, "warmup_epochs": 1})
    sdt = model.adapter
    m = model.backbone.model.layers[0].mixer
    assert m.A_log.requires_grad and m.dt_bias.requires_grad
    with torch.no_grad():  # simulate warm-up updates: head 1 and 3 moved most
        m.A_log[1] += 1.0
        m.dt_bias[3] += 0.5
        m.D[0] += 0.01
    sdt.on_epoch_end(0)
    keep = sdt.selection[0]
    assert keep.sum().item() == 2 and keep[1] and keep[3]
    torch.testing.assert_close(m.D[0], sdt._init["0.D"][0])  # unselected head reset
    # gradients of frozen heads are masked, and after_optimizer_step restores frozen entries
    model(_batch()).loss.backward()
    assert (m.A_log.grad[~keep] == 0).all()
    with torch.no_grad():
        m.A_log += 1.0
    sdt.after_optimizer_step()
    torch.testing.assert_close(m.A_log[~keep], sdt._init["0.A_log"][~keep])


def test_build_adapter_filters_unknown_kwargs_and_composes():
    a = build_adapter("probe", r=16, alpha=32, targets=["in_proj"])
    assert a.name == "probe"
    c = build_adapter("bidir+lora", bidir={"mode": "tied_gate"}, lora={"r": 4})
    assert c.name == "bidir_tied_gate+lora"
    with pytest.raises(KeyError):
        build_adapter("nope")


def test_longmamba_filter_calibrates_and_filters_only_long_inputs():
    model = make_classifier("Mamba2", pooler="mean").eval()
    lm = LongMambaFilter(train_len=6, global_threshold=0.0)  # every head counts as global
    lm.attach(model.backbone)
    calib = [RightPadCollator()(random_features([6, 6]))]
    budget, is_global = lm.calibrate(lambda b: model(b), calib)
    assert budget.shape == (2, model.backbone.model.layers[0].mixer.nheads) and is_global.all()
    short = RightPadCollator()(random_features([5]))
    lm.enabled = False
    ref_short = model(short).logits
    lm.enabled = True
    torch.testing.assert_close(model(short).logits, ref_short)  # no-op up to L_train
    long = RightPadCollator()(random_features([40]))
    lm.enabled = False
    ref_long = model(long).logits
    lm.enabled = True
    assert not torch.allclose(model(long).logits, ref_long)
    dt = torch.rand(1, 40, budget.shape[1])
    mask = lm.mods(0).dt_filter(dt, -torch.ones(budget.shape[1]))
    kept = (dt * mask).sum(1)
    assert (kept <= budget[0] + dt.max()).all() and (mask.sum(1) >= 1).all()


def test_full_finetune_keeps_ssm_params_fp32():
    model = make_classifier("Mamba2", adapter={"name": "full"})
    m = model.backbone.model.layers[0].mixer
    assert m.A_log.dtype == torch.float32 and m.A_log.requires_grad and model.backbone.embedding.weight.requires_grad
