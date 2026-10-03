"""GPU-only kernel parity tests (spec §9). Skipped without CUDA; meant for the nightly GPU job.

* reference (pure PyTorch, fp32) vs fused kernels for Mamba / Mamba-2 / Mamba-3 SISO / MIMO
* decomposed kernel path (mods / recording) vs the module's fused fast path
* varlen-packed vs per-sequence forward through the kernels (issue #758 pattern)
* optional checkpoint smoke (needs Hugging Face access): MAMBACLS_HF_SMOKE=1
"""

import os

import pytest
import torch

from conftest import random_features
from mambacls.data.collate import RightPadCollator, VarlenPackCollator
from mambacls.models.mixers import MixerMods, run_mixer
from mambacls.models.registry import build_backbone, build_classifier

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

GPU_MIXERS = {
    "Mamba1": dict(layer="Mamba1", d_state=16),
    "Mamba2": dict(layer="Mamba2", d_state=64, headdim=32, chunk_size=64),
    "Mamba3": dict(layer="Mamba3", d_state=64, headdim=32, chunk_size=64),
    "Mamba3-mimo": dict(layer="Mamba3", d_state=64, headdim=32, is_mimo=True, mimo_rank=4, chunk_size=16),
}


def _skip_missing(name):
    if name == "Mamba1":
        from mamba_ssm.ops import selective_scan_interface

        if selective_scan_interface.selective_scan_cuda is None:
            pytest.skip("selective_scan_cuda not installed")
    if name == "Mamba3-mimo":
        pytest.importorskip("tilelang")


def _bb(name, impl):
    torch.manual_seed(0)
    return build_backbone({"kind": "mamba", "impl": impl, "config": dict(
        d_model=128, n_layer=2, vocab_size=1000, ssm_cfg=GPU_MIXERS[name], fused_add_norm=False)}, device="cuda")


@cuda
@pytest.mark.parametrize("name", sorted(GPU_MIXERS))
def test_reference_matches_kernels(name):
    _skip_missing(name)
    bb = _bb(name, "kernel")
    u = torch.randn(2, 200, 128, device="cuda")
    for block in bb.model.layers:
        with torch.no_grad():
            fused = block.mixer(u)
            ref = run_mixer(block.mixer, u, "reference")
            decomposed = run_mixer(block.mixer, u, "kernel", record={})
        torch.testing.assert_close(ref, fused, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(decomposed, fused, rtol=1e-3, atol=1e-3)


@cuda
@pytest.mark.parametrize("name", ["Mamba2", "Mamba3"])
def test_offsets_kernel_vs_reference(name):
    _skip_missing(name)
    bb = _bb(name, "kernel")
    m = bb.model.layers[0].mixer
    mods = MixerMods(h_offset=torch.randn(m.nheads, m.headdim, m.d_state, device="cuda") * 0.1)
    u = torch.randn(2, 100, 128, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(run_mixer(m, u, "kernel", mods=mods), run_mixer(m, u, "reference", mods=mods), rtol=1e-3, atol=1e-3)


@cuda
@pytest.mark.parametrize("name", ["Mamba2", "Mamba3", "Mamba3-mimo"])
def test_varlen_kernel_matches_per_sequence(name):
    _skip_missing(name)
    if name == "Mamba2":
        pytest.importorskip("causal_conv1d")
    torch.manual_seed(0)
    model = build_classifier({"backbone": {"kind": "mamba", "impl": "kernel", "config": dict(
        d_model=128, n_layer=2, vocab_size=1000, ssm_cfg=GPU_MIXERS[name], fused_add_norm=False)},
        "pooler": {"name": "mean"}, "adapter": {"name": "probe"}, "data": {"n_classes": 3}}, device="cuda").eval()
    feats = [{**f, "input_ids": [t % 1000 for t in f["input_ids"]]} for f in random_features([50, 120, 33])]
    with torch.no_grad():
        padded = model(RightPadCollator()(feats).to("cuda")).logits
        packed = model(VarlenPackCollator(tokens_per_batch=256)(feats).to("cuda")).logits
    torch.testing.assert_close(packed, padded, rtol=2e-3, atol=2e-3)


@cuda
@pytest.mark.skipif(not os.environ.get("MAMBACLS_HF_SMOKE"), reason="set MAMBACLS_HF_SMOKE=1 (downloads checkpoints)")
def test_checkpoint_smoke():
    import checkpoint_smoke

    for hf_id, layer in checkpoint_smoke.IDS.items():
        r = checkpoint_smoke.check(hf_id, layer, 2048)
        assert r["ok"], r
