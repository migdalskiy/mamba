import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import mamba_ssm.modules.mamba3 as mamba3_module
from mamba_ssm.explain import mamba_lrp
from mamba_ssm.explain.mamba_lrp import _SelectiveScanLRP
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

VOCAB = 101

DEFAULT_SSM_CFG = {
    "Mamba1": dict(d_state=8),
    "Mamba2": dict(d_state=16, headdim=16, chunk_size=8),
    "Mamba3": dict(d_state=16, headdim=16, chunk_size=8),
}


@pytest.fixture(autouse=True)
def _allow_mimo_without_tilelang(monkeypatch):
    # Mamba3(is_mimo=True) asserts that the TileLang kernel imported. mamba_lrp does not use it,
    # so let the layer be constructed on machines without TileLang.
    if mamba3_module.mamba3_mimo_combined is None:
        monkeypatch.setattr(mamba3_module, "mamba3_mimo_combined", object())


def _make_model(layer="Mamba2", device="cpu", d_intermediate=0, rms_norm=True, seed=0, **ssm_cfg):
    torch.manual_seed(seed)
    cfg = dict(layer=layer, **DEFAULT_SSM_CFG[layer])
    cfg.update(ssm_cfg)
    config = MambaConfig(
        d_model=64,
        n_layer=3,
        d_intermediate=d_intermediate,
        vocab_size=VOCAB,
        ssm_cfg=cfg,
        rms_norm=rms_norm,
        residual_in_fp32=True,
        fused_add_norm=False,
    )
    model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
    # Randomise parameters that are initialised to constants so the test exercises them.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm2.weight") or name.endswith("norm_f.weight"):
                p.uniform_(0.5, 1.5)
            elif name.endswith((".D", ".B_bias", ".C_bias")):
                p.uniform_(-1.0, 1.0)
            elif ".mimo_" in name:
                p.uniform_(0.2, 1.0)
            elif name.endswith("conv1d.bias") or (name.endswith(".bias") and "norm" in name):
                p.uniform_(-0.1, 0.1)
    return model.eval()


# ----------------------------------------------------------------------------------------------
# Independent references (plain PyTorch, written from the recurrences, fully differentiable)
# ----------------------------------------------------------------------------------------------

def _ref_norm(x, norm):
    if isinstance(norm, nn.LayerNorm):
        return F.layer_norm(x, norm.weight.shape, norm.weight, norm.bias, norm.eps)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps) * norm.weight


def _ref_mamba1(m, u):
    seqlen = u.shape[1]
    x, z = m.in_proj(u).chunk(2, dim=-1)
    x = F.silu(m.conv1d(x.transpose(1, 2))[..., :seqlen])  # (b d l)
    dt, B, C = torch.split(m.x_proj(x.transpose(1, 2)), [m.dt_rank, m.d_state, m.d_state], dim=-1)
    y = selective_scan_ref(
        x,
        (dt @ m.dt_proj.weight.t()).transpose(1, 2),
        -torch.exp(m.A_log),
        B.transpose(1, 2),
        C.transpose(1, 2),
        m.D,
        z=z.transpose(1, 2),
        delta_bias=m.dt_proj.bias,
        delta_softplus=True,
    )
    return m.out_proj(y.transpose(1, 2))


def _ref_mamba2(m, u):
    """Sequential-recurrence reference for Mamba2.forward."""
    batch, seqlen, _ = u.shape
    zxbcdt = m.in_proj(u)
    d_mlp = (zxbcdt.shape[-1] - 2 * m.d_ssm - 2 * m.ngroups * m.d_state - m.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1
    )
    xBC = F.silu(m.conv1d(xBC.transpose(1, 2))[..., :seqlen].transpose(1, 2))
    x, B, C = torch.split(xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=m.ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=m.ngroups)
    dt = F.softplus(dt + m.dt_bias)
    A = -torch.exp(m.A_log)
    D = m.D.view(m.nheads, m.headdim) if m.D_has_hdim else m.D.view(m.nheads, 1)
    heads_per_group = m.nheads // m.ngroups
    state = torch.zeros(batch, m.nheads, m.headdim, m.d_state)
    ys = []
    for t in range(seqlen):
        Bt = B[:, t].repeat_interleave(heads_per_group, dim=1)  # (b h n)
        Ct = C[:, t].repeat_interleave(heads_per_group, dim=1)
        dA = torch.exp(dt[:, t] * A)  # (b h)
        state = state * dA[..., None, None] + torch.einsum("bh,bhp,bhn->bhpn", dt[:, t], x[:, t], Bt)
        ys.append(torch.einsum("bhpn,bhn->bhp", state, Ct) + D * x[:, t])
    y = rearrange(torch.stack(ys, dim=1), "b l h p -> b l (h p)")
    if m.rmsnorm:
        gs = m.norm.group_size
        g = F.silu(z)
        if not m.norm.norm_before_gate:
            y = y * g
        yg = rearrange(y, "... (n d) -> ... n d", d=gs)
        y = rearrange(yg * torch.rsqrt(yg.square().mean(-1, keepdim=True) + m.norm.eps), "... n d -> ... (n d)")
        y = y * m.norm.weight
        if m.norm.norm_before_gate:
            y = y * g
    else:
        y = y * F.silu(z)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    return m.out_proj(y)


def _ref_rotate(x, theta, pairwise):
    """Rotate the first theta.shape[-1] pairs: (2i, 2i+1) if pairwise, else (i, i + n/2)."""
    out = x.clone()
    half = x.shape[-1] // 2
    for i in range(theta.shape[-1]):
        a, b = (2 * i, 2 * i + 1) if pairwise else (i, i + half)
        cos, sin = torch.cos(theta[..., i]), torch.sin(theta[..., i])
        out[..., a] = x[..., a] * cos - x[..., b] * sin
        out[..., b] = x[..., a] * sin + x[..., b] * cos
    return out


def _ref_mamba3(m, u):
    """Step-by-step trapezoidal recurrence of Mamba3 (SISO and MIMO)."""
    batch, seqlen, _ = u.shape
    R, H, P, N, G = m.mimo_rank, m.nheads, m.headdim, m.d_state, m.num_bc_heads
    z, x, B, C, dd_dt, dd_A, trap, phi = torch.split(
        m.in_proj(u), [m.d_inner, m.d_inner, N * G * R, N * G * R, H, H, H, m.num_rope_angles], dim=-1
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=P)
    x = rearrange(x, "b l (h p) -> b l h p", p=P)
    B = _ref_norm(rearrange(B, "b l (r g n) -> b l r g n", r=R, g=G), m.B_norm)
    C = _ref_norm(rearrange(C, "b l (r g n) -> b l r g n", r=R, g=G), m.C_norm)
    B = B.repeat_interleave(H // G, dim=3) + m.B_bias.transpose(0, 1)  # (b l r h n)
    C = C.repeat_interleave(H // G, dim=3) + m.C_bias.transpose(0, 1)
    A = torch.clamp(-heavy_tail_activation(dd_A), max=-m.A_floor)
    dt = F.softplus(dd_dt + m.dt_bias)
    lam = torch.sigmoid(trap)
    if m.is_mimo:
        v = torch.einsum("blhp,hrp->blrhp", x, m.mimo_x)
        zr = torch.einsum("blhp,hrp->blrhp", z, m.mimo_z)
    else:
        v, zr = x.unsqueeze(2), z.unsqueeze(2)

    theta = torch.zeros(batch, H, m.num_rope_angles)
    state = torch.zeros(batch, H, P, N)
    k_prev, v_prev = torch.zeros(batch, R, H, N), torch.zeros(batch, R, H, P)
    ys = []
    for t in range(seqlen):
        theta = theta + torch.tanh(phi[:, t, None, :]) * math.pi * dt[:, t, :, None]
        q = _ref_rotate(C[:, t], theta[:, None], pairwise=not m.is_mimo)
        k = _ref_rotate(B[:, t], theta[:, None], pairwise=not m.is_mimo)
        alpha = torch.exp(A[:, t] * dt[:, t])[..., None, None]
        beta = ((1 - lam[:, t]) * dt[:, t])[..., None, None] * alpha
        gamma = (lam[:, t] * dt[:, t])[..., None, None]
        state = (
            alpha * state
            + beta * torch.einsum("brhn,brhp->bhpn", k_prev, v_prev)
            + gamma * torch.einsum("brhn,brhp->bhpn", k, v[:, t])
        )
        y = torch.einsum("bhpn,brhn->brhp", state, q) + m.D[:, None] * v[:, t]
        if m.is_outproj_norm:
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + m.norm.eps)
            y = y * m.norm.weight.view(H, P)
        y = y * F.silu(zr[:, t])
        y = torch.einsum("brhp,hrp->bhp", y, m.mimo_o) if m.is_mimo else y[:, 0]
        ys.append(y)
        k_prev, v_prev = k, v[:, t]
    return m.out_proj(rearrange(torch.stack(ys, dim=1), "b l h p -> b l (h p)"))


_REF_MIXER = {"Mamba": _ref_mamba1, "Mamba2": _ref_mamba2, "Mamba3": _ref_mamba3}


def _ref_logits(model, ids=None, embeddings=None):
    bb = model.backbone
    hidden, residual = (bb.embedding(ids) if embeddings is None else embeddings), None
    for block in bb.layers:
        residual = hidden if residual is None else hidden + residual
        hidden = _REF_MIXER[type(block.mixer).__name__](block.mixer, _ref_norm(residual, block.norm))
        if block.mlp is not None:
            residual = hidden + residual
            hidden = block.mlp(_ref_norm(residual, block.norm2))
    residual = hidden + residual
    return model.lm_head(_ref_norm(residual, bb.norm_f))


CONFIGS = [
    ("Mamba1", dict()),
    ("Mamba1", dict(dt_rank=3, d_intermediate=96, rms_norm=False)),
    ("Mamba2", dict()),
    ("Mamba2", dict(ngroups=2)),
    ("Mamba2", dict(d_ssm=64)),  # d_ssm < d_inner: part of the mixer is a gated MLP
    ("Mamba2", dict(rmsnorm=False)),
    ("Mamba2", dict(norm_before_gate=True)),
    ("Mamba2", dict(D_has_hdim=True)),
    ("Mamba2", dict(d_intermediate=96, rms_norm=False)),  # MLP blocks and LayerNorm
    ("Mamba3", dict()),
    ("Mamba3", dict(ngroups=2, rope_fraction=1.0)),
    ("Mamba3", dict(is_outproj_norm=True)),
    ("Mamba3", dict(is_mimo=True, mimo_rank=4, chunk_size=4)),
    ("Mamba3", dict(is_mimo=True, mimo_rank=2, rope_fraction=1.0, is_outproj_norm=True)),
    ("Mamba3", dict(is_mimo=True, mimo_rank=4, is_outproj_norm=True, d_intermediate=96)),
]
CONFIG_IDS = [f"{layer}-{'-'.join(f'{k}={v}' for k, v in cfg.items()) or 'default'}" for layer, cfg in CONFIGS]
MODEL_ARGS = ("d_intermediate", "rms_norm")


def _split_cfg(cfg):
    model_kw = {k: v for k, v in cfg.items() if k in MODEL_ARGS}
    ssm_kw = {k: v for k, v in cfg.items() if k not in MODEL_ARGS}
    return model_kw, ssm_kw


@pytest.mark.parametrize("layer,cfg", CONFIGS, ids=CONFIG_IDS)
@torch.no_grad()
def test_lrp_forward_matches_reference(layer, cfg):
    # The modified forward must compute exactly the model's function.
    model_kw, ssm_kw = _split_cfg(cfg)
    model = _make_model(layer, **model_kw, **ssm_kw)
    input_ids = torch.randint(0, VOCAB, (2, 13))
    output_ids = torch.randint(0, VOCAB, (2, 6))
    ref = _ref_logits(model, torch.cat([input_ids, output_ids], dim=1))
    for idx in [0, 3, -1]:
        attr = mamba_lrp(model, input_ids, output_ids, idx, chunk_size=4)
        pos = attr.position
        assert pos == 13 + (idx % 6) - 1
        expected = ref[:, pos].gather(-1, output_ids[:, idx % 6, None]).squeeze(-1)
        torch.testing.assert_close(attr.target_logit, expected, rtol=1e-4, atol=1e-4)
        assert attr.input_relevance.shape == (2, 13)
        assert attr.output_relevance.shape == (2, idx % 6)


CONSERVATION_CONFIGS = [
    ("Mamba1", dict(conv_bias=False)),
    ("Mamba1", dict(conv_bias=False, d_intermediate=96)),
    ("Mamba2", dict(conv_bias=False)),
    ("Mamba2", dict(conv_bias=False, ngroups=2, d_ssm=64)),
    ("Mamba2", dict(conv_bias=False, d_intermediate=96)),
    ("Mamba3", dict()),  # Mamba3 has no biases on the content path: exact with defaults
    ("Mamba3", dict(is_outproj_norm=True, ngroups=2)),
    ("Mamba3", dict(is_mimo=True, mimo_rank=4, is_outproj_norm=True, d_intermediate=96)),
]


@pytest.mark.parametrize("layer,cfg", CONSERVATION_CONFIGS, ids=[f"{l}-{c}" for l, c in CONSERVATION_CONFIGS])
def test_conservation(layer, cfg):
    # Without biases on the content path, relevance is conserved exactly at every layer.
    model_kw, ssm_kw = _split_cfg(cfg)
    model = _make_model(layer, **model_kw, **ssm_kw)
    input_ids = torch.randint(0, VOCAB, (2, 21))
    attr = mamba_lrp(model, input_ids, layer="all", normalize=None)
    assert attr.layers == [0, 1, 2, 3]
    for l in attr.layers:
        torch.testing.assert_close(attr.raw_total_relevance[l], attr.target_logit, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(attr.layer_relevance[l].sum(-1), attr.target_logit, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layer,cfg", [("Mamba1", dict(conv_bias=False)), ("Mamba2", dict(conv_bias=False)),
                                       ("Mamba3", dict()), ("Mamba3", dict(is_mimo=True, mimo_rank=4))])
def test_plain_gradient_x_input_is_not_conservative(layer, cfg):
    # Motivation for the propagation rules: on the same bias-free model, plain Gradient x Input
    # through the unmodified network does not sum to the logit, while MambaLRP does.
    model = _make_model(layer, seed=1, **cfg)
    input_ids = torch.randint(0, VOCAB, (1, 17))
    attr = mamba_lrp(model, input_ids, normalize=None)
    emb = model.backbone.embedding(input_ids).detach().requires_grad_(True)
    logit = _ref_logits(model, embeddings=emb)[0, -1, attr.target_token_ids[0]]
    (grad,) = torch.autograd.grad(logit, emb)
    plain_total = (emb * grad).sum()
    torch.testing.assert_close(logit.detach(), attr.target_logit[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(attr.raw_total_relevance[0][0], attr.target_logit[0], rtol=1e-4, atol=1e-4)
    assert (plain_total - logit).abs() > 0.1 * logit.abs()


def test_selective_scan_adjoint_gradcheck():
    # The Mamba scan's hand-written backward (adjoint scan) against finite differences.
    torch.manual_seed(0)
    batch, seqlen, dim, dstate = 2, 7, 3, 4
    x = torch.randn(batch, seqlen, dim, dtype=torch.float64, requires_grad=True)
    dt = F.softplus(torch.randn(batch, seqlen, dim, dtype=torch.float64))
    A = -torch.rand(dim, dstate, dtype=torch.float64) * 2
    B = torch.randn(batch, seqlen, dstate, dtype=torch.float64)
    C = torch.randn(batch, seqlen, dstate, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: _SelectiveScanLRP.apply(x, dt, A, B, C), (x,))


def test_batch_independence_and_api():
    model = _make_model()
    input_ids = torch.randint(0, VOCAB, (3, 10))
    output_ids = torch.randint(0, VOCAB, (3, 4))
    batched = mamba_lrp(model, input_ids, output_ids, 2)
    single = mamba_lrp(model, input_ids[1].tolist(), output_ids[1].tolist(), 2)
    torch.testing.assert_close(batched.relevance[1:2], single.relevance, rtol=1e-4, atol=1e-5)

    # Defaults: last output token, embedding layer, normalised relevance.
    attr = mamba_lrp(model, input_ids, output_ids, None, layer=None, target_token_id=None, normalize=None)
    torch.testing.assert_close(attr.target_token_ids, output_ids[:, -1])
    assert attr.layers == [0]
    assert attr.relevance.shape == (3, 13)
    norm = mamba_lrp(model, input_ids, output_ids)
    torch.testing.assert_close(norm.relevance.abs().sum(-1), torch.ones(3))
    torch.testing.assert_close(norm.relevance, attr.relevance / attr.relevance.abs().sum(-1, keepdim=True))
    assert mamba_lrp(model, input_ids, output_ids, normalize="max").relevance.abs().amax(-1).allclose(torch.ones(3))

    # No output ids: explains the argmax next-token prediction.
    nxt = mamba_lrp(model, input_ids)
    with torch.no_grad():
        torch.testing.assert_close(nxt.target_token_ids, _ref_logits(model, input_ids)[:, -1].argmax(-1))
    assert nxt.output_relevance.shape == (3, 0)

    # Layer selection, negative layer indices and an explicit target token.
    multi = mamba_lrp(model, input_ids, output_ids, 1, layer=[2, -1], target_token_id=7)
    assert multi.layers == [2, 3] and set(multi.layer_relevance) == {2, 3}
    assert (multi.target_token_ids == 7).all()
    torch.testing.assert_close(multi.relevance, multi.layer_relevance[2])
    with pytest.raises(ValueError):
        mamba_lrp(model, input_ids, layer=5)
    with pytest.raises(IndexError):
        mamba_lrp(model, input_ids, output_ids, 4)

    model.backbone.layers[1].mixer = nn.Identity()
    with pytest.raises(NotImplementedError):
        mamba_lrp(model, input_ids)


@pytest.mark.parametrize("layer", ["Mamba1", "Mamba2", "Mamba3"])
def test_causality(layer):
    # Tokens after the explained position cannot receive relevance; changing them changes nothing.
    model = _make_model(layer)
    input_ids = torch.randint(0, VOCAB, (1, 9))
    out_a = torch.randint(0, VOCAB, (1, 5))
    out_b = out_a.clone()
    out_b[:, 3:] = (out_b[:, 3:] + 1) % VOCAB
    a = mamba_lrp(model, input_ids, out_a, 2)
    b = mamba_lrp(model, input_ids, out_b, 2)
    torch.testing.assert_close(a.relevance, b.relevance)


def _cuda_model(layer, fused_add_norm, **ssm_cfg):
    torch.manual_seed(0)
    config = MambaConfig(
        d_model=128, n_layer=2, vocab_size=VOCAB, ssm_cfg=dict(layer=layer, **ssm_cfg),
        fused_add_norm=fused_add_norm,
    )
    return MambaLMHeadModel(config, device="cuda", dtype=torch.float32).eval()


CUDA_CONFIGS = [
    ("Mamba1", dict()),
    ("Mamba2", dict(headdim=32)),
    ("Mamba3", dict(headdim=32, d_state=64)),
    ("Mamba3", dict(headdim=32, d_state=64, is_mimo=True, mimo_rank=4, chunk_size=16, is_outproj_norm=True)),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layer,cfg", CUDA_CONFIGS, ids=[f"{l}-{c}" for l, c in CUDA_CONFIGS])
@pytest.mark.parametrize("fused_add_norm", [False, True])
def test_matches_model_forward_cuda(layer, cfg, fused_add_norm):
    if layer == "Mamba1":
        from mamba_ssm.ops import selective_scan_interface
        if selective_scan_interface.selective_scan_cuda is None:
            pytest.skip("selective_scan_cuda is not installed")
    if cfg.get("is_mimo"):
        pytest.importorskip("tilelang")
    model = _cuda_model(layer, fused_add_norm, **cfg)
    input_ids = torch.randint(0, VOCAB, (2, 37), device="cuda")
    output_ids = torch.randint(0, VOCAB, (2, 5), device="cuda")
    attr = mamba_lrp(model, input_ids, output_ids, 3)
    with torch.no_grad():
        logits = model(torch.cat([input_ids, output_ids[:, :3]], dim=1)).logits[:, -1]
    torch.testing.assert_close(attr.target_logit, logits.gather(-1, output_ids[:, 3:4]).squeeze(-1),
                               rtol=1e-3, atol=1e-3)
