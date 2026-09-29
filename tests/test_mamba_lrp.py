import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.explain import mamba_lrp
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def _make_model(device="cpu", d_intermediate=0, rms_norm=True, seed=0, **ssm_cfg):
    torch.manual_seed(seed)
    cfg = dict(layer="Mamba2", d_state=16, headdim=16, chunk_size=8)
    cfg.update(ssm_cfg)
    config = MambaConfig(
        d_model=64,
        n_layer=3,
        d_intermediate=d_intermediate,
        vocab_size=101,
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
            elif name.endswith(".D"):
                p.uniform_(-1.0, 1.0)
            elif name.endswith("conv1d.bias") or (name.endswith(".bias") and "norm" in name):
                p.uniform_(-0.1, 0.1)
    return model.eval()


def _ref_norm(x, norm):
    if isinstance(norm, nn.LayerNorm):
        return F.layer_norm(x, norm.weight.shape, norm.weight, norm.bias, norm.eps)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps) * norm.weight


def _ref_mamba2(m, u):
    """Sequential-recurrence reference for Mamba2.forward, written independently of mamba_lrp."""
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


def _ref_logits(model, ids):
    bb = model.backbone
    hidden, residual = bb.embedding(ids), None
    for block in bb.layers:
        residual = hidden if residual is None else hidden + residual
        hidden = _ref_mamba2(block.mixer, _ref_norm(residual, block.norm))
        if block.mlp is not None:
            residual = hidden + residual
            hidden = block.mlp(_ref_norm(residual, block.norm2))
    residual = hidden + residual
    return model.lm_head(_ref_norm(residual, bb.norm_f))


CONFIGS = [
    dict(),
    dict(ngroups=2),
    dict(d_ssm=64),  # d_ssm < d_inner: part of the mixer is a gated MLP
    dict(rmsnorm=False),
    dict(norm_before_gate=True),
    dict(D_has_hdim=True),
    dict(d_intermediate=96, rms_norm=False),  # MLP blocks and LayerNorm
]


@pytest.mark.parametrize("cfg", CONFIGS)
@torch.no_grad()
def test_lrp_forward_matches_reference(cfg):
    model = _make_model(**cfg)
    input_ids = torch.randint(0, 101, (2, 13))
    output_ids = torch.randint(0, 101, (2, 6))
    ref = _ref_logits(model, torch.cat([input_ids, output_ids], dim=1))
    for idx in [0, 3, -1]:
        attr = mamba_lrp(model, input_ids, output_ids, idx)
        pos = attr.position
        assert pos == 13 + (idx % 6) - 1
        expected = ref[:, pos].gather(-1, output_ids[:, idx % 6, None]).squeeze(-1)
        torch.testing.assert_close(attr.target_logit, expected, rtol=1e-4, atol=1e-4)
        assert attr.input_relevance.shape == (2, 13)
        assert attr.output_relevance.shape == (2, idx % 6)


@pytest.mark.parametrize("cfg", [dict(), dict(ngroups=2, d_ssm=64), dict(d_intermediate=96)])
def test_conservation(cfg):
    # Without biases, relevance is conserved exactly at every layer.
    model = _make_model(conv_bias=False, **cfg)
    input_ids = torch.randint(0, 101, (2, 21))
    attr = mamba_lrp(model, input_ids, layer="all", normalize=None)
    assert attr.layers == [0, 1, 2, 3]
    for l in attr.layers:
        torch.testing.assert_close(attr.raw_total_relevance[l], attr.target_logit, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(attr.layer_relevance[l].sum(-1), attr.target_logit, rtol=1e-4, atol=1e-4)


def test_batch_independence_and_api():
    model = _make_model()
    input_ids = torch.randint(0, 101, (3, 10))
    output_ids = torch.randint(0, 101, (3, 4))
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


def test_causality():
    # Tokens after the explained position cannot receive relevance; changing them changes nothing.
    model = _make_model()
    input_ids = torch.randint(0, 101, (1, 9))
    out_a = torch.randint(0, 101, (1, 5))
    out_b = out_a.clone()
    out_b[:, 3:] = (out_b[:, 3:] + 1) % 101
    a = mamba_lrp(model, input_ids, out_a, 2)
    b = mamba_lrp(model, input_ids, out_b, 2)
    torch.testing.assert_close(a.relevance, b.relevance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fused_add_norm", [False, True])
def test_matches_model_forward_cuda(fused_add_norm):
    torch.manual_seed(0)
    config = MambaConfig(
        d_model=128, n_layer=2, vocab_size=101, ssm_cfg=dict(layer="Mamba2", headdim=32),
        fused_add_norm=fused_add_norm,
    )
    model = MambaLMHeadModel(config, device="cuda", dtype=torch.float32).eval()
    input_ids = torch.randint(0, 101, (2, 37), device="cuda")
    output_ids = torch.randint(0, 101, (2, 5), device="cuda")
    attr = mamba_lrp(model, input_ids, output_ids, 3)
    with torch.no_grad():
        logits = model(torch.cat([input_ids, output_ids[:, :3]], dim=1)).logits[:, -1]
    torch.testing.assert_close(attr.target_logit, logits.gather(-1, output_ids[:, 3:4]).squeeze(-1),
                               rtol=1e-3, atol=1e-3)
