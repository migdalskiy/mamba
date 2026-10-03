import numpy as np
import pandas as pd
import pytest
import torch

from conftest import MIXERS, make_classifier, random_features
from mambacls.data.collate import RightPadCollator
from mambacls.probe.capture import InternalsRecorder
from mambacls.probe.ssd_reference import mamba2_reference
from mambacls.store.artifacts import ArtifactStore
from mambacls.store.results import ResultsStore, config_hash, provenance


@pytest.mark.parametrize("mixer", sorted(MIXERS))
def test_reference_internals_reconstruct_mixer_output(mixer):
    """M x + D x (hidden attention) reproduces the pre-gate SSM output of every mixer type."""
    model = make_classifier(mixer)
    batch = RightPadCollator()(random_features([9, 6]))
    rec = InternalsRecorder(model, what=("resid", "dt", "state", "state_norm", "ssd_matrix", "decay"))
    res = rec.run(batch)
    for layer, r in res.items():
        y = rec._records[layer]["y"].float()
        if mixer == "Mamba1":
            y = y[..., : len(rec.mamba1_channels)]
        torch.testing.assert_close(r["_y_reference"], y, rtol=1e-4, atol=1e-5)
        assert r["resid"].shape[:2] == (2, 9) and "state_norm" in r
        assert ("_experimental" in r) == mixer.startswith("Mamba3")


def test_mamba3_state_recurrence_reproduces_output():
    """Experimental Mamba-3 states: y_t = h_t q_t + D v_t from the trapezoidal recurrence."""
    model = make_classifier("Mamba3-mimo")
    rec = InternalsRecorder(model, layers=[0], what=("state",))
    res = rec.run(RightPadCollator()(random_features([7])))
    r = rec._records[0]
    D = model.backbone.model.layers[0].mixer.D
    y_from_state = torch.einsum("blhpn,blrhn->blrhp", res[0]["state"], r["q"].float()) + r["v"].float() * D.float()[:, None]
    torch.testing.assert_close(y_from_state, r["y"].float(), rtol=1e-4, atol=1e-5)


def test_mamba2_reference_states_match_matrix():
    torch.manual_seed(0)
    b, l, h, p, n = 1, 6, 2, 3, 4
    out = mamba2_reference(torch.randn(b, l, h, p), torch.rand(b, l, h), -torch.rand(h), torch.randn(b, l, 1, n),
                           torch.randn(b, l, 1, n))
    assert out["state"].shape == (b, l, h, p, n) and out["ssd_matrix"].shape == (b, h, l, l)
    assert torch.triu(out["ssd_matrix"][0, 0], 1).abs().max() == 0  # causal


def test_recorder_dump_and_stride(tmp_path):
    model = make_classifier("Mamba2")
    rec = InternalsRecorder(model, what=("dt", "ssd_matrix"), position_stride=2)
    rec.run(RightPadCollator()(random_features([8])))
    rec.dump(str(tmp_path / "a.zarr"), "run1", 5)
    arts = ArtifactStore(tmp_path / "a.zarr")
    assert arts.runs() == ["run1"] and arts.steps("run1") == [5]
    assert arts.read("run1", 5, 0, "dt").shape[1] == 4 and arts.read("run1", 5, 0, "ssd_matrix").shape[-2:] == (4, 4)
    assert arts.attrs("run1", 5, 0, "dt")["position_stride"] == 2
    assert "lengths" in arts.entries("run1", 5)["global"]


def test_results_store_parquet_duckdb(tmp_path):
    store = ResultsStore(tmp_path / "store")
    prov = provenance({"a": 1}, seed=3)
    for k in ("config_hash", "git_sha", "mamba_ssm_version", "torch_version", "seed", "triton_version"):
        assert k in prov
    store.write("metrics", [{"acc": 0.5, "method": "a"}], prov, run_id="r1")
    store.write("metrics", pd.DataFrame([{"acc": 0.7, "method": "b", "extra_col": 1}]), prov, run_id="r2")
    df = store.read("metrics")
    assert len(df) == 2 and set(df.run_id) == {"r1", "r2"} and df.seed.eq(3).all()
    q = store.query("select method, acc from metrics order by acc desc")
    assert q.method.tolist() == ["b", "a"]
    db = store.export_duckdb(tmp_path / "m.duckdb")
    import duckdb

    assert duckdb.connect(str(db)).execute("select count(*) from metrics").fetchone()[0] == 2
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})
