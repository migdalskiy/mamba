import os

import pytest

import sweep as sw
from mambacls.config import CONF_DIR, load_config


@pytest.mark.parametrize("group", ["backbone", "pooler", "adapter", "data"])
def test_every_group_option_composes(group):
    for f in sorted(os.listdir(CONF_DIR / group)):
        cfg = load_config([f"{group}={f[:-5]}"])
        assert cfg[group]


def test_experiment_grids():
    counts = {e: len(sw.expand(e)) for e in ("pooling_sweep", "peft_sweep", "length_sweep", "hybrid_triple")}
    assert counts["pooling_sweep"] == 7 * 13 * 4 * 5  # Phase A: poolers x backbones x datasets x seeds
    # Phase B: 7 adapters x 3 Mamba backbones + 4 baselines x {full, probe}, x 5 datasets x (4 lrs on seed 0 + 4 seeds)
    assert counts["peft_sweep"] == (7 * 3 + 4 * 2) * 5 * 8
    assert counts["hybrid_triple"] == 3 * 3 * 5
    for e in counts:
        cfg = load_config(sw.to_overrides(e, sw.expand(e)[-1]))
        assert cfg["phase"] in ("A", "B", "C", "D")


def test_peft_example_matches_spec():
    cfg = load_config(["+experiment=peft_sweep"])
    assert cfg["backbone"]["hf_id"] == "state-spaces/mamba3-siso-893m" and cfg["pooler"]["name"] == "latent_query"
    assert cfg["adapter"] == {"name": "lora", "r": 16, "alpha": 32, "dropout": 0.05, "targets": ["in_proj", "out_proj"]}
    assert cfg["optim"]["no_decay"] == ["A_log", "D", "dt_bias", "norm", "bias"] and cfg["precision"] == "bf16_amp"


def test_deberta_length_policy():
    runs = [r for r in sw.expand("length_sweep") if r["backbone"] == "deberta_v3_base"]
    assert runs and all(r["max_len"] == 512 and r["data.truncation"] == "head_tail" for r in runs)
