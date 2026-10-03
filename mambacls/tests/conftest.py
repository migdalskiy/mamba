import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mamba_ssm.modules.mamba3 as mamba3_module  # noqa: E402

VOCAB = 101


@pytest.fixture(autouse=True)
def _allow_mimo_without_tilelang(monkeypatch):
    # Mamba3(is_mimo=True) asserts the TileLang kernel imported; the reference path does not use it.
    if mamba3_module.mamba3_mimo_combined is None:
        monkeypatch.setattr(mamba3_module, "mamba3_mimo_combined", object())


MIXERS = {
    "Mamba1": dict(layer="Mamba1", d_state=8),
    "Mamba2": dict(layer="Mamba2", d_state=16, headdim=16, chunk_size=8),
    "Mamba2-groups": dict(layer="Mamba2", d_state=16, headdim=16, chunk_size=8, ngroups=2, d_ssm=32),
    "Mamba3": dict(layer="Mamba3", d_state=16, headdim=16),
    "Mamba3-mimo": dict(layer="Mamba3", d_state=16, headdim=16, is_mimo=True, mimo_rank=2, is_outproj_norm=True),
}


def backbone_cfg(mixer="Mamba2", n_layer=2, d_model=32, impl="reference", **extra):
    ssm = dict(MIXERS[mixer])
    ssm.update(extra.pop("ssm", {}))
    return {"kind": "mamba", "impl": impl, "name": f"tiny-{mixer}",
            "config": dict(d_model=d_model, n_layer=n_layer, vocab_size=VOCAB, ssm_cfg=ssm, fused_add_norm=False, **extra)}


def make_classifier(mixer="Mamba2", pooler="last", adapter=None, n_classes=3, seed=0, **pkw):
    from mambacls.models.registry import build_classifier

    torch.manual_seed(seed)
    cfg = {"backbone": {**backbone_cfg(mixer), "init_seed": seed}, "pooler": {"name": pooler, **pkw},
           "adapter": adapter or {"name": "probe"}, "data": {"n_classes": n_classes}}
    return build_classifier(cfg)


def random_features(lengths, n_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(2, VOCAB, (n,), generator=g).tolist(), "label": i % n_classes, "example_id": i}
            for i, n in enumerate(lengths)]


def keyword_dataset(n_train=240, n_test=80, n_classes=2, seed=0):
    """Tiny learnable task: the label is given by one keyword among random words."""
    import datasets

    from mambacls.data.ingest import finalize_splits
    from mambacls.data.registry import get_spec

    rng = random.Random(seed)
    words = [f"w{i}" for i in range(40)]

    def make(n):
        texts, labels = [], []
        for _ in range(n):
            y = rng.randrange(n_classes)
            t = [rng.choice(words) for _ in range(rng.randint(4, 14))]
            t.insert(rng.randint(0, len(t)), f"key{y}")
            texts.append(" ".join(t))
            labels.append(y)
        return datasets.Dataset.from_dict({"text": texts, "label": labels})

    return finalize_splits(datasets.DatasetDict({"train": make(n_train), "test": make(n_test)}), get_spec("sst2"))
