"""CPU demo: populate a results store with small runs on a synthetic task, then export the report.

Uses tiny random backbones (pure-PyTorch reference implementation, no downloads), so it runs in a
few minutes on a laptop and exercises every dashboard page. The numbers are about the pipeline,
not about Mamba.

    python scripts/demo_cpu.py --out demo_results
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets  # noqa: E402

from mambacls.data.ingest import finalize_splits  # noqa: E402
from mambacls.data.registry import get_spec  # noqa: E402
from mambacls.data.tokenize import WhitespaceTokenizer  # noqa: E402
from mambacls.experiment import run_experiment  # noqa: E402
from mambacls.store.results import ResultsStore  # noqa: E402


def synthetic_task(n_train=600, n_test=200, n_classes=4, seed=0):
    """Topic-style task: one class keyword hidden among distractor words; lengths 8-120 words."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(200)]
    keys = [f"topic{c}" for c in range(n_classes)]

    def make(n):
        texts, labels = [], []
        for _ in range(n):
            y = rng.randrange(n_classes)
            words = [rng.choice(vocab) for _ in range(int(rng.lognormvariate(3.3, 0.6)) + 8)]
            words.insert(rng.randrange(len(words) + 1), keys[y])
            texts.append(" ".join(words[:400]))
            labels.append(y)
        return datasets.Dataset.from_dict({"text": texts, "label": labels})

    return finalize_splits(datasets.DatasetDict({"train": make(n_train), "test": make(n_test)}), get_spec("ag_news"))


def tiny(layer, **ssm):
    cfg = dict(layer=layer, d_state=16, **({"headdim": 16, "chunk_size": 16} if layer != "Mamba1" else {}), **ssm)
    return {"kind": "mamba", "impl": "reference", "name": f"tiny-{layer.lower()}" + ("-mimo" if ssm.get("is_mimo") else ""),
            "config": dict(d_model=48, n_layer=3, vocab_size=50280, ssm_cfg=cfg, fused_add_norm=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_results")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    store = ResultsStore(out / "store")
    data = synthetic_task()
    tok = WhitespaceTokenizer()
    base = {"max_len": 256, "precision": "fp32", "data": {"name": "ag_news", "tokenizer": "whitespace"},
            "batching": {"mode": "right_pad", "bucket_by_length": True, "tokens_per_batch": 2048},
            "optim": {"lr_backbone": 2e-3, "lr_head": 3e-3}, "train": {"epochs": 3, "eval_every": 0.5, "patience": 0},
            "tracking": {"artifacts": str(out / "artifacts.zarr"), "n_embeddings": 200}}
    grid = [
        ("A", tiny("Mamba2"), {"name": "scalar_mix"}, {"name": "probe"}, {}),
        ("B", tiny("Mamba2"), {"name": "mean"}, {"name": "lora", "r": 4}, {"landscape": {"enabled": True, "steps": 9}}),
        ("B", tiny("Mamba2"), {"name": "attn"}, {"name": "full"}, {"landscape": {"enabled": True, "steps": 9}}),
        ("B", tiny("Mamba2"), {"name": "last"}, {"name": "state_offset", "variant": "h"}, {"landscape": {"enabled": True, "steps": 9}}),
        ("B", tiny("Mamba3"), {"name": "latent_query", "n_queries": 4}, {"name": "lora", "r": 4}, {}),
        ("B", tiny("Mamba2"), {"name": "mean"}, {"name": "bidir+lora", "bidir": {"mode": "tied_gate"}, "lora": {"r": 4}}, {}),
        ("C", tiny("Mamba2"), {"name": "mean"}, {"name": "lora", "r": 4},
         {"max_len": 64, "length_eval": {"eval_lengths": [16, 32, 64, 128, 256], "longmamba": True, "train_len": 64}}),
    ]
    for seed in range(a.seeds):
        for i, (phase, bb, pooler, adapter, extra) in enumerate(grid):
            cfg = {**base, **extra, "seed": seed, "phase": phase, "backbone": bb, "pooler": pooler, "adapter": adapter,
                   "probe": {"enabled": seed == 0, "n_examples": 8, "what": ["resid", "dt", "state", "state_norm", "ssd_matrix"],
                             "at": [0.0, 1.0]},
                   "efficiency": {"enabled": seed == 0, "seq_lens": [32, 64, 128, 256], "batch_sizes": [1, 8], "warmup": 1, "iters": 3}}
            row = run_experiment(cfg, datasets_override=data, tokenizer=tok, store=store, device="cpu")
            print(f"[{phase}] seed {seed} {bb['name']:14s} {pooler['name']:12s} {adapter['name']:12s} acc={row['acc']:.3f} f1={row['macro_f1']:.3f}")
    from export_static_report import export

    print("report:", export(out / "store", out / "artifacts.zarr", out / "report"))


if __name__ == "__main__":
    main()
