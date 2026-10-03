"""M0 checkpoint smoke test (spec §9): load every Mamba-2 / hybrid / Mamba-3 checkpoint, run one
bf16 forward at L=2048 and check finite outputs and the expected ssm_cfg.layer. Needs a GPU and
Hugging Face access.

    python scripts/checkpoint_smoke.py [--ids state-spaces/mamba2-130m ...] [--seqlen 2048]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mambacls.models.backbone import load_backbone  # noqa: E402

IDS = {
    **{f"state-spaces/mamba2-{s}": "Mamba2" for s in ("130m", "370m", "780m", "1.3b", "2.7b")},
    "state-spaces/mamba2attn-2.7b": "Mamba2",
    **{f"state-spaces/mamba3-siso-{s}": "Mamba3" for s in ("187m", "443m", "893m", "1.5b")},
    **{f"state-spaces/mamba3-mimo-{s}": "Mamba3" for s in ("187m", "444m", "894m", "1.5b")},
}


def check(hf_id, expected, seqlen):
    bb = load_backbone(hf_id, allowed_layers=(expected,), device="cuda", dtype=torch.float32)
    layer = (bb.config.ssm_cfg or {}).get("layer", "Mamba1")
    ids = torch.randint(0, bb.embedding.num_embeddings, (1, seqlen), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = bb(ids)
    ok = layer == expected and torch.isfinite(out.last_hidden).all().item()
    return {"hf_id": hf_id, "layer": layer, "expected": expected, "finite": bool(torch.isfinite(out.last_hidden).all()), "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=list(IDS))
    ap.add_argument("--seqlen", type=int, default=2048)
    a = ap.parse_args()
    failures = 0
    for hf_id in a.ids:
        try:
            r = check(hf_id, IDS.get(hf_id, "Mamba2"), a.seqlen)
        except Exception as e:  # report and continue: the M0 gate lists every failing ID
            r = {"hf_id": hf_id, "ok": False, "error": repr(e)[:300]}
        failures += not r["ok"]
        print(r, flush=True)
        torch.cuda.empty_cache()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
