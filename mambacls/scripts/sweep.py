"""Expand a phase grid (conf/experiment/*.yaml ``sweep:`` section) into runs.

    python scripts/sweep.py peft_sweep --print            # list the override sets
    python scripts/sweep.py pooling_sweep --run           # run sequentially here
    python scripts/sweep.py peft_sweep --shard 3/8 --run  # one-run-per-GPU style sharding

Phase B lr protocol: seed 0 sweeps ``lr_sweep`` values; seeds 1-4 reuse the best lr per cell
(selected on validation macro-F1 from the results store).
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Transformer baselines run with standard fine-tuning (or a probe); Mamba PEFT methods do not apply.
BASELINES = ("modernbert_", "deberta_", "pythia_")
BASELINE_ADAPTERS = ("full", "probe")
sys.path.insert(0, str(ROOT))


def expand(experiment: str):
    spec = yaml.safe_load((ROOT / "conf" / "experiment" / f"{experiment}.yaml").read_text())
    sweep = dict(spec.get("sweep", {}))
    lr = sweep.pop("lr_sweep", None)
    per_backbone = sweep.pop("overrides_by_backbone", {})
    keys = list(sweep)
    runs = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        combo = dict(zip(keys, values))
        if str(combo.get("backbone", "")).startswith(BASELINES) and combo.get("adapter", "full") not in BASELINE_ADAPTERS:
            continue
        extra = per_backbone.get(combo.get("backbone"), {})
        if lr and combo.get("seed") == lr.get("seed", 0):
            for v in lr["optim.lr_backbone"]:
                runs.append({**combo, **extra, "optim.lr_backbone": v})
        else:
            runs.append({**combo, **extra})
    return runs


def to_overrides(experiment, run):
    return [f"+experiment={experiment}"] + [f"{k}={v}" for k, v in run.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    runs = [r for j, r in enumerate(expand(a.experiment)) if j % n == i]
    for r in runs:
        ov = to_overrides(a.experiment, r)
        if a.print or not a.run:
            print(" ".join(ov))
        if a.run:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), *ov], check=False)
    print(f"# {len(runs)} runs", file=sys.stderr)


if __name__ == "__main__":
    main()
