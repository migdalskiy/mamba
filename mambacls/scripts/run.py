"""Single run from the Hydra config tree.

    python scripts/run.py backbone=mamba3_siso_893m pooler=latent_query adapter=lora data=banking77 seed=0
    python scripts/run.py +experiment=peft_sweep
    python scripts/run.py +experiment=smoke            # CPU-sized sanity run
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mambacls.experiment import run_experiment  # noqa: E402


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("sweep", None)
    root = Path(hydra.utils.get_original_cwd())
    for k in ("store_dir", "artifacts"):
        p = Path(cfg["tracking"][k])
        cfg["tracking"][k] = str(p if p.is_absolute() else root / p)
    row = run_experiment(cfg)
    print({k: row.get(k) for k in ("run_id", "dataset", "backbone", "pooler", "adapter", "seed", "acc", "macro_f1", "ece")})


if __name__ == "__main__":
    main()
