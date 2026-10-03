"""Config loading through Hydra's compose API (for programmatic use, sweeps and tests)."""

from pathlib import Path
from typing import Dict, List, Optional

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def load_config(overrides: Optional[List[str]] = None, config_dir: Optional[str] = None, config_name: str = "config") -> Dict:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(config_dir or CONF_DIR).resolve()), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return OmegaConf.to_container(cfg, resolve=True)
