"""Results store: Parquet files are the source of truth, DuckDB provides SQL views over them
(spec §5). Every row carries the run's provenance."""

import hashlib
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

TABLES = ("runs", "metrics", "history", "predictions", "efficiency", "data_stats", "lengths", "grad_norms",
          "landscape", "significance", "probe_index")


def config_hash(cfg) -> str:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_sha(path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def provenance(cfg=None, seed: Optional[int] = None) -> Dict:
    """config hash, git SHA, mamba_ssm version + commit, torch / CUDA / triton versions, GPU, seed."""
    import torch

    import mamba_ssm

    info = {
        "config_hash": config_hash(cfg) if cfg is not None else None,
        "git_sha": _git_sha(Path(__file__).resolve().parent),
        "mamba_ssm_version": getattr(mamba_ssm, "__version__", None),
        "mamba_ssm_commit": _git_sha(Path(mamba_ssm.__file__).resolve().parent),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": platform.python_version(),
        "seed": seed,
    }
    try:
        import triton

        info["triton_version"] = triton.__version__
    except ImportError:
        info["triton_version"] = None
    return info


class ResultsStore:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, table: str, rows: Union[pd.DataFrame, Iterable[Dict]], provenance_info: Optional[Dict] = None,
              run_id: Optional[str] = None) -> Path:
        df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if run_id is not None:
            df["run_id"] = run_id
        for k, v in (provenance_info or {}).items():
            df[k] = v
        d = self.root / table
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"part-{uuid.uuid4().hex}.parquet"
        df.to_parquet(path, index=False)
        return path

    def tables(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.parquet")))

    def read(self, table: str) -> pd.DataFrame:
        files = sorted((self.root / table).glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def connect(self, database: Optional[str] = None):
        """DuckDB connection with one view per table (``union_by_name`` tolerates schema growth)."""
        import duckdb

        con = duckdb.connect(database or ":memory:")
        for t in self.tables():
            pattern = str(self.root / t / "*.parquet").replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
        return con

    def query(self, sql: str) -> pd.DataFrame:
        con = self.connect()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def export_duckdb(self, path: Union[str, Path]) -> Path:
        """Materialise all tables into a DuckDB file (e.g. results/mambacls.duckdb)."""
        import duckdb

        path = Path(path)
        if path.exists():
            path.unlink()
        con = duckdb.connect(str(path))
        try:
            for t in self.tables():
                pattern = str(self.root / t / "*.parquet").replace("'", "''")
                con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
        finally:
            con.close()
        return path
