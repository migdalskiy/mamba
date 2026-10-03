"""Tensor artifacts (probe captures) in Zarr: <root>/<run_id>/<step>/<layer>/<name>."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


class ArtifactStore:
    def __init__(self, path: Union[str, Path]):
        import zarr

        self.path = str(path)
        self._zarr = zarr
        self.root = zarr.open_group(self.path, mode="a")

    @staticmethod
    def key(run_id: str, step: int, layer: Optional[int], name: str) -> str:
        layer_part = "global" if layer is None else f"layer_{layer:03d}"
        return f"{run_id}/step_{step:08d}/{layer_part}/{name}"

    def write(self, run_id: str, step: int, layer: Optional[int], name: str, array, attrs: Optional[Dict] = None):
        arr = np.asarray(array.detach().float().cpu() if hasattr(array, "detach") else array)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        group_path, leaf = self.key(run_id, step, layer, name).rsplit("/", 1)
        g = self.root.require_group(group_path)
        z = g.create_array(leaf, shape=arr.shape, dtype=arr.dtype, overwrite=True)
        z[...] = arr
        if attrs:
            z.attrs.update({k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v)) for k, v in attrs.items()})
        return z

    def read(self, run_id: str, step: int, layer: Optional[int], name: str) -> np.ndarray:
        return self.root[self.key(run_id, step, layer, name)][...]

    def attrs(self, run_id, step, layer, name) -> Dict:
        return dict(self.root[self.key(run_id, step, layer, name)].attrs)

    def runs(self) -> List[str]:
        return sorted(self.root.group_keys())

    def steps(self, run_id: str) -> List[int]:
        return sorted(int(k.split("_")[1]) for k in self.root[run_id].group_keys())

    def entries(self, run_id: str, step: int) -> Dict[str, List[str]]:
        g = self.root[f"{run_id}/step_{step:08d}"]
        return {lk: sorted(g[lk].array_keys()) for lk in sorted(g.group_keys())}
