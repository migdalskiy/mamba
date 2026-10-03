"""InternalsRecorder: capture residual streams and SSM internals on a small probe batch (spec §7.2).

Only used on a fixed probe set at checkpoints, never in the training hot path. Mixers of the
selected layers run through the decomposed forward (§7.4 workaround 1), the states / decay /
hidden-attention matrices are then recomputed in float32 by ``probe.ssd_reference``
(workaround 2).
"""

from typing import Dict, Iterable, List, Literal, Optional, Set

import torch

from mambacls.models.mixers import mixer_kind
from mambacls.probe.hidden_attention import internals_from_record

WHAT = ("resid", "dt", "B", "C", "x", "state", "state_norm", "ssd_matrix", "decay")

# position axes per quantity, for ``position_stride`` subsampling when dumping
_POS_AXES = {"resid": (1,), "dt": (1,), "B": (1,), "C": (1,), "x": (1,), "state": (1,), "state_norm": (1,),
             "ssd_matrix": (-2, -1), "decay": (-2, -1)}


class InternalsRecorder:
    def __init__(self, model, layers: Optional[Iterable[int]] = None, what: Iterable[str] = ("resid", "dt", "state_norm", "ssd_matrix"),
                 position_stride: int = 1, mamba1_channels: Optional[List[int]] = None):
        self.backbone = getattr(model, "backbone", model)
        n = self.backbone.n_layers
        self.layers = list(range(n)) if layers in (None, "all") else [l % n for l in layers]
        self.what: Set[str] = set(what)
        unknown = self.what - set(WHAT)
        if unknown:
            raise ValueError(f"unknown quantities {unknown}; choose from {WHAT}")
        self.stride = position_stride
        self.mamba1_channels = mamba1_channels if mamba1_channels is not None else list(range(8))
        self._records: Dict[int, dict] = {}
        self.results: Dict[int, Dict[str, torch.Tensor]] = {}
        self.meta: Dict = {}

    # used by MambaBackbone._mixer
    def slot(self, i: int):
        if i not in self.layers:
            return None
        return self._records.setdefault(i, {})

    def __enter__(self):
        self._records = {}
        self.backbone.recorder = self
        return self

    def __exit__(self, *exc):
        self.backbone.recorder = None
        return False

    @torch.no_grad()
    def run(self, batch) -> Dict[int, Dict[str, torch.Tensor]]:
        was_training = self.backbone.training
        self.backbone.eval()
        with self:
            out = self.backbone(batch.input_ids, lengths=batch.lengths, return_all_layers="resid" in self.what,
                                normalize_layers=False)
        self.backbone.train(was_training)
        self.meta = {"lengths": batch.lengths.cpu(), "input_ids": batch.input_ids.cpu()}
        self.results = self.compute(out)
        return self.results

    def compute(self, out=None) -> Dict[int, Dict[str, torch.Tensor]]:
        results: Dict[int, Dict[str, torch.Tensor]] = {}
        need_ref = bool(self.what & {"state", "state_norm", "ssd_matrix", "decay"})
        for i in self.layers:
            res: Dict[str, torch.Tensor] = {}
            if "resid" in self.what and out is not None and out.layers is not None:
                res["resid"] = out.layers[i].float().cpu()
            rec = self._records.get(i)
            if rec:
                mixer = self.backbone.model.layers[i].mixer
                kind = mixer_kind(mixer)
                for name in ("dt", "B", "C", "x"):
                    if name in self.what and name in rec:
                        res[name] = rec[name].float().cpu()
                if need_ref:
                    ref = internals_from_record(kind, rec, mixer, return_states=bool(self.what & {"state", "state_norm"}),
                                                mamba1_channels=self.mamba1_channels)
                    for name in ("state", "state_norm", "ssd_matrix", "decay"):
                        if name in self.what and name in ref:
                            res[name] = ref[name].float().cpu()
                    res["_y_reference"] = ref["y"].float().cpu()
                    if "experimental" in ref:
                        res["_experimental"] = torch.tensor(True)
            results[i] = res
        return results

    def _subsample(self, name, t):
        if self.stride <= 1 or name not in _POS_AXES:
            return t
        for ax in _POS_AXES[name]:
            idx = torch.arange(0, t.shape[ax], self.stride)
            t = t.index_select(ax, idx)
        return t

    def dump(self, zarr_path: str, run_id: str, step: int) -> None:
        from mambacls.store.artifacts import ArtifactStore

        store = ArtifactStore(zarr_path)
        for i, res in self.results.items():
            experimental = "_experimental" in res
            for name, t in res.items():
                if name.startswith("_"):
                    continue
                store.write(run_id, step, i, name, self._subsample(name, t),
                            attrs={"layer": i, "position_stride": self.stride, "experimental": experimental,
                                   "kind": self.backbone.layer_types[i]})
        for name, t in self.meta.items():
            store.write(run_id, step, None, name, t)
