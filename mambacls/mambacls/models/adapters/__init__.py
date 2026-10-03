import inspect

from mambacls.models.adapters.base import Adapter, CompositeAdapter, FullFinetune, ProbeAdapter
from mambacls.models.adapters.bidir import BidirAdapter
from mambacls.models.adapters.longmamba import LongMambaFilter
from mambacls.models.adapters.lora import LoRAAdapter
from mambacls.models.adapters.prompt import PromptAdapter
from mambacls.models.adapters.sdt import SDLoRAAdapter
from mambacls.models.adapters.state_offset import StateOffsetAdapter

ADAPTERS = {
    "probe": ProbeAdapter,
    "full": FullFinetune,
    "lora": LoRAAdapter,
    "sdlora": SDLoRAAdapter,
    "state_offset": StateOffsetAdapter,
    "prompt": PromptAdapter,
    "bidir": BidirAdapter,
    "longmamba": LongMambaFilter,
}


def _accepted(cls, kwargs):
    """Drop config keys the adapter does not take (experiment-level ``adapter:`` blocks are merged
    into whichever adapter a sweep selects). LoRA keys are forwarded to SDLoRA's parent."""
    params = set(inspect.signature(cls.__init__).parameters)
    if cls is SDLoRAAdapter:
        params |= set(inspect.signature(LoRAAdapter.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in params}


def build_adapter(name: str, **kwargs) -> Adapter:
    """name may combine adapters with '+', e.g. 'bidir+lora'; each part then takes its kwargs from
    the sub-dict with its name."""
    parts = name.split("+")
    if len(parts) == 1:
        if name not in ADAPTERS:
            raise KeyError(f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}")
        return ADAPTERS[name](**_accepted(ADAPTERS[name], kwargs))
    return CompositeAdapter([ADAPTERS[p](**_accepted(ADAPTERS[p], kwargs.get(p, {}))) for p in parts])


__all__ = [
    "ADAPTERS", "Adapter", "BidirAdapter", "CompositeAdapter", "FullFinetune", "LoRAAdapter", "LongMambaFilter",
    "ProbeAdapter", "PromptAdapter", "SDLoRAAdapter", "StateOffsetAdapter", "build_adapter",
]
