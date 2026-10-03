"""Dispatch from a recorded mixer to its reference internals (hidden attention, states, decay)."""

from typing import Dict

from mambacls.probe.ssd_reference import mamba1_reference, mamba2_reference, mamba3_reference


def internals_from_record(kind: str, record: Dict, mixer=None, return_states: bool = True, mamba1_channels=None):
    if kind == "Mamba2":
        x = record["x"]
        return mamba2_reference(x, record["dt"], record["A"], record["B"], record["C"], record.get("D"), return_states)
    if kind == "Mamba1":
        return mamba1_reference(record["x"], record["dt"], record["A"], record["B"], record["C"],
                                None if mixer is None else mixer.D, channels=mamba1_channels, return_states=return_states)
    if kind == "Mamba3":
        return mamba3_reference(record, None if mixer is None else mixer.D, return_states)
    raise ValueError(f"no reference internals for {kind}")
