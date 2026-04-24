"""Emission primitives used by the unified SHHG pipeline."""

from harmonyemissions.emission.spikes import (
    CutoffMode,
    relativistic_spikes_filter,
    spikes_cutoff_harmonic,
    universal_envelope,
)

__all__ = [
    "CutoffMode",
    "relativistic_spikes_filter",
    "spikes_cutoff_harmonic",
    "universal_envelope",
]
