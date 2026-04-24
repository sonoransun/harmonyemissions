"""Coherent Harmonic Focus (CHF) — spatiotemporal compression of SHHG."""

from harmonyemissions.chf.gain import (
    ChfGainBreakdown,
    extrapolate_3d_gain,
    scaling_I_chf_over_I,
)
from harmonyemissions.chf.propagation import (
    apply_denting_phase,
    harmonic_far_field,
    harmonic_near_field,
)

__all__ = [
    "ChfGainBreakdown",
    "apply_denting_phase",
    "extrapolate_3d_gain",
    "harmonic_far_field",
    "harmonic_near_field",
    "scaling_I_chf_over_I",
]
