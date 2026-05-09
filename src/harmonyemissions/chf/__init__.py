"""Coherent Harmonic Focus (CHF) — spatiotemporal compression of SHHG.

The 2-D module surface (``gain``, ``propagation``) is the legacy
single-beam set used by every Timmis 2026 ROM-class run. The 3-D
``geometry`` / ``superposition`` / ``timing`` / ``modes`` modules
implement the chf3d Phase C kernel (homogeneous coherent multi-beam)
and the ω-grid resampling helpers needed by the Extreme-Power overlay.
"""

from harmonyemissions.chf.gain import (
    ChfGainBreakdown,
    extrapolate_3d_gain,
    predict_chf_intensity,
    scaling_I_chf_over_I,
)
from harmonyemissions.chf.geometry import (
    BeamArray,
    build_beam_array,
    from_record,
    to_record,
    with_delays,
    with_phases,
)
from harmonyemissions.chf.modes import (
    apply_structured_mode,
    azimuthal_vector_profile,
    bessel_profile,
    lg_profile,
    radial_vector_profile,
)
from harmonyemissions.chf.propagation import (
    apply_denting_phase,
    harmonic_far_field,
    harmonic_near_field,
    stack_harmonics_far_amplitude,
    stack_harmonics_far_field,
)
from harmonyemissions.chf.superposition import (
    FocalVolume,
    FocalVolumeAccumulator,
    build_omega_grid,
    coherent_sum_heterogeneous,
    coherent_sum_heterogeneous_diag,
    coherent_sum_homogeneous,
    resample_to_omega,
)
from harmonyemissions.chf.timing import (
    analytic_phase_lock,
    geometric_delays,
    optimise_phases,
)

__all__ = [
    "BeamArray",
    "ChfGainBreakdown",
    "FocalVolume",
    "FocalVolumeAccumulator",
    "analytic_phase_lock",
    "apply_denting_phase",
    "apply_structured_mode",
    "azimuthal_vector_profile",
    "bessel_profile",
    "build_beam_array",
    "build_omega_grid",
    "coherent_sum_heterogeneous",
    "coherent_sum_heterogeneous_diag",
    "coherent_sum_homogeneous",
    "extrapolate_3d_gain",
    "from_record",
    "geometric_delays",
    "harmonic_far_field",
    "harmonic_near_field",
    "lg_profile",
    "optimise_phases",
    "predict_chf_intensity",
    "radial_vector_profile",
    "resample_to_omega",
    "scaling_I_chf_over_I",
    "stack_harmonics_far_amplitude",
    "stack_harmonics_far_field",
    "to_record",
    "with_delays",
    "with_phases",
]
