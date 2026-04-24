"""Double plasma mirror (DPM) contrast model.

In the Timmis 2026 paper, the single most influential knob for SHHG
efficiency is the **sub-picosecond laser contrast**, characterised by
the high-dynamic-range rise time ``t_HDR`` — the time it takes the pulse
intensity to climb from 10⁻⁶ to its peak. The paper shows that cutting
``t_HDR`` from 711 fs to 351 fs increases the observed harmonic yield by
several orders of magnitude at the same on-target intensity.

Under the hood, ``t_HDR`` sets how much time the leading edge of the
pulse has to hydrodynamically expand the plasma surface before the main
pulse arrives. Longer ``t_HDR`` → larger scale length ``L/λ`` → ROM /
BGP harmonic emission is suppressed.

This module turns a contrast description into a scale length used by the
denting and emission models. The mapping is phenomenological — the
underlying physics is a 1-D isothermal expansion with sound speed
``c_s ∼ √(Z k_B T_e / M_i)``, but realistic quantitative prediction
requires either a hydrodynamic code or fits to data. Here we provide a
simple, tunable fit that reproduces the paper's qualitative behaviour:

    L(t_HDR) = L₀ + (L_∞ − L₀) · (1 − exp(−(t_HDR − t₀)/τ))

with defaults chosen so that ``L(351 fs) ≈ 0.14 λ`` (paper's optimum)
and ``L(711 fs) ≈ 0.35 λ`` (over-expanded).

An independent prepulse (intensity ratio × delay) adds a baseline
expansion contribution ``L_pp = 2 c_s · delay`` where ``c_s`` rises with
prepulse intensity; this reproduces the "prepulse timing window" of
Fig. 2b, c of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Defaults tuned to the Timmis 2026 Gemini experiment.
L0_LAMBDA = 0.05          # baseline scale length (near-critical surface)
L_INF_LAMBDA = 0.45       # asymptotic scale length for very long t_HDR
T0_FS = 250.0             # below this t_HDR, plasma expansion hasn't kicked in
TAU_FS = 300.0            # expansion timescale

# Prepulse contribution — controls how fast ion sound expansion runs.
PREPULSE_REF_INTENSITY_REL = 1.0e-3  # fraction of main intensity at which C_s below applies
PREPULSE_CS_UM_PER_PS = 0.12         # ion-acoustic velocity at the reference intensity


@dataclass(frozen=True)
class ContrastInputs:
    """Flattened knobs that drive the scale-length model."""

    t_HDR_fs: float
    prepulse_intensity_rel: float = 0.0   # I_prepulse / I_main
    prepulse_delay_fs: float = 0.0        # positive = prepulse arrives before main
    wavelength_um: float = 0.8


def scale_length_from_thdr(t_HDR_fs: float) -> float:
    """Scale length [in units of λ] for the DPM ``t_HDR`` component alone."""
    if t_HDR_fs <= T0_FS:
        return L0_LAMBDA
    arg = (t_HDR_fs - T0_FS) / TAU_FS
    return L0_LAMBDA + (L_INF_LAMBDA - L0_LAMBDA) * (1.0 - math.exp(-arg))


def scale_length_from_prepulse(
    prepulse_intensity_rel: float,
    prepulse_delay_fs: float,
    wavelength_um: float = 0.8,
) -> float:
    """Scale length [in units of λ] added by an independent prepulse."""
    if prepulse_intensity_rel <= 0.0 or prepulse_delay_fs <= 0.0:
        return 0.0
    # Ion sound speed scales as √I for ablation-driven expansion at mildly
    # relativistic prepulses; take a simple linear model that is exact at
    # the reference intensity and falls to zero at zero.
    cs_um_per_ps = PREPULSE_CS_UM_PER_PS * math.sqrt(
        prepulse_intensity_rel / PREPULSE_REF_INTENSITY_REL
    )
    delta_um = cs_um_per_ps * (prepulse_delay_fs * 1e-3)
    return float(delta_um / wavelength_um)


def scale_length(inputs: ContrastInputs) -> float:
    """Total plasma scale length L/λ combining t_HDR and prepulse."""
    return (
        scale_length_from_thdr(inputs.t_HDR_fs)
        + scale_length_from_prepulse(
            inputs.prepulse_intensity_rel,
            inputs.prepulse_delay_fs,
            inputs.wavelength_um,
        )
    )


def optimum_prepulse_delay_window(
    intensity_w_per_cm2: float,
    wavelength_um: float = 0.8,
) -> tuple[float, float]:
    """Return the approximate (min, max) prepulse delay in fs for optimum SHHG.

    Paper's Fig. 2b, c: ~50 fs wide window near 3.6 × 10²⁰ W/cm²; ~200 fs wide
    window at 9.3 × 10²⁰ W/cm². We fit that trend loosely.
    """
    # Optimum shifts to shorter delay as intensity grows.
    if intensity_w_per_cm2 <= 0:
        return (0.0, 0.0)
    centre = 2500.0 * (1e20 / intensity_w_per_cm2) ** 0.5
    # Window width grows with intensity (more room in the roll-over regime).
    width = 50.0 + 200.0 * max(0.0, math.log10(intensity_w_per_cm2 / 1e20))
    return (max(0.0, centre - 0.5 * width), centre + 0.5 * width)
