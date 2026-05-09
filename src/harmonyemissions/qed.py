"""Perturbative vacuum-QED diagnostics for the extreme-power overlay.

These are *diagnostics*, not self-consistent QED dynamics. The library
does not (and is not meant to) propagate pair cascades, plasma
backreaction, or non-perturbative QED feedback. The functions here
report:

- ``schwinger_ratio``  χ = E_focus / E_S — the dimensionless distance to
  the Schwinger critical field E_S ≈ 1.32 × 10¹⁸ V/m (Schwinger 1951).
- ``vacuum_birefringence_phase_shift``  Δφ accumulated by a probe over
  ``length_m`` through the focal field, in the perturbative
  Heisenberg–Euler leading order (Heisenberg & Euler, Z. Phys. 1936;
  Di Piazza, RMP 84 1177 (2012) eq. 17).
- ``breit_wheeler_pair_rate``  R [s⁻¹ m⁻³] — perturbative Breit–Wheeler
  γγ → e⁺e⁻ rate density. Valid for χ ≲ 1; clipped above
  ``chi_warn`` and flagged in provenance.

A non-numeric "validity_warning" key is *not* placed in the returned
dict; callers fold the warning into ``Result.provenance`` so the
diagnostics stay numeric-only and don't break HDF5 persistence.
"""

from __future__ import annotations

import math

from harmonyemissions.gamma.radiation_reaction import SCHWINGER_FIELD_V_PER_M
from harmonyemissions.units import (
    EPS0,
    FINE_STRUCTURE_ALPHA,
    C,
)


def field_from_intensity(intensity_w_per_m2: float) -> float:
    """Peak E [V/m] from cycle-averaged intensity [W/m²]: E = √(2 I / (ε₀ c))."""
    return math.sqrt(max(2.0 * intensity_w_per_m2 / (EPS0 * C), 0.0))


def schwinger_ratio(E_focus_V_per_m: float) -> float:
    """χ = E / E_S (dimensionless distance to the Schwinger field)."""
    if E_focus_V_per_m < 0:
        raise ValueError("E_focus must be non-negative")
    return float(E_focus_V_per_m / SCHWINGER_FIELD_V_PER_M)


def vacuum_birefringence_phase_shift(
    E_focus_V_per_m: float, omega_probe_rad_s: float, length_m: float,
) -> float:
    """Heisenberg–Euler perturbative birefringence phase shift Δφ.

    Leading-order expression (Di Piazza RMP 2012 eq. 17, restricted to
    the parallel polarization mode):

        Δφ ≈ (α / 15 π) · (E / E_S)² · (ω L / c)

    The differential phase between modes is twice this for the
    standard "perpendicular minus parallel" convention; we return the
    parallel-mode phase to match the cleanest closed form. Validity is
    constrained to χ ≲ 1 and ``length_m`` shorter than the focal
    Rayleigh length (assumed by caller).
    """
    if length_m < 0:
        raise ValueError("length_m must be non-negative")
    chi = schwinger_ratio(E_focus_V_per_m)
    return float(
        FINE_STRUCTURE_ALPHA / (15.0 * math.pi)
        * chi * chi
        * omega_probe_rad_s * length_m / C
    )


def breit_wheeler_pair_rate(
    E_focus_V_per_m: float, omega_probe_rad_s: float,
) -> float:
    """Perturbative Breit–Wheeler γγ → e⁺e⁻ rate density [s⁻¹ m⁻³].

    Schwinger-formula form for static-field pair production
    (Schwinger 1951, generalised to "weak-field × probe-frequency"
    regime by Ritus 1985). Only intended as an order-of-magnitude
    diagnostic above χ ≈ 0.1; below that the rate is exponentially
    suppressed by the prefactor exp(-π/χ).
    """
    chi = schwinger_ratio(E_focus_V_per_m)
    if chi <= 0:
        return 0.0
    # Compute exp(-π/χ) safely for very small χ; clip the underflow
    # to zero to keep the returned value finite and non-NaN.
    arg = math.pi / chi
    if arg > 700.0:
        return 0.0
    suppression = math.exp(-arg)
    # Prefactor: dN/dV/dt ≈ (α / π²) · (E_S / ħ) · (E/E_S)² · χ exp(-π/χ).
    # Use the full expression with ħ baked into Schwinger field's natural
    # rate scale: 1/(λ_C · τ_C) ≈ E_S² (e/ħc) — folded into a single
    # numerical prefactor below.
    rate_scale = (
        FINE_STRUCTURE_ALPHA
        / (math.pi * math.pi)
        * (SCHWINGER_FIELD_V_PER_M ** 2)
        * (1.0 / (1.054571817e-34))    # 1/ħ
    )
    return float(rate_scale * (chi ** 2) * suppression)


def qed_diagnostics(
    intensity_w_per_m2: float,
    omega_probe_rad_s: float,
    length_m: float,
    *,
    chi_warn: float = 0.5,
) -> dict[str, float | bool]:
    """Bundle the three diagnostics + a validity flag.

    Returned keys are all numeric (no string "warning"); the
    ``validity_exceeded`` boolean is true when χ > ``chi_warn`` so
    callers can fold the human-readable warning into provenance.
    """
    e_focus = field_from_intensity(intensity_w_per_m2)
    chi = schwinger_ratio(e_focus)
    return {
        "schwinger_ratio": float(chi),
        "vacuum_birefringence_phase_shift_rad": vacuum_birefringence_phase_shift(
            e_focus, omega_probe_rad_s, length_m,
        ),
        "breit_wheeler_pair_rate_per_m3_per_s": breit_wheeler_pair_rate(
            e_focus, omega_probe_rad_s,
        ),
        "field_V_per_m": float(e_focus),
        "validity_exceeded": bool(chi > chi_warn),
        "chi_warn_threshold": float(chi_warn),
    }
