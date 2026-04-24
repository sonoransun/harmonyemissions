"""Ponderomotive denting of the plasma surface.

Implements the analytical model of Vincenti et al. (2014) and the
extension used in Timmis 2026 (eqs. 10–12 of that paper).

**Ion dent** — cumulative motion of the ion front under radiation
pressure on an exponential preplasma of scale length ``L``:

    Δz_i(x', y') = 2 L · ln(1 + Π₀/(2 L cos θ) · ∫_{-∞}^{t_p} a_L(x', y', t') dt')

with

    Π₀ = √(R · Z · m_e · cos θ / (2 · A · M_p))

where ``R`` is the reflectivity of the relativistic plasma mirror, ``Z``
and ``A`` are the ion charge state and mass number, ``m_e`` and ``M_p``
are the electron and proton masses, and θ is the angle of incidence.

**Electron excursion** — short-timescale surface oscillation. At the
peak of the laser cycle, electrons pile up over a length ``~a₀/(k_L γ)``,
giving a much smaller contribution than the ion dent for most of the
pulse. We use the standard relativistic-similarity scaling.

**Phase imprint** — the total dent Δz(x', y') imprints a phase

    φ_n(x', y') = 2 k_n · Δz(x', y') · cos θ

on the nth harmonic's far-field amplitude, which is applied in
:func:`harmonyemissions.chf.propagation.apply_denting_phase`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Physical constants (SI).
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27


@dataclass(frozen=True)
class DentingInputs:
    """Parameters controlling the denting calculation."""

    scale_length_lambda: float       # L / λ
    angle_deg: float = 45.0          # θ
    z_ion: int = 14                  # Si
    a_ion: int = 28                  # Si
    reflectivity: float = 0.6        # R — from PIC; decreases slightly with intensity
    oxygen_admixture: float = 2.0 / 3.0  # SiO₂ has 2× more O than Si — averaged ion
    wavelength_um: float = 0.8


def _effective_za(inputs: DentingInputs) -> tuple[float, float]:
    """Effective (Z, A) for SiO₂ averaged over Si and O."""
    # Fused silica unit: 1 Si (Z=14, A=28) + 2 O (Z=8, A=16) = average
    # Z_avg = (1*14 + 2*8)/3 = 10, A_avg = (1*28 + 2*16)/3 = 20.
    if abs(inputs.oxygen_admixture) < 1e-12:
        return float(inputs.z_ion), float(inputs.a_ion)
    z = inputs.z_ion + 2.0 * 8.0
    a = inputs.a_ion + 2.0 * 16.0
    return z / 3.0, a / 3.0


def pi0(inputs: DentingInputs) -> float:
    """Dimensionless Π₀ = √(R · Z · m_e · cos θ / (2 · A · M_p))."""
    z, a = _effective_za(inputs)
    cos_theta = math.cos(math.radians(inputs.angle_deg))
    return math.sqrt(inputs.reflectivity * z * M_E * cos_theta / (2.0 * a * M_P))


def dent_depth_ion(
    fluence: np.ndarray,
    inputs: DentingInputs,
) -> np.ndarray:
    """Return Δz_i (ion dent depth) in units of λ.

    ``fluence(x', y') = ∫ a_L(x', y', t') dt'`` is the time-integrated
    normalized vector potential at each transverse location, in units
    of the laser period T₀ (so values ~ a₀ · τ/T₀ for a pulse of
    τ cycles).
    """
    L = inputs.scale_length_lambda
    if L <= 0:
        return np.zeros_like(fluence)
    cos_theta = math.cos(math.radians(inputs.angle_deg))
    prefactor = pi0(inputs) / (2.0 * L * cos_theta)
    return 2.0 * L * np.log(np.maximum(1.0 + prefactor * fluence, 1e-30))


def dent_depth_electron(
    a0_map: np.ndarray,
    inputs: DentingInputs,
) -> np.ndarray:
    """Return Δz_e (electron excursion) in units of λ.

    Relativistic-similarity scaling: Δz_e / λ ≈ a₀ / (2 π γ), with
    γ = √(1 + a₀²/2). Much smaller than the ion dent for a₀ ≫ 1.
    """
    a0 = np.asarray(a0_map, dtype=float)
    gamma = np.sqrt(1.0 + 0.5 * a0 ** 2)
    return a0 / (2.0 * math.pi * np.maximum(gamma, 1e-9))


def dent_map(
    a0_peak_map: np.ndarray,
    duration_T0: float,
    inputs: DentingInputs,
) -> np.ndarray:
    """Combined Δz(x', y') = Δz_i + Δz_e in units of λ.

    ``a0_peak_map`` is the peak normalized vector potential at each
    transverse point; ``duration_T0`` is the Gaussian intensity FWHM in
    units of T₀. The integrated ``|a_L|`` over the pulse is
    ``a₀ · √(π/(4 ln 2)) · duration_T0`` (standard Gaussian integral).
    """
    a0 = np.asarray(a0_peak_map, dtype=float)
    # For a Gaussian envelope a(t) = a0 exp(−t²/2σ²) with sigma = FWHM/(2√ln2),
    # ∫ |a| dt = a0 · σ · √(π/2); but the paper's eq. 10 integrates a_L to t_p
    # (peak time) not symmetric — so we take half of the symmetric integral.
    duration_integral = 0.5 * a0 * math.sqrt(math.pi / 2.0) * (
        duration_T0 / (2.0 * math.sqrt(math.log(2.0)))
    )
    dzi = dent_depth_ion(duration_integral, inputs)
    dze = dent_depth_electron(a0, inputs)
    return dzi + dze


def denting_phase(dent_map_lambda: np.ndarray, harmonic_n: int, angle_deg: float) -> np.ndarray:
    """Phase φ_n(x', y') = 2 k_n Δz cos θ (eq. 12 of the paper).

    Returns radians. k_n = n · k_L → in units of λ: k_n · Δz = 2π n · (Δz/λ).
    """
    cos_theta = math.cos(math.radians(angle_deg))
    return 2.0 * (2.0 * math.pi * harmonic_n) * dent_map_lambda * cos_theta
