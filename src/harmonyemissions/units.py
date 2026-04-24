"""Unit conversions for relativistic laser-plasma interactions.

Conventions
-----------
- SI for user-facing inputs (wavelength in μm, duration in fs, intensity in W/cm²).
- Normalized Gaussian-CGS-like units for the internals:
    * time  → laser period T₀ = λ/c
    * space → laser wavelength λ
    * field → a = eE/(mₑωc)  (dimensionless vector potential)
    * density → critical density n_c(ω) = ε₀ mₑ ω² / e²

`a₀` (peak normalized vector potential) is the key knob — the interaction is
relativistic when a₀ ≳ 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (SI).
C = 2.99792458e8  # m/s
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg
EPS0 = 8.8541878128e-12  # F/m
H_PLANCK = 6.62607015e-34  # J·s
HBAR = H_PLANCK / (2 * math.pi)
M_E_C2_KEV = 510.998950  # electron rest mass energy in keV
CLASSICAL_ELECTRON_RADIUS_M = 2.8179403262e-15  # m
FINE_STRUCTURE_ALPHA = 7.2973525693e-3  # ≈ 1/137.036
HC_KEV_NM = 1.2398419843320025  # ħc product: E[keV] · λ[nm] = HC_KEV_NM


def omega_from_wavelength(wavelength_m: float) -> float:
    """Angular frequency [rad/s] from free-space wavelength [m]."""
    return 2 * math.pi * C / wavelength_m


def critical_density(omega: float) -> float:
    """Critical density n_c [m⁻³] at which ω_p = ω."""
    return EPS0 * M_E * omega**2 / E_CHARGE**2


def plasma_frequency(n_e: float) -> float:
    """Angular plasma frequency ω_p [rad/s] for electron density n_e [m⁻³]."""
    return math.sqrt(n_e * E_CHARGE**2 / (EPS0 * M_E))


def intensity_to_a0(intensity_w_per_cm2: float, wavelength_um: float) -> float:
    """Convert peak intensity [W/cm²] and wavelength [μm] to normalized a₀.

    Uses the linearly-polarized convention a₀² = I[W/cm²] · (λ[μm])² / 1.37e18.
    """
    return math.sqrt(intensity_w_per_cm2 * wavelength_um**2 / 1.37e18)


def a0_to_intensity(a0: float, wavelength_um: float) -> float:
    """Inverse of :func:`intensity_to_a0`."""
    return a0**2 * 1.37e18 / wavelength_um**2


def gamma_from_a0(a0: float, polarization: str = "linear") -> float:
    """Peak Lorentz factor of a quiver electron in a plane wave.

    Linearly polarized: γ = √(1 + a₀²/2).
    Circularly polarized: γ = √(1 + a₀²).
    """
    if polarization == "linear":
        return math.sqrt(1.0 + 0.5 * a0**2)
    if polarization == "circular":
        return math.sqrt(1.0 + a0**2)
    raise ValueError(f"unknown polarization: {polarization!r}")


def ponderomotive_energy_ev(intensity_w_per_cm2: float, wavelength_um: float) -> float:
    """Ponderomotive energy U_p [eV] in the non-relativistic limit.

    U_p [eV] ≈ 9.33e-14 · I[W/cm²] · (λ[μm])².
    """
    return 9.33e-14 * intensity_w_per_cm2 * wavelength_um**2


def keV_per_harmonic(wavelength_um: float) -> float:
    """Photon energy [keV] per unit harmonic order for carrier wavelength [μm]."""
    return HC_KEV_NM * 1e-3 / wavelength_um


def photon_energy_keV_from_harmonic(harmonic, wavelength_um: float):
    """Vectorised: E[keV] = n · (hc/λ) for harmonic order n and carrier λ [μm]."""
    import numpy as np
    return np.asarray(harmonic, dtype=float) * keV_per_harmonic(wavelength_um)


def photon_energy_keV_from_omega(omega_rad_s: float) -> float:
    """E[keV] from angular frequency [rad/s]."""
    return HBAR * omega_rad_s / E_CHARGE * 1e-3


def hot_electron_temperature_keV(
    a0: float, wavelength_um: float, scaling: str = "wilks"
) -> float:
    """Hot-electron temperature [keV] from laser ponderomotive drive.

    "wilks": T = m_ec² · (√(1 + a₀²/2) − 1) — canonical Wilks ponderomotive form.
    "beg":   T = 215 · (I₁₈·λ²_μm)^{1/3} keV — Beg 1997 fit, valid a₀ ≳ 3.
    """
    if scaling == "wilks":
        return M_E_C2_KEV * (math.sqrt(1.0 + 0.5 * a0 * a0) - 1.0)
    if scaling == "beg":
        intensity_18 = a0_to_intensity(a0, wavelength_um) / 1.0e18
        return 215.0 * (intensity_18 * wavelength_um**2) ** (1.0 / 3.0)
    raise ValueError(f"unknown hot-electron scaling: {scaling!r}")


def default_xray_energy_grid(
    E_min_keV: float = 0.1, E_max_keV: float = 1.0e4, n_points: int = 4096
):
    """Canonical log-spaced photon-energy grid [keV] shared across hard-X-ray models."""
    import numpy as np
    return np.geomspace(E_min_keV, E_max_keV, n_points)


@dataclass(frozen=True)
class LaserUnits:
    """Convenience bundle: SI inputs together with normalized derived quantities."""

    wavelength_m: float
    omega: float  # rad/s
    period_s: float
    k0: float  # rad/m
    critical_density: float  # m⁻³

    @classmethod
    def from_wavelength_um(cls, wavelength_um: float) -> LaserUnits:
        wavelength_m = wavelength_um * 1e-6
        omega = omega_from_wavelength(wavelength_m)
        return cls(
            wavelength_m=wavelength_m,
            omega=omega,
            period_s=2 * math.pi / omega,
            k0=omega / C,
            critical_density=critical_density(omega),
        )
