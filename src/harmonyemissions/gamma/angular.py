"""Angular emission helpers — divergence, collimation, brightness.

For relativistic single-electron synchrotron / ICS / bremsstrahlung
sources the emission is tightly forward-collimated into a cone of
half-angle ≈ 1/γ with some refinement for the transverse oscillation
amplitude K.
"""

from __future__ import annotations

import math

import numpy as np


def divergence_cone_fwhm_mrad(gamma_e: float, K_wiggler: float | None = None) -> float:
    """Full-width at half-max of the on-axis emission cone [mrad].

    Synchrotron limit (K ≲ 1): θ_FWHM ≈ 1/γ.
    Wiggler limit (K ≫ 1): θ_FWHM ≈ K/γ (the oscillation amplitude
    dominates the instantaneous emission direction).
    """
    g = max(float(gamma_e), 1e-6)
    if K_wiggler is None or K_wiggler <= 1.0:
        return 1000.0 / g
    return 1000.0 * float(K_wiggler) / g


def synchrotron_angular_pattern(
    theta_mrad: np.ndarray, gamma_e: float, harmonic_n: float = 1.0
) -> np.ndarray:
    """Normalised angular intensity pattern of synchrotron-like emission.

    For an electron radiating at the nth harmonic of its transverse
    motion, the angular distribution perpendicular to the orbit plane
    is close to ``exp(−(γ θ)² / 2 n)`` (small-angle Gaussian approx).
    Returns the normalised shape (peak = 1 at θ = 0).
    """
    theta = np.asarray(theta_mrad, dtype=float) * 1e-3  # → radians
    g = max(float(gamma_e), 1e-6)
    width = max(float(harmonic_n), 1e-3)
    return np.exp(-0.5 * (g * theta) ** 2 / width)


def brightness_estimate(
    n_photons_per_pulse: float,
    divergence_mrad: float,
    bandwidth_percent: float = 0.1,
    source_size_um: float = 1.0,
    rep_rate_hz: float = 1.0,
) -> float:
    """Peak brightness [photons / s / mm² / mrad² / 0.1 %BW] (approximate).

    Uses the standard accelerator-physics definition

        B ≈ N_γ / (t · σ_x σ_y · σ_θx σ_θy · 0.1 %BW)

    where we approximate σ_x σ_y ≈ (source_size_um × 1e-3 mm)²,
    σ_θx σ_θy ≈ divergence_mrad², and use ``t = 1/rep_rate`` as the
    pulse-length proxy so the numbers match published "peak"
    quantities.  Divide by 1000 for average brightness at the same
    charge per shot.
    """
    solid_angle_mrad2 = max(float(divergence_mrad), 1e-6) ** 2
    source_area_mm2 = max(float(source_size_um) * 1e-3, 1e-6) ** 2
    bandwidth = max(float(bandwidth_percent) / 0.1, 1e-6)
    return n_photons_per_pulse * float(rep_rate_hz) / (
        source_area_mm2 * solid_angle_mrad2 * bandwidth
    )


def collimation_factor(theta_opening_mrad: float, gamma_e: float) -> float:
    """Fraction of emission captured by an aperture of opening angle θ.

    Uses the Gaussian approximation ``F ≈ 1 − exp(−(γ θ)²/2)``.
    """
    g_theta = float(gamma_e) * float(theta_opening_mrad) * 1e-3
    return float(1.0 - math.exp(-0.5 * g_theta * g_theta))
