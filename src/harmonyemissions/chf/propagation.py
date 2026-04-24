"""Near/far-field propagation of CHF harmonic beams (Timmis 2026 eqs. 7–12).

Given a driver near-field U₀(x', y') and a denting map Δz(x', y'), this
module builds the nth harmonic's:

- **Near-field amplitude** U₀(x', y', n) = √S(n, a₀(x', y')) · U₀(x', y')
  · exp(−2 i k_n Δz cos θ)
- **Far-field amplitude** U(x, y, z, n) via 2-D Fraunhofer propagation

which is the essence of equation 12 in the paper.
"""

from __future__ import annotations

import math

import numpy as np

from harmonyemissions.beam import fraunhofer, intensity
from harmonyemissions.emission.spikes import CutoffMode, relativistic_spikes_filter


def apply_denting_phase(
    u0: np.ndarray,
    dent_map_lambda: np.ndarray,
    harmonic_n: int,
    angle_deg: float,
) -> np.ndarray:
    """Multiply u0 by the curvature phase e^{−2 i k_n Δz cos θ} per pixel."""
    cos_theta = math.cos(math.radians(angle_deg))
    k_n = 2.0 * math.pi * harmonic_n  # in units of 1/λ
    phase = np.exp(-2.0j * k_n * dent_map_lambda * cos_theta)
    return u0 * phase


def harmonic_near_field(
    u0: np.ndarray,
    a0_map: np.ndarray,
    dent_map_lambda: np.ndarray,
    harmonic_n: int,
    angle_deg: float,
    cutoff_mode: CutoffMode = CutoffMode.LOGISTIC,
) -> np.ndarray:
    """Return U₀(x', y', n) = √S · U₀ · exp(−2 i k_n Δz cos θ)."""
    s = relativistic_spikes_filter(harmonic_n, a0_map, mode=cutoff_mode)
    amplitude = np.sqrt(np.maximum(s, 0.0)) * u0
    return apply_denting_phase(amplitude, dent_map_lambda, harmonic_n, angle_deg)


def harmonic_far_field(
    u0: np.ndarray,
    a0_map: np.ndarray,
    dent_map_lambda: np.ndarray,
    harmonic_n: int,
    angle_deg: float,
    dx: float,
    wavelength_m: float,
    focus_distance_m: float,
    cutoff_mode: CutoffMode = CutoffMode.LOGISTIC,
) -> tuple[np.ndarray, float]:
    """Far-field |U(x,y,z,n)|² and its pixel size.

    Returns ``(far_field_amplitude, dx_far)``; intensity is ``|far|²``.
    The wavelength of the nth harmonic is ``wavelength_m / n``.
    """
    u_near = harmonic_near_field(u0, a0_map, dent_map_lambda, harmonic_n, angle_deg, cutoff_mode)
    lambda_n = wavelength_m / harmonic_n
    u_far, dx_far = fraunhofer(u_near, dx, lambda_n, focus_distance_m)
    return u_far, dx_far


def stack_harmonics_far_field(
    u0: np.ndarray,
    a0_map: np.ndarray,
    dent_map_lambda: np.ndarray,
    harmonics: np.ndarray,
    angle_deg: float,
    dx: float,
    wavelength_m: float,
    focus_distance_m: float,
    cutoff_mode: CutoffMode = CutoffMode.LOGISTIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute |U(x,y,z,n)|² for every n in ``harmonics``.

    Returns ``(intensities_nxy, dx_far_per_n)`` where intensities_nxy has
    shape ``(len(harmonics), N, N)`` and dx_far_per_n has length
    ``len(harmonics)``.
    """
    harmonics = np.asarray(harmonics, dtype=int)
    n_grid = u0.shape[0]
    out = np.zeros((harmonics.size, n_grid, n_grid), dtype=float)
    dx_out = np.zeros(harmonics.size, dtype=float)
    for i, n in enumerate(harmonics):
        u_far, dx_far = harmonic_far_field(
            u0, a0_map, dent_map_lambda, int(n), angle_deg,
            dx, wavelength_m, focus_distance_m, cutoff_mode,
        )
        out[i] = intensity(u_far)
        dx_out[i] = dx_far
    return out, dx_out
