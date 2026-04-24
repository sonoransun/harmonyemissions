"""Inverse Compton / Thomson scattering primitives.

Two regimes:

- **Thomson / linear-Compton** (a₀ ≪ 1, χ_e ≪ 1): classical cross section
  σ_T = 8π r_e²/3, scattered-photon energy in head-on collision

      E_γ_max = 4 γ² · E_laser / (1 + 4 γ · E_laser / m_e c²).

- **Nonlinear / Klein–Nishina** (χ_e ~ 1): full KN total cross section
  ``σ_KN(x) = 2π r_e² · [ (1-4/x-8/x²)·ln(1+x) + 1/2 + 8/x - 1/(2(1+x)²) ]/x``
  with ``x = 2 γ ħω / m_e c²``; converges to σ_T as x → 0.

For ICS, the on-axis spectrum peaks near ``E_γ_max``; we model the
useful part with a sharp-edge approximation (a flat top up to
``E_γ_max``, falling like 1/E beyond due to collection angle) rather
than the full double-differential Klein–Nishina.  That shape is
representative of published γ-spectra from LWFA-ICS experiments
(Khrennikov et al., PRL 114, 195003, 2015; Sarri et al., PRL 113,
224801, 2014).
"""

from __future__ import annotations

import numpy as np

from harmonyemissions.units import (
    CLASSICAL_ELECTRON_RADIUS_M,
    HC_KEV_NM,
    M_E_C2_KEV,
)

# σ_T = 8π/3 · r_e²  in m²  (≈ 6.6524587e-29 m²).
THOMSON_CROSS_SECTION_M2 = (8.0 / 3.0) * np.pi * CLASSICAL_ELECTRON_RADIUS_M**2


def thomson_cross_section() -> float:
    """Classical Thomson cross section σ_T [m²]."""
    return float(THOMSON_CROSS_SECTION_M2)


def klein_nishina_total_cross_section(photon_energy_keV: float | np.ndarray) -> np.ndarray:
    """Klein–Nishina total cross section σ_KN(E_γ) [m²] in the electron rest frame.

    ``photon_energy_keV`` is the photon energy as seen by the electron; in
    an ICS head-on geometry this is ``2 γ · E_laser`` (already
    Doppler-boosted into the rest frame).
    """
    x = np.asarray(photon_energy_keV, dtype=float) / M_E_C2_KEV
    # Handle the x→0 limit explicitly (σ_KN → σ_T) to avoid 0/0.
    safe = np.where(x > 1e-12, x, 1e-12)
    term = (
        (1.0 - 4.0 / safe - 8.0 / safe**2) * np.log1p(safe)
        + 0.5
        + 8.0 / safe
        - 1.0 / (2.0 * (1.0 + safe) ** 2)
    )
    sigma = 2.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_M**2 * term / safe
    # At x ≪ 1 the analytic expansion is σ_T · (1 − x + ...).  Substitute σ_T
    # for extreme low x to dodge numerical cancellation.
    return np.where(x < 1e-4, THOMSON_CROSS_SECTION_M2, sigma)


def compton_max_photon_energy_keV(
    gamma_e: float, laser_wavelength_um: float, head_on: bool = True,
) -> float:
    """Maximum backscattered-photon energy [keV] for a head-on ICS collision.

    ``E_γ_max = 4 γ² · E_L / (1 + 4 γ · E_L / m_e c²)`` in head-on geometry
    (factor halves to ``2 γ² · E_L / (1 + 2 γ · E_L / m_e c²)`` at 90°, which
    we approximate with ``head_on=False``).
    """
    E_L_keV = HC_KEV_NM * 1e-3 / laser_wavelength_um  # laser photon energy
    factor = 4.0 if head_on else 2.0
    denom = 1.0 + factor * gamma_e * E_L_keV / M_E_C2_KEV
    return float(factor * gamma_e**2 * E_L_keV / denom)


def ics_photon_spectrum_keV(
    gamma_e: float,
    laser_wavelength_um: float,
    energy_keV: np.ndarray | None = None,
    head_on: bool = True,
    recoil_slope: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """On-axis ICS photon spectrum dN/dE over a user-supplied energy grid.

    Uses a sharp-edge approximation: plateau up to ``E_γ_max``, falling
    like ``(E_γ_max / E)^recoil_slope`` at higher energies (collection
    angle × finite beam emittance).  Good to an order of magnitude
    against published LWFA-ICS spectra (Sarri 2014, Khrennikov 2015).
    Returns ``(energy_keV, dNdE)``.
    """
    E_max = compton_max_photon_energy_keV(gamma_e, laser_wavelength_um, head_on)
    if energy_keV is None:
        from harmonyemissions.units import default_xray_energy_grid

        energy_keV = default_xray_energy_grid(
            E_min_keV=max(0.1, E_max * 1e-4),
            E_max_keV=max(10.0, E_max * 10.0),
            n_points=2048,
        )
    E = np.asarray(energy_keV, dtype=float)
    plateau = np.minimum(1.0, (E_max / np.maximum(E, 1e-12)) ** recoil_slope)
    spectrum = np.where(E <= E_max, 1.0, plateau)
    # Divide by E to get dN/dE (photon number density per unit energy) that
    # integrates to a reasonable number; the absolute normalisation is
    # anchored by the Thomson cross section × beam / photon flux in the
    # model layer, not here.
    return E, spectrum / np.maximum(E, 1e-12)
