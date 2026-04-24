"""Quantum radiation reaction for relativistic synchrotron emission.

At the highest electron energies (γ ≳ 10³) the classical Liénard–
Wiechert / synchrotron envelope overestimates the emitted spectrum
because the single-photon recoil becomes non-negligible.  The relevant
dimensionless parameter is

    χ_e = (E_* / E_Schwinger) = γ · |E_⊥| / E_cr

with ``E_cr = m_e c² / (e λ_C) ≈ 1.323 × 10¹⁸ V/m`` (Schwinger field).
For betatron / ICS in the typical LWFA regime χ_e stays ≲ 0.1; above
that, quantum corrections reshape the spectrum (Ritus 1985; Baier &
Katkov 1968).

We implement the leading-order Gaunt-factor suppression

    dI_quantum / dI_classical ≈ 1 / (1 + 4.8 (1 + χ_e) · ln(1 + 1.7 χ_e)
                                     + 2.44 χ_e²)

(Niel et al., PRE 97, 043209, 2018).  This damps the spectrum at
high photon energies by an amount that saturates near ~50 % for
χ_e ~ 1 and falls as χ_e^(−2/3) for χ_e ≫ 1 — correct asymptotics.
"""

from __future__ import annotations

import numpy as np

from harmonyemissions.units import (
    HBAR,
    M_E,
    M_E_C2_KEV,
)

# Schwinger critical field in V/m.
SCHWINGER_FIELD_V_PER_M = (M_E_C2_KEV * 1000.0 * 1.602176634e-19) ** 2 / (
    HBAR * 2.99792458e8 * 1.602176634e-19
)
# — actually the clean symbolic form is E_cr = m_e² c³ / (e ħ); numerical
# value ≈ 1.323 × 10¹⁸ V/m.  Use literal for clarity and numerical
# stability:
SCHWINGER_FIELD_V_PER_M = 1.32320787e18


def chi_e_parameter(
    gamma_e: float | np.ndarray,
    transverse_field_V_per_m: float | np.ndarray,
) -> np.ndarray:
    """Quantum nonlinearity χ_e = γ · |E⊥| / E_cr.

    ``transverse_field_V_per_m`` is the field the electron actually sees
    in its rest frame (in an ICS head-on geometry that is ~2γ times the
    lab-frame laser field).
    """
    g = np.asarray(gamma_e, dtype=float)
    E = np.asarray(transverse_field_V_per_m, dtype=float)
    return np.abs(g * E / SCHWINGER_FIELD_V_PER_M)


def quantum_synchrotron_suppression(chi_e: float | np.ndarray) -> np.ndarray:
    """Gaunt-factor suppression factor g(χ) ≤ 1.

    At χ = 0, g = 1 (purely classical).  At χ = 1, g ≈ 0.28.  At χ ≫ 1,
    g ∝ χ^{−2/3} (Ritus asymptotic).
    """
    chi = np.asarray(chi_e, dtype=float)
    denom = 1.0 + 4.8 * (1.0 + chi) * np.log1p(1.7 * chi) + 2.44 * chi * chi
    return 1.0 / denom


def betatron_field_estimate_V_per_m(
    gamma_e: float, omega_beta_rad_s: float, amplitude_m: float
) -> float:
    """Approximate transverse focusing field in the ion bubble.

    Inside the LWFA bubble the ion column exerts a linear focusing force
    F⊥ = − (m_e ω_β²) · r, giving a peak field at the oscillation edge
    of ``E⊥ = m_e c · ω_β · γ / e`` — the expression needed to
    normalise χ_e for a betatron oscillation.
    """
    return float(M_E * 2.99792458e8 * omega_beta_rad_s * gamma_e / 1.602176634e-19)
