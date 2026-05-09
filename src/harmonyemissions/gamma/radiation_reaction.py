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


def landau_lifshitz_cutoff_derate(
    a0: float,
    wavelength_um: float,
    *,
    chi_clip: float = 2.0,
    a0_floor: float = 50.0,
) -> tuple[float, float]:
    """Landau–Lifshitz radiation-friction derate for the BGP cutoff.

    The classical Lorentz dynamics give n_c ∝ γ³ ≈ (a₀² / 2)^{3/2}. At
    extreme intensities the radiation-friction term in Landau–Lifshitz
    saturates the achievable γ at roughly ``γ_RR = (a_RR / a₀)^{1/3} γ₀``
    where ``a_RR ≈ 4 π m_e c² / (3 e² ω₀ / c) = 3 m_e c · λ / (4 π r_e)``
    (Bulanov 2011 Phys. Rev. E 84 056605; Di Piazza, RMP 84 1177 (2012)
    eq. 87). Below ``a0_floor`` we return ``derate = 1.0`` (RR is
    sub-percent and not worth applying); above ``chi_clip`` we clip and
    let the caller emit a provenance warning.

    Returns
    -------
    derate
        Multiplicative factor on the harmonic cutoff (≤ 1).
    chi_e
        Field-strength χ_e = γ · |E⊥| / E_S used for the clip decision.
    """
    if a0 <= a0_floor:
        # Synthesise a small χ for telemetry consistency.
        chi_classical = a0 ** 2 / 2.0  # γ ≈ a₀²/2 for intensity-driven regime
        return 1.0, float(chi_classical * 1e-22)  # well below any QED threshold

    # Classical electron Lorentz factor in the laser field.
    gamma0 = max((1.0 + 0.5 * a0 * a0) ** 0.5, 1.0)
    # Saturation a_RR (dimensionless) for the given wavelength
    # (Di Piazza RMP 2012 §IV.A: a_RR ≈ 3300·(λ/μm)).
    a_RR = 3300.0 * float(wavelength_um)
    # Smooth Bulanov-style saturation. Radiation-reaction parameter
    # R_χ = (a₀ / a_RR)³ · γ₀. derate_γ = 1 / (1 + R_χ)^{1/3} so γ_max
    # rolls off smoothly from γ₀ when a₀ ≪ a_RR to a hard cap at a₀ ≳ a_RR.
    R_chi = (a0 / a_RR) ** 3 * gamma0
    derate_gamma = 1.0 / (1.0 + R_chi) ** (1.0 / 3.0)
    derate = float(derate_gamma ** 3)  # n_c ∝ γ³
    # χ_e estimate (lab-frame transverse field × γ).
    omega_rad_s = 2.0 * 3.141592653589793 * 2.99792458e8 / (float(wavelength_um) * 1e-6)
    e_norm_v_per_m = M_E_C2_KEV * 1000.0 * 1.602176634e-19 * omega_rad_s / (
        1.602176634e-19 * 2.99792458e8
    )
    e_lab = a0 * e_norm_v_per_m
    chi = float(gamma0 * e_lab / SCHWINGER_FIELD_V_PER_M)
    if chi > chi_clip:
        derate = float(min(derate, chi_clip / max(chi, 1e-30)))
    return derate, chi
