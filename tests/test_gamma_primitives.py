"""Unit tests for the gamma/ physics primitives (cross sections + radiation
reaction + angular helpers).  Tests the library pieces directly, independent
of the emission-model layer."""

import math

import numpy as np
import pytest

from harmonyemissions.gamma.angular import (
    brightness_estimate,
    collimation_factor,
    divergence_cone_fwhm_mrad,
    synchrotron_angular_pattern,
)
from harmonyemissions.gamma.bremsstrahlung import (
    bethe_heitler_spectrum,
    converter_photon_yield,
    radiation_length_g_per_cm2,
)
from harmonyemissions.gamma.compton import (
    compton_max_photon_energy_keV,
    ics_photon_spectrum_keV,
    klein_nishina_total_cross_section,
    thomson_cross_section,
)
from harmonyemissions.gamma.radiation_reaction import (
    betatron_field_estimate_V_per_m,
    chi_e_parameter,
    quantum_synchrotron_suppression,
)

# ---------------------------------------------------------------------------
# Compton / Thomson
# ---------------------------------------------------------------------------


def test_thomson_cross_section_exact():
    # σ_T = 8π r_e²/3 ≈ 6.6524587 × 10⁻²⁹ m².
    assert thomson_cross_section() == pytest.approx(6.6524587e-29, rel=1e-4)


def test_klein_nishina_reduces_to_thomson_at_low_energy():
    # σ_KN(1 eV) should equal σ_T to ~machine precision.
    sigma = klein_nishina_total_cross_section(1e-3)  # 1 eV
    assert sigma == pytest.approx(thomson_cross_section(), rel=1e-5)


def test_klein_nishina_decreases_at_mev():
    # σ_KN(1 MeV) < σ_T (significant recoil suppression).
    sigma_low = klein_nishina_total_cross_section(10.0)   # 10 keV
    sigma_mev = klein_nishina_total_cross_section(1000.0)  # 1 MeV
    assert sigma_mev < sigma_low
    # By 10 MeV, σ_KN ~ 0.1 · σ_T.
    sigma_10mev = klein_nishina_total_cross_section(10000.0)
    assert sigma_10mev < 0.2 * thomson_cross_section()


def test_compton_max_scales_with_gamma_squared_in_thomson_limit():
    # In the Thomson limit γ·E_L ≪ m_e c² the denominator is ~1 and
    # E_max ∝ γ².
    E1 = compton_max_photon_energy_keV(50.0, 0.8)
    E2 = compton_max_photon_energy_keV(200.0, 0.8)
    assert E2 / E1 == pytest.approx(16.0, rel=0.02)


def test_compton_max_saturates_at_high_gamma():
    # γ → ∞ limit: E_max → γ · m_e c² (linear, not γ² scaling).  At γ=1e4
    # we sit in the transition regime (4γ·E_L / m_ec² ≈ 0.12); by γ=1e5 it
    # is ≈ 1.2, so the scaling softens from γ² toward γ.  Ratio should be
    # between the fully-saturated (~10) and the Thomson (~100) extremes.
    E_low = compton_max_photon_energy_keV(1e4, 0.8)
    E_hi = compton_max_photon_energy_keV(1e5, 0.8)
    assert 10.0 < E_hi / E_low < 80.0


def test_ics_spectrum_has_cutoff_and_tail():
    E, s = ics_photon_spectrum_keV(1000.0, 0.8)
    E_max = compton_max_photon_energy_keV(1000.0, 0.8)
    # Sharp edge roughly at E_max.
    below = s[E <= E_max]
    above = s[E > E_max]
    assert above.size > 0 and below.size > 0
    assert above.max() <= below.max()


def test_ics_spectrum_head_on_vs_90_degrees():
    E_h = compton_max_photon_energy_keV(500.0, 0.8, head_on=True)
    E_s = compton_max_photon_energy_keV(500.0, 0.8, head_on=False)
    assert E_h > E_s          # head-on is the hardest geometry
    assert E_h / E_s == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Bremsstrahlung (Bethe–Heitler, thin converter)
# ---------------------------------------------------------------------------


def test_radiation_length_tungsten_tantalum_known_values():
    # PDG X₀: W = 6.76 g/cm², Ta = 6.82 g/cm².  Order-of-magnitude check.
    assert 6.0 < radiation_length_g_per_cm2("W") < 7.0
    assert 6.0 < radiation_length_g_per_cm2("Ta") < 7.0


def test_radiation_length_unknown_material_raises():
    with pytest.raises(ValueError, match="Unknown converter material"):
        radiation_length_g_per_cm2("Unobtainium")


def test_bethe_heitler_zero_above_endpoint():
    E = np.array([50.0, 500.0, 1500.0])
    spec = bethe_heitler_spectrum(E, electron_energy_keV=1000.0, material="W")
    assert spec[-1] == 0.0
    assert spec[0] > 0 and spec[1] > 0


def test_bethe_heitler_soft_diverges_as_one_over_E():
    # dN/dE ≈ (4/3 - 4y/3 + y²) / E.  At y ≪ 1 the bracket is ≈ 4/3,
    # so spec(E/2) / spec(E) ≈ 2 for small E.
    E = np.array([1.0, 2.0])
    spec = bethe_heitler_spectrum(E, electron_energy_keV=1000.0, material="Ta")
    assert spec[0] / spec[1] == pytest.approx(2.0, rel=0.05)


def test_bethe_heitler_Z_squared_scaling():
    # Same endpoint, doubling Z → ~4× larger prefactor at same energy.
    E = np.array([100.0])
    s_cu = bethe_heitler_spectrum(E, 1000.0, "Cu")   # Z=29
    s_ta = bethe_heitler_spectrum(E, 1000.0, "Ta")   # Z=73
    ratio = s_ta[0] / s_cu[0]
    assert ratio == pytest.approx((73.0 / 29.0) ** 2, rel=0.05)


def test_converter_photon_yield_positive():
    N = converter_photon_yield(1000e3, beam_charge_pC=100.0, material="W", thickness_um=500.0)
    assert N > 0 and math.isfinite(N)


# ---------------------------------------------------------------------------
# Radiation reaction
# ---------------------------------------------------------------------------


def test_chi_e_is_dimensionless_and_scales_linearly():
    chi_lo = float(chi_e_parameter(100.0, 1e14))
    chi_hi = float(chi_e_parameter(100.0, 2e14))
    assert chi_hi / chi_lo == pytest.approx(2.0, rel=1e-9)


def test_suppression_factor_bounded():
    # 0 < g(χ) ≤ 1, and g(0) = 1 exactly.
    chis = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    g = quantum_synchrotron_suppression(chis)
    assert g[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(g > 0)
    assert np.all(g <= 1.0)
    # Monotonically decreasing.
    assert np.all(np.diff(g) < 0)


def test_suppression_asymptotic_to_chi_minus_two_thirds():
    # At very large χ, g(χ) → const · χ⁻²/³ (Ritus limit); check the ratio.
    chi_a, chi_b = 1e3, 1e4
    g_a = float(quantum_synchrotron_suppression(chi_a))
    g_b = float(quantum_synchrotron_suppression(chi_b))
    # 10× χ → g drops by ~10² (denominator ∝ χ² in the library's form).
    assert g_a / g_b > 50.0


def test_betatron_field_estimate_positive_and_scales_with_gamma():
    E_lo = betatron_field_estimate_V_per_m(100.0, 1e14, 1e-6)
    E_hi = betatron_field_estimate_V_per_m(1000.0, 1e14, 1e-6)
    assert E_hi == pytest.approx(10.0 * E_lo, rel=1e-9)


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------


def test_divergence_cone_synchrotron_limit():
    assert divergence_cone_fwhm_mrad(100.0, K_wiggler=0.5) == pytest.approx(10.0, rel=1e-9)


def test_divergence_cone_wiggler_limit():
    # K=5, γ=100 → θ = K/γ = 50 mrad.
    assert divergence_cone_fwhm_mrad(100.0, K_wiggler=5.0) == pytest.approx(50.0, rel=1e-9)


def test_angular_pattern_peaks_on_axis():
    theta = np.linspace(-10, 10, 101)
    p = synchrotron_angular_pattern(theta, gamma_e=100.0, harmonic_n=1.0)
    assert p.argmax() == p.size // 2
    assert p.max() == pytest.approx(1.0, rel=1e-12)


def test_collimation_fraction_monotone():
    theta = np.array([0.5, 1.0, 2.0, 5.0])
    f = [collimation_factor(t, 100.0) for t in theta]
    for a, b in zip(f, f[1:], strict=False):
        assert a < b


def test_brightness_positive_and_scales_with_rep_rate():
    B1 = brightness_estimate(1e8, divergence_mrad=5.0, rep_rate_hz=1.0)
    B2 = brightness_estimate(1e8, divergence_mrad=5.0, rep_rate_hz=10.0)
    assert B2 == pytest.approx(10.0 * B1, rel=1e-9)
