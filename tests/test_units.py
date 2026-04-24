import math

import numpy as np
import pytest

from harmonyemissions.units import (
    FINE_STRUCTURE_ALPHA,
    HC_KEV_NM,
    M_E_C2_KEV,
    LaserUnits,
    a0_to_intensity,
    critical_density,
    default_xray_energy_grid,
    gamma_from_a0,
    hot_electron_temperature_keV,
    intensity_to_a0,
    keV_per_harmonic,
    omega_from_wavelength,
    photon_energy_keV_from_harmonic,
    plasma_frequency,
    ponderomotive_energy_ev,
)


def test_a0_intensity_roundtrip():
    # a₀ = 1 at 800 nm ≈ 2.14e18 W/cm²
    i = a0_to_intensity(1.0, 0.8)
    assert 2.0e18 < i < 2.2e18
    assert intensity_to_a0(i, 0.8) == pytest.approx(1.0)


def test_gamma_from_a0():
    assert gamma_from_a0(0.0) == pytest.approx(1.0)
    assert gamma_from_a0(1.0) == pytest.approx(math.sqrt(1.5))
    assert gamma_from_a0(10.0) == pytest.approx(math.sqrt(51.0))
    assert gamma_from_a0(1.0, "circular") == pytest.approx(math.sqrt(2.0))
    with pytest.raises(ValueError):
        gamma_from_a0(1.0, "elliptical")


def test_ponderomotive_energy():
    # Corkum cutoff at I = 1e14 W/cm², 800 nm: U_p ≈ 5.97 eV
    up = ponderomotive_energy_ev(1e14, 0.8)
    assert 5.5 < up < 6.5


def test_omega_critical_density_consistency():
    lam = 8e-7
    omega = omega_from_wavelength(lam)
    nc = critical_density(omega)
    # Plasma frequency at critical density must equal the laser omega.
    omega_p_at_nc = plasma_frequency(nc)
    assert omega_p_at_nc == pytest.approx(omega, rel=1e-9)


def test_laser_units_dataclass():
    u = LaserUnits.from_wavelength_um(0.8)
    # T₀ at 800 nm ≈ 2.67 fs
    assert 2.6e-15 < u.period_s < 2.7e-15
    # k₀ · λ = 2π
    assert u.k0 * u.wavelength_m == pytest.approx(2 * math.pi)


def test_keV_per_harmonic_matches_hc_over_lambda():
    # At 800 nm, fundamental photon is 1.55 eV, i.e. 1.55e-3 keV.
    assert keV_per_harmonic(0.8) == pytest.approx(HC_KEV_NM * 1e-3 / 0.8, rel=1e-9)
    # At Yb:YAG 1.03 μm.
    assert keV_per_harmonic(1.03) == pytest.approx(1.2039e-3, rel=1e-3)


def test_photon_energy_from_harmonic_vectorised():
    n = np.array([1, 10, 100, 1000])
    E = photon_energy_keV_from_harmonic(n, 0.8)
    assert E.shape == n.shape
    assert E[0] == pytest.approx(1.5498e-3, rel=1e-3)  # 1.55 eV
    # n = 10,000 at 800 nm lands near 15.5 keV.
    assert photon_energy_keV_from_harmonic(10_000, 0.8) == pytest.approx(15.498, rel=1e-3)


def test_hot_electron_temperature_wilks():
    # a₀=1 is non-relativistic limit; γ_pond=√1.5 ≈ 1.2247, so T ≈ 0.2247·m_ec²≈115 keV.
    T = hot_electron_temperature_keV(1.0, 0.8, scaling="wilks")
    assert T == pytest.approx(M_E_C2_KEV * (math.sqrt(1.5) - 1.0), rel=1e-9)
    # a₀=10 at λ=0.8: γ_pond=√51 ≈ 7.14, T ≈ 6.14·m_ec² ≈ 3138 keV.
    T10 = hot_electron_temperature_keV(10.0, 0.8)
    assert 3000 < T10 < 3200


def test_hot_electron_temperature_beg_softer_than_wilks():
    # Beg's T ∝ I^{1/3} rises more slowly than Wilks' T ∝ √I.
    # Crossover behaviour: Beg < Wilks in the ultra-relativistic regime.
    T_beg_3 = hot_electron_temperature_keV(3.0, 0.8, scaling="beg")
    T_wilks_3 = hot_electron_temperature_keV(3.0, 0.8, scaling="wilks")
    assert T_beg_3 < T_wilks_3
    # At a₀=3 both scalings land in the hundreds of keV.
    assert 300.0 < T_beg_3 < 700.0
    # Doubling a₀ multiplies Wilks' T by ~2 but Beg's T by only ~2^{2/3} ≈ 1.59.
    T_beg_6 = hot_electron_temperature_keV(6.0, 0.8, scaling="beg")
    T_wilks_6 = hot_electron_temperature_keV(6.0, 0.8, scaling="wilks")
    assert T_beg_6 / T_beg_3 == pytest.approx(2.0 ** (2.0 / 3.0), rel=0.05)
    # Wilks ratio: √(1+0.5·36) ≈ 4.36, √(1+0.5·9) ≈ 2.34; (4.36-1)/(2.34-1) ≈ 2.50.
    assert T_wilks_6 / T_wilks_3 == pytest.approx(
        (math.sqrt(1.0 + 0.5 * 36.0) - 1.0) / (math.sqrt(1.0 + 0.5 * 9.0) - 1.0),
        rel=1e-9,
    )


def test_hot_electron_temperature_bad_scaling():
    with pytest.raises(ValueError):
        hot_electron_temperature_keV(1.0, 0.8, scaling="bogus")


def test_default_xray_energy_grid_shape_and_range():
    g = default_xray_energy_grid(E_min_keV=0.5, E_max_keV=5000.0, n_points=256)
    assert g.shape == (256,)
    assert g[0] == pytest.approx(0.5)
    assert g[-1] == pytest.approx(5000.0)
    # Log-spaced: successive ratios are constant.
    r = g[1:] / g[:-1]
    assert np.allclose(r, r[0], rtol=1e-10)


def test_fine_structure_alpha_consistent():
    # Used to appear ad-hoc in bgp.py / emission/spikes.py; canonical value ≈ 1/137.036.
    assert 1.0 / FINE_STRUCTURE_ALPHA == pytest.approx(137.036, rel=1e-4)
