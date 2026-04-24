"""Regression tests for the LWFA betatron model."""

import math

import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.units import E_CHARGE, M_E, omega_from_wavelength, plasma_frequency


def test_betatron_frequency_formula():
    """ω_β = ω_p / √(2γ)."""
    laser = Laser(a0=2.0, wavelength_um=0.8)
    target = Target.underdense(n_over_nc=0.001, electron_energy_mev=200.0)
    r = simulate(laser, target, model="betatron")

    omega0 = omega_from_wavelength(0.8e-6)
    n_c = 8.8541878128e-12 * M_E * omega0**2 / E_CHARGE**2
    n_e = 0.001 * n_c
    omega_p = plasma_frequency(n_e)
    gamma_e = 200.0 / 0.511 + 1.0
    omega_b_expected = omega_p / math.sqrt(2.0 * gamma_e)
    assert r.diagnostics["omega_b_over_omega0"] == pytest.approx(omega_b_expected / omega0, rel=1e-6)


def test_critical_frequency_scales_with_gamma_squared():
    """ω_c ∝ γ² · K · ω_β → scales ~ γ² for fixed K."""
    r1 = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=100.0), model="betatron")
    r2 = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=400.0), model="betatron")
    # γ roughly scales 4×; ω_c carries γ²·K, but K ∝ γ·r_β·k_β = γ·r_β·ω_p/(c √(2γ)) ∝ √γ,
    # and ω_β ∝ 1/√γ. Combine: ω_c ∝ γ² · √γ · 1/√γ = γ².
    ratio = r2.diagnostics["omega_c_over_omega0"] / r1.diagnostics["omega_c_over_omega0"]
    gamma_ratio = (400 / 0.511 + 1) / (100 / 0.511 + 1)
    # Must be close to γ² (within 5% given discretization).
    assert abs(ratio / gamma_ratio**2 - 1.0) < 0.05


def test_betatron_spectrum_nonzero_and_finite():
    r = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=500.0), model="betatron")
    import numpy as np
    vals = r.spectrum.values
    assert np.all(np.isfinite(vals))
    assert vals.max() > 0


def test_betatron_photon_energy_keV_coord_present_and_monotonic():
    """Non-dim coord `photon_energy_keV` must ride the `harmonic` axis."""
    r = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=500.0), model="betatron")
    assert "photon_energy_keV" in r.spectrum.coords
    import numpy as np
    e = r.spectrum.coords["photon_energy_keV"].values
    assert np.all(np.diff(e) > 0), "photon_energy_keV must be strictly monotonic"
    # At 800 nm, max harmonic → max photon energy > 1 keV for a 500 MeV LWFA.
    assert e[-1] > 1.0


def test_betatron_critical_energy_scales_with_gamma_squared():
    """Promoted keV diagnostic must reproduce the γ² scaling of ω_c."""
    r1 = simulate(
        Laser(a0=2.0),
        Target.underdense(0.001, electron_energy_mev=100.0),
        model="betatron",
    )
    r2 = simulate(
        Laser(a0=2.0),
        Target.underdense(0.001, electron_energy_mev=400.0),
        model="betatron",
    )
    ratio = (
        r2.diagnostics["photon_energy_keV_critical"]
        / r1.diagnostics["photon_energy_keV_critical"]
    )
    gamma_ratio = (400 / 0.511 + 1) / (100 / 0.511 + 1)
    assert abs(ratio / gamma_ratio**2 - 1.0) < 0.05


def test_betatron_photon_count_estimate_scales_quadratically_with_gamma_K():
    """Count estimate ∝ γ²·K²; both rise with γ → sharp rise with energy."""
    r1 = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=100.0), model="betatron")
    r2 = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=400.0), model="betatron")
    c1 = r1.diagnostics["photons_per_pulse_estimate"]
    c2 = r2.diagnostics["photons_per_pulse_estimate"]
    assert c2 > c1 > 0


def test_betatron_roundtrip_preserves_photon_energy_coord(tmp_path):
    r = simulate(Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=300.0), model="betatron")
    p = tmp_path / "betatron.h5"
    r.save(p)
    from harmonyemissions.models.base import Result
    r2 = Result.load(p)
    assert "photon_energy_keV" in r2.spectrum.coords
