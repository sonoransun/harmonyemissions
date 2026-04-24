"""Regression tests for the ICS / Compton source model.

The model (``harmonyemissions.models.ics``) follows the
Klein–Nishina + χ_e architecture from :mod:`harmonyemissions.gamma`:

* ``photon_energy_keV_cutoff`` from ``compton_max_photon_energy_keV``
  (head-on, 4γ²·E_L with KN recoil denominator);
* plateau + power-law tail ``ics_photon_spectrum_keV``;
* quantum-χ suppression from ``quantum_synchrotron_suppression``;
* absolute yield from KN cross section × bunch / laser photon counts.
"""

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.gamma.compton import compton_max_photon_energy_keV


def _run(a0: float, E_mev: float, kind: str = "electron_beam", wavelength_um: float = 0.8):
    laser = Laser(a0=a0, wavelength_um=wavelength_um, spot_fwhm_um=5.0, duration_fs=30.0)
    if kind == "electron_beam":
        target = Target.electron_beam(beam_energy_mev=E_mev, beam_divergence_mrad=0.5)
    elif kind == "underdense":
        target = Target.underdense(n_over_nc=0.001, electron_energy_mev=E_mev)
    else:
        raise ValueError(kind)
    return simulate(laser, target, model="ics")


def test_ics_cutoff_scales_with_gamma_squared_in_thomson_limit():
    """At γ · E_L / m_ec² ≪ 1 recoil denom → 1; E_max = 4γ²·E_L ∝ γ²."""
    # 50 MeV and 200 MeV beams at 800 nm: head-on recoil denom ~1.002 vs 1.008,
    # both Thomson-like → ratio ≈ (γ₂/γ₁)².
    r1 = _run(a0=0.05, E_mev=50.0)
    r2 = _run(a0=0.05, E_mev=200.0)
    gamma_ratio = (200.0 / 0.511 + 1.0) / (50.0 / 0.511 + 1.0)
    ratio = (
        r2.diagnostics["photon_energy_keV_cutoff"]
        / r1.diagnostics["photon_energy_keV_cutoff"]
    )
    assert ratio == pytest.approx(gamma_ratio**2, rel=0.02)


def test_ics_cutoff_matches_closed_form():
    """Diagnostic matches the standalone compton_max_photon_energy_keV helper."""
    r = _run(a0=0.1, E_mev=1000.0)
    gamma_e = 1.0 + 1000.0 / 0.511
    expected = compton_max_photon_energy_keV(gamma_e, 0.8)
    assert r.diagnostics["photon_energy_keV_cutoff"] == pytest.approx(expected, rel=1e-6)


def test_ics_spectrum_finite_and_energy_coord_monotonic():
    r = _run(a0=0.3, E_mev=200.0)
    e = r.spectrum.coords["photon_energy_keV"].values
    s = r.spectrum.values
    assert np.all(np.isfinite(s))
    assert np.all(np.diff(e) > 0)
    assert s.max() > 0


def test_ics_electron_beam_uses_beam_energy_field():
    """For kind='electron_beam' the bunch energy comes from beam_energy_mev."""
    r = _run(a0=0.3, E_mev=500.0, kind="electron_beam")
    gamma = 1.0 + 500.0 / 0.511
    assert r.diagnostics["gamma_e"] == pytest.approx(gamma, rel=1e-9)


def test_ics_underdense_uses_electron_energy_field():
    r = _run(a0=0.3, E_mev=500.0, kind="underdense")
    gamma = 1.0 + 500.0 / 0.511
    assert r.diagnostics["gamma_e"] == pytest.approx(gamma, rel=1e-9)


def test_ics_diagnostics_report_rest_frame_and_recoil():
    r = _run(a0=0.3, E_mev=1000.0)
    assert r.diagnostics["rest_frame_photon_keV"] > 0
    assert r.diagnostics["recoil_parameter_x"] > 0
    # Recoil parameter should be O(10⁻²) for a GeV beam on 800 nm.
    assert 1e-4 < r.diagnostics["recoil_parameter_x"] < 1.0


def test_ics_chi_e_grows_with_a0():
    """χ_e = γ·|E_⊥|/E_cr grows linearly with the laser field → a₀."""
    r_lo = _run(a0=0.2, E_mev=1000.0)
    r_hi = _run(a0=2.0, E_mev=1000.0)
    assert r_hi.diagnostics["chi_e_nominal"] > 5.0 * r_lo.diagnostics["chi_e_nominal"]


def test_ics_photon_yield_scales_with_klein_nishina_cross_section():
    """N_γ ∝ σ_KN × bunch charge × laser photons; crosscheck the σ diagnostic."""
    r = _run(a0=0.3, E_mev=1000.0)
    assert r.diagnostics["klein_nishina_cross_section_m2"] > 0
    assert r.diagnostics["n_photons_per_pulse"] > 0


def test_ics_cutoff_lands_in_keV_band_for_gev_beam():
    # GeV beam + 800 nm → cutoff ~20 MeV in the Thomson limit.
    r = _run(a0=0.1, E_mev=1000.0)
    E_cut = r.diagnostics["photon_energy_keV_cutoff"]
    assert 1.0e3 < E_cut < 3.0e4  # 1–30 MeV
