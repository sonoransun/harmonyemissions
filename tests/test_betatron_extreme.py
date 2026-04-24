"""Regression tests for the extreme-energy betatron additions.

Locks in the new diagnostics (χ_e, quantum suppression, divergence,
brightness, rest-frame transverse field) and the photon_energy_keV
coord that the γ-band detector pipeline relies on.
"""

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate


def _run(E_mev: float, n_over_nc: float = 0.001, r_b_um: float = 1.0):
    return simulate(
        Laser(a0=2.0, wavelength_um=0.8),
        Target.underdense(
            n_over_nc=n_over_nc,
            electron_energy_mev=E_mev,
            betatron_amplitude_um=r_b_um,
        ),
        model="betatron",
    )


def test_photon_energy_keV_coord_present():
    r = _run(500.0)
    assert "photon_energy_keV" in r.spectrum.coords
    assert r.spectrum.coords["photon_energy_keV"].values.max() > 0


def test_chi_e_grows_with_beam_energy():
    """χ_e = γ · |E_⊥| / E_Schwinger grows linearly in γ at fixed field."""
    r_lo = _run(200.0)
    r_hi = _run(2000.0)
    assert r_hi.diagnostics["chi_e"] > r_lo.diagnostics["chi_e"]


def test_quantum_suppression_smaller_at_higher_gamma():
    """Higher γ → larger χ_e → stronger suppression at the cutoff."""
    s_lo = _run(200.0).diagnostics["quantum_suppression_at_cutoff"]
    s_hi = _run(5000.0).diagnostics["quantum_suppression_at_cutoff"]
    assert 0.0 < s_hi < s_lo <= 1.0


def test_divergence_fwhm_mrad_inversely_with_gamma_when_K_small():
    """In the synchrotron limit (K ≪ 1) θ_FWHM ≈ 1/γ → mrad."""
    # Very small r_β keeps K below 1.
    r1 = _run(500.0, r_b_um=0.05)
    r2 = _run(2000.0, r_b_um=0.05)
    div_ratio = (
        r1.diagnostics["divergence_FWHM_mrad"]
        / r2.diagnostics["divergence_FWHM_mrad"]
    )
    gamma_ratio = r2.diagnostics["gamma_e"] / r1.diagnostics["gamma_e"]
    assert div_ratio == pytest.approx(gamma_ratio, rel=0.05)


def test_brightness_positive_and_finite():
    r = _run(500.0)
    b = r.diagnostics["brightness_peak_ph_s_mm2_mrad2_0p1bw"]
    assert b > 0 and np.isfinite(b)


def test_transverse_field_reported():
    r = _run(1000.0)
    assert r.diagnostics["transverse_field_V_per_m"] > 0


def test_spectrum_finite_and_cutoff_in_keV_range_for_GeV_beam():
    r = _run(1000.0)
    s = r.spectrum.values
    assert np.all(np.isfinite(s))
    # Critical energy for a GeV LWFA beam (n/n_c=10⁻³, r_β=1 μm) lands
    # in the soft-to-hard X-ray range.
    E_crit = r.diagnostics["photon_energy_keV_critical"]
    assert 0.001 < E_crit < 1.0e3   # keV — 1 eV to 1 MeV is wide but safe
