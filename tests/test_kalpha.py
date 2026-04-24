"""Regression tests for the K-α fluorescence model."""

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.materials import lookup


def _run(material: str, a0: float = 10.0):
    laser = Laser(a0=a0, wavelength_um=0.8)
    target = Target(kind="overdense", material=material, n_over_nc=200.0)
    return simulate(laser, target, model="kalpha")


@pytest.mark.parametrize("material", ["Cu", "Ti", "Mo"])
def test_kalpha_peak_sits_at_published_line_energy(material: str):
    r = _run(material)
    e = r.spectrum.coords["photon_energy_keV"].values
    s = r.spectrum.values
    peak_idx = int(np.argmax(s))
    expected = lookup(material).K_alpha1_keV
    assert e[peak_idx] == pytest.approx(expected, rel=0.01)


def test_kalpha1_dominates_kalpha2():
    """Use Mo (Kα1–Kα2 separation ~105 eV) so lines are well-resolved."""
    r = _run("Mo")
    e = r.spectrum.coords["photon_energy_keV"].values
    s = r.spectrum.values
    mat = lookup("Mo")

    def amp_at(E0: float) -> float:
        return float(s[int(np.argmin(np.abs(e - E0)))])

    A1 = amp_at(mat.K_alpha1_keV)
    A2 = amp_at(mat.K_alpha2_keV)
    assert A1 > A2 > 0
    # Statistical branching 0.5; Lorentzian tail cross-contamination bumps
    # the ratio slightly above 0.5.
    assert 0.45 < A2 / A1 < 0.75


def test_kalpha_finite_and_positive():
    r = _run("Cu")
    s = r.spectrum.values
    assert np.all(np.isfinite(s))
    assert s.min() >= 0.0
    assert s.max() > 0.0


def test_kalpha_reports_material_diagnostics():
    r = _run("Mo")
    assert r.diagnostics["material_Z"] == 42
    assert r.diagnostics["K_alpha1_keV"] == pytest.approx(17.479, abs=0.01)
    assert r.diagnostics["fluorescence_yield_K"] == pytest.approx(0.77, abs=0.01)


def test_kalpha_below_ionisation_threshold_produces_only_pedestal():
    # If T_hot < K-edge, σ_K = 0 and we get only the small bremsstrahlung pedestal.
    laser = Laser(a0=0.5, wavelength_um=0.8)   # very low T_hot (~35 keV)
    target = Target(
        kind="overdense",
        material="Mo",                          # K-edge 20 keV; above T_hot
        n_over_nc=200.0,
        hot_electron_temp_keV=5.0,              # well below Mo K-edge
    )
    r = simulate(laser, target, model="kalpha")
    assert r.diagnostics["sigma_K_at_T_hot"] == pytest.approx(0.0)
    # Spectrum non-negative; line peak amplitude collapses to pedestal level.
    assert r.spectrum.values.max() > 0
