"""Regression tests for the hot-electron bremsstrahlung model."""

import math

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.units import hot_electron_temperature_keV


def _run(a0: float, material: str = "Cu", T_override: float | None = None):
    laser = Laser(a0=a0, wavelength_um=0.8)
    target = Target(
        kind="overdense",
        n_over_nc=100.0,
        material=material,
        hot_electron_temp_keV=T_override,
    )
    return simulate(laser, target, model="bremsstrahlung")


def test_bremsstrahlung_spectrum_finite_and_monotonic_energy_coord():
    r = _run(a0=5.0)
    s = r.spectrum.values
    e = r.spectrum.coords["photon_energy_keV"].values
    assert np.all(np.isfinite(s))
    assert np.all(np.diff(e) > 0)
    assert s.max() > 0


def test_bremsstrahlung_T_hot_override_respected():
    r = _run(a0=5.0, T_override=250.0)
    assert r.diagnostics["hot_electron_temp_keV"] == pytest.approx(250.0)
    assert "override" in r.provenance["T_hot_source"]


def test_bremsstrahlung_T_hot_default_from_wilks():
    r = _run(a0=5.0)
    expected = hot_electron_temperature_keV(5.0, 0.8)
    assert r.diagnostics["hot_electron_temp_keV"] == pytest.approx(expected, rel=1e-9)
    assert "Wilks" in r.provenance["T_hot_source"]


def test_bremsstrahlung_efolding_slope_matches_minus_one_over_T_hot():
    """exp1(x) ≈ e^{-x}/x; fit ln(s·E) to subtract the 1/x prefactor."""
    T_hot = 200.0
    r = _run(a0=5.0, T_override=T_hot)
    e = r.spectrum.coords["photon_energy_keV"].values
    s = r.spectrum.values
    mask = (e > T_hot) & (e < 5.0 * T_hot) & (s > 0)
    slope, _ = np.polyfit(e[mask], np.log(s[mask] * e[mask]), 1)
    # exp1(x) = -γ - ln(x) + x - x²/4 + ... · e^-x: high-order residual
    # leaves ~10% curvature in the linear fit over [T_hot, 5·T_hot].
    assert slope == pytest.approx(-1.0 / T_hot, rel=0.12)


def test_bremsstrahlung_doubling_a0_raises_efolding_per_wilks():
    """T_hot scales via √(1+a²/2)−1; e-folding moves with it."""
    r_lo = _run(a0=5.0)
    r_hi = _run(a0=10.0)
    ratio = r_hi.diagnostics["photon_energy_keV_efolding"] / r_lo.diagnostics["photon_energy_keV_efolding"]
    expected = (math.sqrt(1.0 + 0.5 * 100.0) - 1.0) / (math.sqrt(1.0 + 0.5 * 25.0) - 1.0)
    assert ratio == pytest.approx(expected, rel=1e-3)
