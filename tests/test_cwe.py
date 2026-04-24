"""Tests for the Coherent Wake Emission model."""

import math

import numpy as np

from harmonyemissions import Laser, Target, simulate


def test_cwe_cutoff_at_plasma_harmonic():
    """Cutoff harmonic should equal √(n_e/n_c)."""
    r = simulate(Laser(a0=0.3), Target.overdense(n_over_nc=100.0), model="cwe")
    assert r.diagnostics["n_plasma_cutoff"] == math.sqrt(100.0)


def test_cwe_cutoff_independent_of_a0():
    """Paper's ref. 44: n_p depends only on plasma density, not laser a₀."""
    r1 = simulate(Laser(a0=0.1), Target.overdense(n_over_nc=50.0), model="cwe")
    r2 = simulate(Laser(a0=5.0), Target.overdense(n_over_nc=50.0), model="cwe")
    assert r1.diagnostics["n_plasma_cutoff"] == r2.diagnostics["n_plasma_cutoff"]


def test_cwe_cutoff_scales_with_sqrt_density():
    """Doubling density should multiply cutoff by √2."""
    r_low = simulate(Laser(a0=0.5), Target.overdense(n_over_nc=50.0), model="cwe")
    r_high = simulate(Laser(a0=0.5), Target.overdense(n_over_nc=200.0), model="cwe")
    ratio = r_high.diagnostics["n_plasma_cutoff"] / r_low.diagnostics["n_plasma_cutoff"]
    assert abs(ratio - 2.0) < 1e-6


def test_cwe_spectrum_finite_and_monotone_in_plateau():
    r = simulate(Laser(a0=0.5), Target.overdense(n_over_nc=400.0), model="cwe")
    n_p = r.diagnostics["n_plasma_cutoff"]
    s = r.spectrum.values
    n = r.spectrum.coords["harmonic"].values
    # Everything finite and positive.
    assert np.all(np.isfinite(s))
    assert np.all(s > 0)
    # In the plateau the spectrum should fall monotonically.
    in_plateau = (n >= 2) & (n <= 0.9 * n_p)
    s_plateau = s[in_plateau]
    if s_plateau.size > 2:
        diffs = np.diff(s_plateau)
        # Mostly decreasing.
        assert (diffs <= 0).sum() >= len(diffs) - 1


def test_cwe_plateau_slope_is_minus_four():
    """Log-log slope in the plateau should be ≈ −4."""
    r = simulate(Laser(a0=0.5), Target.overdense(n_over_nc=1000.0), model="cwe")
    n_p = r.diagnostics["n_plasma_cutoff"]
    slope, _ = r.fit_power_law(n_min=3.0, n_max=0.5 * n_p)
    assert abs(slope - (-4.0)) < 0.25
