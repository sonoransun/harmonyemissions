"""Regression tests for the CSE model."""

from harmonyemissions import Laser, Target, simulate


def test_cse_produces_spectrum():
    r = simulate(Laser(a0=10.0), Target.overdense(200.0, 0.01), model="cse")
    assert r.spectrum.size > 100
    assert r.diagnostics["n_cutoff"] > 10.0


def test_cse_cutoff_scales_with_gamma_cubed_like():
    """CSE also predicts n_c ∝ γ³ though with a different prefactor."""
    r_low = simulate(Laser(a0=5.0), Target.overdense(200.0, 0.01), model="cse")
    r_high = simulate(Laser(a0=10.0), Target.overdense(200.0, 0.01), model="cse")
    ratio = r_high.diagnostics["n_cutoff"] / r_low.diagnostics["n_cutoff"]
    assert 5.0 < ratio < 9.0  # (γ_high/γ_low)^3 ≈ 7 for a₀=5→10
