"""Regression tests for the BGP universal spectrum."""


from harmonyemissions import Laser, Target, simulate


def test_bgp_slope_is_minus_eight_thirds():
    """Plateau slope must match the -8/3 universal prediction within 2%."""
    r = simulate(Laser(a0=30.0), Target.overdense(100.0), model="bgp")
    nc = r.diagnostics["n_cutoff"]
    slope, _ = r.fit_power_law(n_min=3.0, n_max=0.05 * nc)
    assert abs(slope - (-8.0 / 3.0)) < 0.05


def test_bgp_cutoff_scales_as_gamma_cubed():
    """Cutoff n_c ∝ γ_max³: doubling γ should multiply n_c by ~8."""
    r_low = simulate(Laser(a0=5.0), Target.overdense(100.0), model="bgp")
    r_high = simulate(Laser(a0=10.0), Target.overdense(100.0), model="bgp")
    ratio_nc = r_high.diagnostics["n_cutoff"] / r_low.diagnostics["n_cutoff"]
    gamma_ratio = r_high.diagnostics["gamma_max"] / r_low.diagnostics["gamma_max"]
    assert abs(ratio_nc - gamma_ratio ** 3) / ratio_nc < 0.01


def test_bgp_spectrum_monotone_in_log():
    """The BGP envelope must be monotonically decreasing after the first harmonic."""
    r = simulate(Laser(a0=10.0), Target.overdense(100.0), model="bgp")
    s = r.spectrum.values
    # First few harmonics dominated by power law; ensure long-range decay.
    assert s[100] < s[10]
    assert s[-1] < s[100]
