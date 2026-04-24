"""Regression tests for the ROM model."""

import numpy as np

from harmonyemissions import Laser, Target, simulate


def test_rom_produces_time_field_and_spectrum():
    r = simulate(Laser(a0=10.0), Target.overdense(100.0, 0.05), model="rom")
    assert r.time_field is not None
    assert r.time_field.size > 1000
    assert r.spectrum.size > 100


def test_rom_gamma_peak_near_a0_for_sharp_gradient():
    """For L/λ ≪ 1 and dense plasma, γ_mirror ≈ a₀."""
    r = simulate(Laser(a0=20.0), Target.overdense(200.0, 0.01), model="rom")
    gm = r.diagnostics["gamma_mirror_peak"]
    assert 15.0 < gm <= 20.0


def test_rom_cutoff_scales_as_gamma_cubed():
    """n_cutoff should scale like γ³ in the BGP-anchored ROM."""
    r_low = simulate(Laser(a0=5.0), Target.overdense(200.0, 0.01), model="rom")
    r_high = simulate(Laser(a0=10.0), Target.overdense(200.0, 0.01), model="rom")
    ratio_nc = r_high.diagnostics["n_cutoff"] / r_low.diagnostics["n_cutoff"]
    gamma_ratio = r_high.diagnostics["gamma_mirror_peak"] / r_low.diagnostics["gamma_mirror_peak"]
    assert abs(ratio_nc - gamma_ratio ** 3) / ratio_nc < 0.01


def test_rom_plateau_slope_matches_bgp():
    """ROM is BGP-anchored, so its plateau slope is also -8/3."""
    r = simulate(Laser(a0=30.0), Target.overdense(200.0, 0.01), model="rom")
    nc = r.diagnostics["n_cutoff"]
    slope, _ = r.fit_power_law(n_min=3.0, n_max=0.05 * nc)
    assert abs(slope - (-8.0 / 3.0)) < 0.05


def test_rom_attosecond_pulse_when_window_requested():
    """Passing a harmonic_window should produce a bandpass-synthesized pulse."""
    from harmonyemissions.config import NumericsConfig
    r = simulate(
        Laser(a0=10.0),
        Target.overdense(100.0, 0.05),
        model="rom",
        numerics=NumericsConfig(harmonic_window=(20.0, 60.0)),
    )
    assert r.attosecond_pulse is not None
    assert np.any(np.abs(r.attosecond_pulse.values) > 0)
