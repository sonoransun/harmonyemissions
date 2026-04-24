"""Tests for the surface denting model (Vincenti 2014 / Timmis 2026 eq. 10–12)."""

import math

import numpy as np

from harmonyemissions.surface.denting import (
    DentingInputs,
    dent_depth_electron,
    dent_depth_ion,
    dent_map,
    denting_phase,
    pi0,
)


def test_pi0_for_sio2_is_reasonable():
    """Π₀ for SiO₂ at 45° should be dimensionless and ~0.01–0.05."""
    inp = DentingInputs(scale_length_lambda=0.14, angle_deg=45.0)
    p = pi0(inp)
    assert 1e-3 < p < 0.1


def test_zero_fluence_gives_zero_ion_dent():
    inp = DentingInputs(scale_length_lambda=0.14)
    f = np.zeros((8, 8))
    assert np.all(dent_depth_ion(f, inp) == 0.0)


def test_ion_dent_grows_logarithmically():
    """Δz_i = 2L·ln(1 + C·f) — monotone in f, sub-linear in the big-f regime."""
    inp = DentingInputs(scale_length_lambda=0.14)
    # Pick fluences deep in the logarithmic regime so successive decades
    # add comparable increments.
    f = np.array([[0.0, 1e4, 1e5, 1e6, 1e7]])
    d = dent_depth_ion(f, inp)
    d0, d1, d2, d3, d4 = d[0]
    assert d0 == 0.0
    # Monotone growth.
    assert d1 < d2 < d3 < d4
    # In the log-asymptotic regime each tenfold adds the same 2L·ln(10) ≈ 0.64·L.
    for a, b in [(d2 - d1, d3 - d2), (d3 - d2, d4 - d3)]:
        assert abs(a - b) / max(a, b) < 0.15


def test_zero_scale_length_gives_zero_ion_dent():
    inp = DentingInputs(scale_length_lambda=0.0)
    f = np.array([[10.0]])
    assert np.all(dent_depth_ion(f, inp) == 0.0)


def test_electron_dent_scales_with_a0():
    inp = DentingInputs(scale_length_lambda=0.14)
    assert dent_depth_electron(np.array([0.0]), inp)[0] == 0.0
    # For a₀=1: γ = √1.5 ≈ 1.22, so Δz_e ≈ 1/(2π · 1.22) ≈ 0.13 λ.
    d = dent_depth_electron(np.array([1.0]), inp)[0]
    assert 0.10 < d < 0.15
    # a₀=10: γ ≈ 7.14, Δz_e ≈ 10/(2π · 7.14) ≈ 0.22 λ.
    d = dent_depth_electron(np.array([10.0]), inp)[0]
    assert 0.20 < d < 0.26


def test_dent_map_combines_ion_and_electron():
    inp = DentingInputs(scale_length_lambda=0.14)
    a0_map = np.full((4, 4), 10.0)
    dz = dent_map(a0_map, duration_T0=20.0, inputs=inp)
    assert dz.shape == (4, 4)
    assert np.all(dz > 0)


def test_denting_phase_proportional_to_harmonic():
    """φ_n ∝ n: doubling harmonic doubles the phase."""
    dz = np.full((4, 4), 0.02)
    phi30 = denting_phase(dz, 30, angle_deg=45.0)
    phi60 = denting_phase(dz, 60, angle_deg=45.0)
    # Ratio should be exactly 2 (up to floating-point).
    np.testing.assert_allclose(phi60 / phi30, 2.0, rtol=1e-12)


def test_denting_phase_angle_dependence():
    """φ_n ∝ cos θ: normal incidence gives the largest phase."""
    dz = np.full((2, 2), 0.02)
    phi_0 = denting_phase(dz, 30, angle_deg=0.0)
    phi_45 = denting_phase(dz, 30, angle_deg=45.0)
    phi_60 = denting_phase(dz, 30, angle_deg=60.0)
    assert phi_0[0, 0] > phi_45[0, 0] > phi_60[0, 0]
    # φ(45)/φ(0) should be 1/√2.
    np.testing.assert_allclose(
        phi_45[0, 0] / phi_0[0, 0], 1.0 / math.sqrt(2.0), rtol=1e-6
    )
