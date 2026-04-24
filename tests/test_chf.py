"""Tests for the Coherent Harmonic Focus gain + propagation."""

import numpy as np
import pytest

from harmonyemissions.chf.gain import (
    ChfGainBreakdown,
    extrapolate_3d_gain,
    predict_chf_intensity,
    scaling_I_chf_over_I,
)
from harmonyemissions.chf.propagation import (
    apply_denting_phase,
    harmonic_near_field,
)


def test_gamma_3d_is_gamma_2d_squared():
    g = extrapolate_3d_gain(intensity_attosecond=10.0, intensity_driver=1.0,
                            intensity_at_chf_focus_2d=30.0)
    assert g.gamma_d == pytest.approx(10.0)
    assert g.gamma_2d == pytest.approx(3.0)
    assert g.gamma_3d == pytest.approx(9.0)
    assert g.gamma_total == pytest.approx(90.0)


def test_gain_total_is_d_times_3d():
    g = extrapolate_3d_gain(5.0, 1.0, 20.0)
    assert g.gamma_total == pytest.approx(g.gamma_d * g.gamma_3d)


def test_gain_breakdown_dict_shape():
    g = ChfGainBreakdown(2.0, 3.0, 9.0, 18.0)
    d = g.to_dict()
    assert set(d.keys()) == {"Gamma_D", "Gamma_2D", "Gamma_3D", "Gamma_total"}


def test_ichf_scaling_is_cubic_in_a0():
    """I_CHF/I ∝ a₀³: doubling a₀ should increase the ratio 8×."""
    r1 = scaling_I_chf_over_I(12.0)
    r2 = scaling_I_chf_over_I(24.0)
    r4 = scaling_I_chf_over_I(48.0)
    assert abs(r2 / r1 - 8.0) < 1e-9
    assert abs(r4 / r2 - 8.0) < 1e-9


def test_reference_anchor():
    """Gemini anchor: a₀=24 → ratio ≈ 80 by default."""
    assert scaling_I_chf_over_I(24.0) == pytest.approx(80.0)


def test_predict_chf_intensity():
    """Intensity × a₀³ ratio."""
    i = predict_chf_intensity(driver_intensity_w_per_cm2=1.2e21, a0=24.0)
    # 1.2e21 × 80 = 9.6e22. Paper's Fig. 4 anchor (Gemini).
    assert 8e22 < i < 1.1e23


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        extrapolate_3d_gain(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        scaling_I_chf_over_I(-1.0)


def test_denting_phase_is_unitary():
    """Applying the phase preserves |U|²."""
    rng = np.random.default_rng(0)
    u0 = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex128)
    dz = rng.uniform(0.0, 0.1, size=(32, 32))
    u_phased = apply_denting_phase(u0, dz, harmonic_n=30, angle_deg=45.0)
    np.testing.assert_allclose(np.abs(u_phased), np.abs(u0), rtol=1e-12)


def test_harmonic_near_field_suppresses_above_cutoff():
    """For a uniform a₀ map, the near-field amplitude must fall off as n crosses n_c."""
    u0 = np.ones((8, 8), dtype=np.complex128)
    a0_map = np.full((8, 8), 10.0)
    dz = np.zeros((8, 8))
    low = float(np.abs(harmonic_near_field(u0, a0_map, dz, 5, 45.0))[0, 0])
    high = float(np.abs(harmonic_near_field(u0, a0_map, dz, 10000, 45.0))[0, 0])
    assert low > high


def test_pipeline_end_to_end_produces_chf_gain():
    """surface_pipeline run should populate chf_gain with the four keys."""
    from harmonyemissions import Laser, Target, simulate
    laser = Laser(a0=10.0, spatial_profile="super_gaussian", spot_fwhm_um=2.0)
    target = Target.sio2(t_HDR_fs=351.0)
    r = simulate(laser, target, model="surface_pipeline")
    assert set(r.chf_gain.keys()) == {"Gamma_D", "Gamma_2D", "Gamma_3D", "Gamma_total"}
    # All finite and positive.
    for k, v in r.chf_gain.items():
        assert np.isfinite(v) and v > 0, f"{k}={v}"
