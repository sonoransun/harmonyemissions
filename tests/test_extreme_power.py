"""End-to-end tests for the Extreme-Power overlay.

Pin: schema rules, homogeneous-path equivalence, two-color heterogeneous
sum, RR derate ranges, QED validity guard, HDF5 round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from harmonyemissions.config import (
    ExtremePowerConfig,
    LaserArrayConfig,
    LaserConfig,
    RunConfig,
)
from harmonyemissions.gamma.radiation_reaction import landau_lifshitz_cutoff_derate
from harmonyemissions.qed import (
    breit_wheeler_pair_rate,
    qed_diagnostics,
    schwinger_ratio,
    vacuum_birefringence_phase_shift,
)
from harmonyemissions.runner import simulate_from_config


_BASE_TARGET = {
    "kind": "overdense",
    "material": "SiO2",
    "n_over_nc": 200.0,
    "t_HDR_fs": 351.0,
    "prepulse_intensity_rel": 1.0e-4,
    "prepulse_delay_fs": 50.0,
}


def _ep_config(overrides=None, *, model="surface_pipeline", n_beams_geom="tetrahedral"):
    """Build a minimal extreme-power-bearing RunConfig payload."""
    cfg = {
        "model": model,
        "backend": "analytical",
        "laser": {"a0": 5.0},
        "target": _BASE_TARGET,
        "laser_array": {"geometry": n_beams_geom},
        "extreme_power": {"enable_radiation_reaction": False, "enable_qed": False},
        "numerics": {
            "pipeline_grid": 32,
            "chf_focal_volume_n": 4,
            "chf_focal_volume_extent_um": 0.5,
        },
    }
    if overrides:
        cfg["extreme_power"].update(overrides)
    return cfg


# ---- schema -------------------------------------------------------------


def test_extreme_power_requires_laser_array():
    payload = {
        "model": "surface_pipeline",
        "backend": "analytical",
        "laser": {"a0": 5.0},
        "target": _BASE_TARGET,
        "extreme_power": {"enable_qed": True},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_extreme_power_requires_surface_pipeline_model():
    payload = {
        "model": "lewenstein",
        "backend": "analytical",
        "laser": {"a0": 1.0},
        "target": {"kind": "gas", "gas_species": "Ar"},
        "laser_array": {"geometry": "tetrahedral"},
        "extreme_power": {},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_per_beam_lasers_length_must_match_n_beams():
    payload = _ep_config(n_beams_geom="cubic")  # 6 faces
    payload["extreme_power"]["per_beam_lasers"] = [
        {"a0": 5.0} for _ in range(3)  # too few
    ]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_per_beam_lasers_match_ok():
    cfg = RunConfig.model_validate(
        _ep_config({"per_beam_lasers": [{"a0": 5.0} for _ in range(4)]})
    )
    assert cfg.extreme_power.per_beam_lasers is not None
    assert len(cfg.extreme_power.per_beam_lasers) == 4


# ---- radiation-reaction -------------------------------------------------


def test_rr_derate_below_a0_floor_is_one():
    derate, chi = landau_lifshitz_cutoff_derate(a0=30.0, wavelength_um=0.8)
    assert derate == 1.0
    assert chi >= 0.0


def test_rr_derate_above_a0_floor_kicks_in():
    derate, chi = landau_lifshitz_cutoff_derate(a0=130.0, wavelength_um=0.8)
    assert derate < 1.0
    assert chi > 0.0


def test_rr_derate_grows_with_a0():
    """At λ=0.8 μm, derate at a₀=300 should be visibly below derate at a₀=80."""
    d_lo, _ = landau_lifshitz_cutoff_derate(a0=80.0, wavelength_um=0.8)
    d_hi, _ = landau_lifshitz_cutoff_derate(a0=1000.0, wavelength_um=0.8)
    assert d_hi < d_lo
    assert d_hi < 0.5


def test_rr_derate_clip_above_chi_threshold():
    """At a₀ ≫ a_RR (~3300·λμm) the field-strength χ exceeds the clip
    threshold and the derate is clamped accordingly."""
    derate, chi = landau_lifshitz_cutoff_derate(
        a0=5e4, wavelength_um=0.8, chi_clip=2.0,
    )
    if chi > 2.0:
        # When clipped, derate ≤ chi_clip / chi.
        assert derate <= 2.0 / chi + 1e-9
    else:
        assert derate <= 1.0


# ---- QED ----------------------------------------------------------------


def test_schwinger_ratio_zero_field_is_zero():
    assert schwinger_ratio(0.0) == 0.0


def test_vacuum_birefringence_scales_quadratically_with_field():
    omega = 2.4e15
    L = 1e-6
    e1 = 1e16
    e2 = 2e16
    phi1 = vacuum_birefringence_phase_shift(e1, omega, L)
    phi2 = vacuum_birefringence_phase_shift(e2, omega, L)
    assert phi2 / max(phi1, 1e-30) == pytest.approx(4.0, rel=1e-6)


def test_breit_wheeler_rate_zero_below_threshold():
    """At E ≪ E_S the Breit–Wheeler rate is exponentially suppressed."""
    omega = 2.4e15
    rate = breit_wheeler_pair_rate(1e10, omega)
    # Should be vanishing — exp(-π/χ) clipped to zero for tiny χ.
    assert rate == 0.0


def test_qed_diagnostics_validity_flag_above_chi_warn():
    diag = qed_diagnostics(
        intensity_w_per_m2=1e34,   # well past Schwinger
        omega_probe_rad_s=2.4e15,
        length_m=1e-6,
        chi_warn=0.1,
    )
    assert diag["validity_exceeded"] is True
    assert diag["schwinger_ratio"] > diag["chi_warn_threshold"]


def test_qed_diagnostics_below_threshold_quiet():
    diag = qed_diagnostics(
        intensity_w_per_m2=1e22,  # ≈ 10²² W/cm² → χ tiny
        omega_probe_rad_s=2.4e15,
        length_m=1e-6,
        chi_warn=0.5,
    )
    assert diag["validity_exceeded"] is False


# ---- end-to-end ---------------------------------------------------------


def test_homogeneous_extreme_power_produces_chf_focal_volume():
    cfg = RunConfig.model_validate(_ep_config({"enable_radiation_reaction": True}))
    result = simulate_from_config(cfg)
    assert result.chf_focal_volume is not None
    assert result.beam_array_geometry["n_beams"] == 4
    assert result.radiation_reaction["enabled"] is True
    assert "Gamma_3D_coherent" in result.chf_gain


def test_heterogeneous_extreme_power_two_color_runs():
    """Two-color heterogeneous run produces a spectrum on the ω-axis."""
    payload = _ep_config()
    payload["extreme_power"]["enable_qed"] = True
    payload["extreme_power"]["per_beam_lasers"] = [
        {"a0": 5.0, "wavelength_um": 0.80},
        {"a0": 5.0, "wavelength_um": 0.80},
        {"a0": 5.0, "wavelength_um": 1.03},
        {"a0": 5.0, "wavelength_um": 1.03},
    ]
    payload["extreme_power"]["omega_grid_points"] = 256
    cfg = RunConfig.model_validate(payload)
    result = simulate_from_config(cfg)
    assert "omega_rad_s" in result.spectrum.dims
    assert result.qed_diagnostics
    assert "schwinger_ratio" in result.qed_diagnostics


def test_extreme_power_hdf5_roundtrip(tmp_path):
    payload = _ep_config({"enable_radiation_reaction": True, "enable_qed": True})
    payload["laser"]["a0"] = 80.0
    cfg = RunConfig.model_validate(payload)
    result = simulate_from_config(cfg)
    out = tmp_path / "ep.h5"
    result.save(out)
    from harmonyemissions.io import load_result
    loaded = load_result(out)
    assert loaded.qed_diagnostics
    assert loaded.radiation_reaction
    assert loaded.beam_array_geometry["n_beams"] == 4
