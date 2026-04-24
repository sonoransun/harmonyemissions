"""End-to-end tests for the Timmis 2026 surface pipeline."""

import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.config import NumericsConfig


@pytest.fixture(scope="module")
def gemini_result():
    laser = Laser(
        a0=24.0, wavelength_um=0.8, duration_fs=50.0,
        spatial_profile="super_gaussian", spot_fwhm_um=2.0,
        super_gaussian_order=8, angle_deg=45.0,
    )
    target = Target.sio2(t_HDR_fs=351.0,
                         prepulse_intensity_rel=1e-3, prepulse_delay_fs=100.0)
    numerics = NumericsConfig(pipeline_grid=64, pipeline_dx_um=0.1)
    return simulate(laser, target, model="surface_pipeline", numerics=numerics)


def test_populates_all_pipeline_outputs(gemini_result):
    assert gemini_result.spectrum is not None
    assert gemini_result.dent_map is not None
    assert gemini_result.beam_profile_near is not None
    assert gemini_result.beam_profile_far is not None
    assert gemini_result.chf_gain


def test_chf_gamma_3d_is_gamma_2d_squared(gemini_result):
    g = gemini_result.chf_gain
    assert g["Gamma_3D"] == pytest.approx(g["Gamma_2D"] ** 2, rel=1e-12)


def test_chf_gamma_total_is_d_times_3d(gemini_result):
    g = gemini_result.chf_gain
    assert g["Gamma_total"] == pytest.approx(g["Gamma_D"] * g["Gamma_3D"], rel=1e-12)


def test_plateau_slope_matches_bgp(gemini_result):
    slope, _ = gemini_result.fit_power_law(n_min=5.0, n_max=60.0)
    assert abs(slope - (-8.0 / 3.0)) < 0.08


def test_dent_map_peak_non_zero(gemini_result):
    assert gemini_result.dent_map.max() > 0.05  # at least 5 % of a wavelength
    assert gemini_result.dent_map.max() < 5.0


def test_n_cutoff_scales_as_a0_cubed():
    """n_cutoff_spikes reported on the pipeline result must scale ∝ a₀³."""
    numerics = NumericsConfig(pipeline_grid=32, pipeline_dx_um=0.2)
    runs = []
    for a0 in [5.0, 10.0, 20.0]:
        r = simulate(
            Laser(a0=a0, spatial_profile="super_gaussian", spot_fwhm_um=2.0),
            Target.sio2(),
            model="surface_pipeline",
            numerics=numerics,
        )
        runs.append(r.diagnostics["n_cutoff_spikes"])
    ratio_lo = runs[1] / runs[0]
    ratio_hi = runs[2] / runs[1]
    assert ratio_lo == pytest.approx(8.0, rel=1e-6)
    assert ratio_hi == pytest.approx(8.0, rel=1e-6)


def test_spectrum_positive_and_finite(gemini_result):
    import numpy as np
    s = gemini_result.spectrum.values
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0)
