"""Tests for the matplotlib plot helpers."""

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from harmonyemissions import Laser, Target, simulate  # noqa: E402
from harmonyemissions.config import NumericsConfig  # noqa: E402
from harmonyemissions.viz import (  # noqa: E402
    plot_beam_profile,
    plot_chf_gain,
    plot_dent_map,
    plot_instrument_spectrum,
    plot_pulse,
    plot_scaling,
    plot_spectrum,
)


@pytest.fixture
def bgp_result():
    return simulate(Laser(a0=10.0), Target.overdense(100.0), model="bgp")


@pytest.fixture
def rom_result():
    return simulate(
        Laser(a0=10.0),
        Target.overdense(100.0, 0.05),
        model="rom",
        numerics=NumericsConfig(harmonic_window=(10.0, 30.0)),
    )


@pytest.fixture
def pipeline_result():
    return simulate(
        Laser(a0=10.0, spatial_profile="super_gaussian", spot_fwhm_um=2.0),
        Target.sio2(t_HDR_fs=351.0),
        model="surface_pipeline",
        numerics=NumericsConfig(pipeline_grid=64, pipeline_dx_um=0.1),
    )


def test_plot_spectrum_non_empty(bgp_result):
    fig, ax = plt.subplots()
    plot_spectrum(bgp_result, ax=ax)
    assert ax.lines  # at least one line drawn
    plt.close(fig)


def test_plot_pulse_on_rom(rom_result):
    fig, ax = plt.subplots()
    plot_pulse(rom_result, ax=ax)
    assert ax.lines
    plt.close(fig)


def test_plot_pulse_on_bgp_raises(bgp_result):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="no time-domain field"):
        plot_pulse(bgp_result, ax=ax)
    plt.close(fig)


def test_plot_dent_map_on_pipeline(pipeline_result):
    fig, ax = plt.subplots()
    plot_dent_map(pipeline_result, ax=ax)
    assert ax.images  # imshow drawn
    plt.close(fig)


def test_plot_dent_map_on_bgp_raises(bgp_result):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="no dent_map"):
        plot_dent_map(bgp_result, ax=ax)
    plt.close(fig)


def test_plot_beam_profile_near(pipeline_result):
    fig, ax = plt.subplots()
    plot_beam_profile(pipeline_result, which="near", ax=ax)
    assert ax.images
    plt.close(fig)


def test_plot_beam_profile_far(pipeline_result):
    fig, ax = plt.subplots()
    plot_beam_profile(pipeline_result, which="far", harmonic_idx=0, ax=ax)
    assert ax.images
    plt.close(fig)


def test_plot_beam_profile_bad_which(pipeline_result):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="which must be"):
        plot_beam_profile(pipeline_result, which="wrong", ax=ax)  # type: ignore[arg-type]
    plt.close(fig)


def test_plot_chf_gain(pipeline_result):
    fig, ax = plt.subplots()
    plot_chf_gain(pipeline_result, ax=ax)
    assert ax.patches  # bar chart patches
    plt.close(fig)


def test_plot_chf_gain_on_bgp_raises(bgp_result):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="no chf_gain"):
        plot_chf_gain(bgp_result, ax=ax)
    plt.close(fig)


def test_plot_instrument_requires_instrument_spectrum(bgp_result):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="no instrument_spectrum"):
        plot_instrument_spectrum(bgp_result, ax=ax)
    plt.close(fig)


def test_plot_scaling_reads_scan_dir(tmp_path, bgp_result):
    # Build a tiny "scan" by simulating a few a₀ values and saving.
    paths = []
    for a0 in [1.0, 3.0, 10.0]:
        r = simulate(Laser(a0=a0), Target.overdense(100.0), model="bgp")
        p = tmp_path / f"run_laser-a0={a0}.h5"
        r.save(p)
        paths.append(p)
    fig, ax = plt.subplots()
    plot_scaling(paths, param="a0", ax=ax)
    assert ax.lines
    plt.close(fig)
