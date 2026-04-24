"""Performance regressions via pytest-benchmark.

These benchmarks run only when you opt in with ``pytest -m benchmark``.
They are not part of the default suite; their job is to catch a ≥20 %
regression in any of the hot paths. The baseline is established by the
first `--benchmark-autosave` run; subsequent `--benchmark-compare` runs
refuse to merge if a mean time regresses beyond threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.accel.fft import fft2
from harmonyemissions.beam import SpatialGrid, gaussian_spot
from harmonyemissions.config import NumericsConfig
from harmonyemissions.detector.deconvolve import apply_instrument_response

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Individual models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def overdense_target():
    return Target.overdense(100.0, 0.05)


@pytest.fixture(scope="module")
def gas_target():
    return Target.gas("Ar")


@pytest.fixture(scope="module")
def underdense_target():
    return Target.underdense(0.001, electron_energy_mev=500.0)


def test_bench_bgp(benchmark, overdense_target):
    laser = Laser(a0=10.0)
    benchmark(simulate, laser, overdense_target, model="bgp")


def test_bench_rom(benchmark, overdense_target):
    laser = Laser(a0=10.0)
    benchmark(simulate, laser, overdense_target, model="rom")


def test_bench_cse(benchmark, overdense_target):
    laser = Laser(a0=10.0)
    benchmark(simulate, laser, overdense_target, model="cse")


def test_bench_lewenstein(benchmark, gas_target):
    laser = Laser(a0=0.08, duration_fs=20.0)
    # Warmup to amortise the one-time numba compile.
    simulate(laser, gas_target, model="lewenstein")
    benchmark(simulate, laser, gas_target, model="lewenstein")


def test_bench_betatron(benchmark, underdense_target):
    laser = Laser(a0=2.0)
    benchmark(simulate, laser, underdense_target, model="betatron")


# ---------------------------------------------------------------------------
# Surface pipeline at three grid sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [32, 64, 128])
def test_bench_surface_pipeline(benchmark, grid):
    laser = Laser(a0=10.0, spatial_profile="super_gaussian", spot_fwhm_um=2.0, angle_deg=45.0)
    target = Target.sio2(t_HDR_fs=351.0)
    numerics = NumericsConfig(pipeline_grid=grid, pipeline_dx_um=0.1,
                              diag_harmonics=(1, 15, 30, 45))
    benchmark(simulate, laser, target, model="surface_pipeline", numerics=numerics)


# ---------------------------------------------------------------------------
# Fraunhofer 2-D FFT standalone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [64, 128, 256])
def test_bench_fft2(benchmark, n):
    rng = np.random.default_rng(0)
    u = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex128)
    benchmark(fft2, u)


def test_bench_fraunhofer_full(benchmark):
    from harmonyemissions.beam import fraunhofer as _fraunhofer
    grid = SpatialGrid(n=128, dx=0.1e-6)
    u0 = gaussian_spot(grid, 2e-6)
    benchmark(_fraunhofer, u0, grid.dx, 8e-7, 1e-2)


# ---------------------------------------------------------------------------
# Detector pipeline
# ---------------------------------------------------------------------------


def test_bench_detector_response(benchmark, overdense_target):
    r = simulate(Laser(a0=10.0), overdense_target, model="bgp")
    benchmark(apply_instrument_response, r.spectrum, 0.8)
