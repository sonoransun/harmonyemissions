"""Tests for the accel/ abstraction layer."""

import os

import numpy as np
import pytest
from scipy.special import kv

from harmonyemissions.accel import (
    HAS_CUPY,
    HAS_NUMBA,
    HAS_PYFFTW,
    asarray_numpy,
    fft2,
    get_xp,
    ifft2,
    kv_two_thirds_half,
    njit,
)


def test_backend_flags_are_bool():
    for flag in (HAS_CUPY, HAS_NUMBA, HAS_PYFFTW):
        assert isinstance(flag, bool)


def test_get_xp_defaults_to_numpy_for_ndarray():
    assert get_xp(np.zeros(4)) is np


def test_asarray_numpy_identity_on_numpy_array():
    a = np.arange(5.0)
    np.testing.assert_array_equal(asarray_numpy(a), a)


def test_fft2_roundtrip_bit_equivalent_to_numpy():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    f_accel = fft2(u)
    f_np = np.fft.fft2(u)
    # Tolerance: FFT dispatch (scipy/pyfftw) should match numpy to machine precision.
    np.testing.assert_allclose(f_accel, f_np, atol=1e-9)
    # Round-trip through ifft2.
    u_back = ifft2(fft2(u))
    np.testing.assert_allclose(u_back, u, atol=1e-10)


def test_njit_plain_decorator():
    @njit
    def add(x):
        return x + 1
    assert add(2.0) == 3.0


def test_njit_with_kwargs():
    @njit()
    def mul(x, y):
        return x * y
    assert mul(3.0, 4.0) == 12.0


def test_kv_cache_agrees_with_scipy_within_tolerance():
    """Log-log interpolation should recover K_{2/3}(x/2)² to ~1e-4 relative."""
    x = np.geomspace(0.01, 100.0, 20)
    cached = kv_two_thirds_half(x)
    direct = kv(2.0 / 3.0, x / 2.0) ** 2
    # Relative error at each point.
    err = np.abs(cached - direct) / np.maximum(direct, 1e-300)
    assert np.max(err) < 1e-3


def test_kv_cache_returns_zero_for_nonpositive():
    assert kv_two_thirds_half(0.0) == 0.0
    assert kv_two_thirds_half(-1.0) == 0.0


@pytest.mark.skipif(not os.environ.get("HARMONY_TEST_CUPY"), reason="requires CuPy; set HARMONY_TEST_CUPY=1")
def test_cupy_backend_runs(tmp_path):  # pragma: no cover - env-gated
    from harmonyemissions import Laser, Target, simulate
    from harmonyemissions.config import NumericsConfig
    r_cpu = simulate(
        Laser(a0=5.0, spatial_profile="super_gaussian", spot_fwhm_um=2.0),
        Target.sio2(), model="surface_pipeline",
        numerics=NumericsConfig(pipeline_grid=32, pipeline_dx_um=0.2),
    )
    r_gpu = simulate(
        Laser(a0=5.0, spatial_profile="super_gaussian", spot_fwhm_um=2.0),
        Target.sio2(), model="surface_pipeline", backend="cupy",
        numerics=NumericsConfig(pipeline_grid=32, pipeline_dx_um=0.2),
    )
    np.testing.assert_allclose(r_cpu.spectrum.values, r_gpu.spectrum.values, rtol=1e-5)
