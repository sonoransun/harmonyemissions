import numpy as np
import pytest

from harmonyemissions.laser import Laser


def test_laser_field_amplitude_matches_a0():
    laser = Laser(a0=3.0, wavelength_um=0.8, duration_fs=10.0, envelope="gaussian")
    t = laser.time_grid(n_periods=30.0, samples_per_period=256)
    e = laser.field(t)
    # Peak |E| must be close to a₀ at pulse center.
    assert np.max(np.abs(e)) == pytest.approx(3.0, rel=5e-2)


def test_envelope_fwhm_gaussian():
    laser = Laser(a0=1.0, wavelength_um=0.8, duration_fs=5.0, envelope="gaussian")
    T0 = laser.units.period_s
    t = laser.time_grid(n_periods=30.0, samples_per_period=512)
    env = laser.envelope_value(t)
    # Find FWHM of intensity (env²).
    ints = env ** 2
    half = ints >= 0.5 * ints.max()
    fwhm_T0 = (t[half][-1] - t[half][0])
    fwhm_fs = fwhm_T0 * T0 * 1e15
    assert fwhm_fs == pytest.approx(5.0, rel=0.05)


def test_flat_top_and_sin2_shapes():
    t = np.linspace(-10, 10, 4096)
    laser_flat = Laser(a0=1.0, duration_fs=1e15 * 5 * 2.67e-15, envelope="flat-top")
    laser_s2 = Laser(a0=1.0, duration_fs=1e15 * 5 * 2.67e-15, envelope="sin2")
    env_flat = laser_flat.envelope_value(t)
    env_s2 = laser_s2.envelope_value(t)
    assert 0 <= env_flat.min() and env_flat.max() == pytest.approx(1.0)
    assert 0 <= env_s2.min() and env_s2.max() == pytest.approx(1.0)


def test_cep_shifts_field():
    laser_0 = Laser(a0=1.0, cep=0.0)
    laser_p = Laser(a0=1.0, cep=3.14159 / 2)
    t = laser_0.time_grid(5, 256)
    e0 = laser_0.field(t)
    ep = laser_p.field(t)
    # Different CEPs should produce different fields.
    assert np.any(np.abs(e0 - ep) > 1e-3)


def test_from_intensity():
    laser = Laser.from_intensity(1e19, wavelength_um=0.8)
    # a₀ = √(I λ²/1.37e18) → for I=1e19, λ=0.8 → a₀ ≈ √(6.4/1.37) ≈ 2.16
    assert 2.0 < laser.a0 < 2.3
