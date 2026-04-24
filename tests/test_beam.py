"""Tests for the 2-D beam profiles and Fraunhofer propagation."""

import numpy as np
import pytest

from harmonyemissions.beam import (
    SpatialGrid,
    build_profile,
    fraunhofer,
    fwhm_spot_size,
    gaussian_spot,
    intensity,
    inverse_fraunhofer,
    super_gaussian_spot,
    top_hat_spot,
)


def test_gaussian_fwhm_matches_input():
    g = SpatialGrid(n=256, dx=0.05e-6)  # 50 nm/pixel → 12.8 μm window
    u = gaussian_spot(g, fwhm_m=2e-6)
    fwhm = fwhm_spot_size(u, g.dx)
    assert abs(fwhm - 2e-6) / 2e-6 < 0.05


def test_super_gaussian_converges_to_top_hat():
    g = SpatialGrid(n=256, dx=0.05e-6)
    u2 = super_gaussian_spot(g, 2e-6, order=2)
    u8 = super_gaussian_spot(g, 2e-6, order=8)
    # Measure "flatness": ratio of intensity at centre to at FWHM radius.
    i2, i8 = intensity(u2), intensity(u8)
    # Higher-order should have flatter top: ratio of (I at 80% of FWHM) to peak
    # should be closer to 1 for order 8 than for order 2.
    c = g.n // 2
    r_idx = int(0.4 * 2e-6 / g.dx)  # 80% of FWHM/2 from centre
    assert i8[c, c + r_idx] / i8.max() > i2[c, c + r_idx] / i2.max()


def test_top_hat_has_correct_area():
    g = SpatialGrid(n=512, dx=0.05e-6)
    u = top_hat_spot(g, diameter_m=3e-6)
    area = float(np.sum(intensity(u))) * g.dx ** 2
    expected = np.pi * (1.5e-6) ** 2
    assert abs(area - expected) / expected < 0.05


def test_build_profile_dispatch():
    g = SpatialGrid(n=64, dx=0.1e-6)
    for name in ["gaussian", "super_gaussian", "top_hat", "jinc"]:
        u = build_profile(name, g, 1e-6, super_gaussian_order=6)
        assert u.shape == (64, 64)
        assert intensity(u).max() > 0

    with pytest.raises(ValueError):
        build_profile("not-a-profile", g, 1e-6)


def test_fraunhofer_preserves_energy():
    g = SpatialGrid(n=128, dx=0.1e-6)
    u0 = gaussian_spot(g, 2e-6)
    u_far, dx_far = fraunhofer(u0, g.dx, wavelength=8e-7, z=1.0)
    power0 = np.sum(intensity(u0)) * g.dx ** 2
    power_far = np.sum(intensity(u_far)) * dx_far ** 2
    assert abs(power_far - power0) / power0 < 1e-6


def test_fraunhofer_roundtrip():
    g = SpatialGrid(n=128, dx=0.1e-6)
    u0 = gaussian_spot(g, 2e-6)
    u_far, dx_far = fraunhofer(u0, g.dx, 8e-7, 1.0)
    u_back, dx_back = inverse_fraunhofer(u_far, dx_far, 8e-7, 1.0)
    # Pixel size should match within floating-point rounding.
    assert abs(dx_back - g.dx) / g.dx < 1e-10
    # Intensity patterns should agree well.
    i0, ib = intensity(u0), intensity(u_back)
    scale = ib.max() / i0.max()
    assert abs(scale - 1.0) < 1e-3
    assert np.corrcoef(i0.ravel(), ib.ravel())[0, 1] > 0.999


def test_fraunhofer_top_hat_gives_airy_pattern():
    """Top-hat aperture → far-field should look like an Airy pattern."""
    g = SpatialGrid(n=256, dx=0.05e-6)
    u = top_hat_spot(g, diameter_m=4e-6)
    u_far, _ = fraunhofer(u, g.dx, 8e-7, 0.1)
    i_far = intensity(u_far)
    # Peak should be at centre.
    c = g.n // 2
    assert i_far[c, c] == i_far.max()
    # First ring: radial intensity crosses zero around θ ≈ 1.22 λ/D.
    # We just check that we have a strong central lobe > 10x the off-axis mean.
    ring_mean = i_far[c, c + 50 : c + 80].mean() if c + 80 < g.n else i_far[-1].mean()
    assert i_far[c, c] > 10.0 * ring_mean
