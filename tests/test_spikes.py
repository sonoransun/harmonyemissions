"""Tests for the relativistic-spikes spatial filter S(n, a₀)."""

import numpy as np
import pytest

from harmonyemissions.emission.spikes import (
    CUTOFF_PREFACTOR,
    CutoffMode,
    relativistic_spikes_filter,
    spikes_cutoff_harmonic,
    universal_envelope,
)


def test_cutoff_scales_as_a0_cubed():
    a0s = np.array([1.0, 3.0, 10.0, 30.0])
    nc = spikes_cutoff_harmonic(a0s)
    # ratios should be a0³ within machine precision.
    np.testing.assert_allclose(nc / nc[0], (a0s / a0s[0]) ** 3, rtol=1e-12)
    # Prefactor matches BGP.
    np.testing.assert_allclose(nc, CUTOFF_PREFACTOR * a0s ** 3, rtol=1e-12)


def test_logistic_midpoint_at_ncutoff():
    """S at n = n_c should evaluate to 0.5 · plateau_at_nc."""
    a0 = 10.0
    nc = float(spikes_cutoff_harmonic(a0))
    s_at_nc = float(relativistic_spikes_filter(nc, a0, mode=CutoffMode.LOGISTIC))
    # At n = n_c: plateau factor is 1, logistic factor is 0.5 → S = 0.5.
    assert abs(s_at_nc - 0.5) < 1e-6


def test_sharp_mode_is_heaviside():
    a0 = 5.0
    nc = float(spikes_cutoff_harmonic(a0))
    below = float(relativistic_spikes_filter(0.5 * nc, a0, mode=CutoffMode.SHARP))
    above = float(relativistic_spikes_filter(1.5 * nc, a0, mode=CutoffMode.SHARP))
    assert below > 0
    assert above == 0.0


def test_plateau_slope_is_minus_eight_thirds():
    """Below cutoff, log-log slope of S(n) vs n should be −8/3."""
    a0 = 50.0
    nc = float(spikes_cutoff_harmonic(a0))
    n = np.logspace(0, np.log10(nc * 0.05), 20)  # deep in plateau
    s = relativistic_spikes_filter(n, a0, mode=CutoffMode.EXPONENTIAL)
    slope = np.polyfit(np.log(n), np.log(s), 1)[0]
    assert abs(slope - (-8.0 / 3.0)) < 0.02


def test_universal_envelope_positive_and_decreasing():
    a0 = 10.0
    n = np.arange(2.0, 3.0 * spikes_cutoff_harmonic(a0))
    env = universal_envelope(n, a0)
    assert np.all(env > 0)
    assert env[0] > env[-1]


def test_spatial_map_of_a0():
    """Pass a 2-D map of a₀ and a scalar n; output shape matches the map."""
    a0_map = np.linspace(1.0, 10.0, 100).reshape(10, 10)
    s = relativistic_spikes_filter(30, a0_map, mode=CutoffMode.LOGISTIC)
    assert s.shape == a0_map.shape
    # Higher a₀ → further from cutoff at n=30 → higher S.
    assert s[-1, -1] > s[0, 0]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        relativistic_spikes_filter(1, 1.0, mode="bogus")  # type: ignore[arg-type]
