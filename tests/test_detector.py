"""Tests for the XUV detector / instrument-response modeling."""

import numpy as np
import pytest
import xarray as xr

from harmonyemissions.detector.al_filter import (
    al_filter_transmission,
    harmonic_to_wavelength_nm,
)
from harmonyemissions.detector.deconvolve import (
    DetectorConfig,
    apply_instrument_response,
    spectral_response,
)
from harmonyemissions.detector.grating import (
    deconvolve_second_order,
    grating_order_ratio,
)


def test_second_order_logistic_matches_paper():
    """At n = 40 (midpoint) the logistic should equal ν/2 + b = 0.58."""
    assert grating_order_ratio(40, 2) == pytest.approx(0.92 / 2 + 0.12, abs=1e-6)


def test_third_order_logistic_matches_paper():
    """At n = 44 (midpoint) → 0.79/2 + 0.13 = 0.525."""
    assert grating_order_ratio(44, 3) == pytest.approx(0.79 / 2 + 0.13, abs=1e-6)


def test_grating_monotone_in_harmonic():
    ns = np.arange(5, 80)
    r2 = grating_order_ratio(ns, 2)
    # Logistic → monotonically increasing.
    assert np.all(np.diff(r2) >= 0)


def test_only_2nd_and_3rd_orders_parametrised():
    with pytest.raises(ValueError):
        grating_order_ratio(30, 5)


def test_al_filter_blocks_below_L_edge():
    t = al_filter_transmission(np.array([10.0, 15.0]), thickness_um=1.0)
    assert np.all(t == 0.0)


def test_al_filter_transmits_in_band():
    t_thin = al_filter_transmission(np.array([40.0]), thickness_um=0.2)
    t_thick = al_filter_transmission(np.array([40.0]), thickness_um=3.0)
    assert t_thin[0] > t_thick[0]
    assert 0.0 < t_thin[0] <= 1.0


def test_harmonic_to_wavelength():
    w = harmonic_to_wavelength_nm(np.array([10, 20, 40]), wavelength_um_driver=0.8)
    np.testing.assert_allclose(w, [80.0, 40.0, 20.0])


def test_spectral_response_shape():
    w = np.linspace(17.0, 70.0, 20)
    r = spectral_response(w, DetectorConfig())
    assert np.all(r >= 0)
    # Peak response should be somewhere in the bulk of the Al pass band.
    peak_idx = int(np.argmax(r))
    assert 20.0 < w[peak_idx] < 60.0


def test_apply_instrument_response_preserves_coords():
    n = np.arange(10, 51)
    vals = np.ones_like(n, dtype=float)
    raw = xr.DataArray(vals, coords={"harmonic": n}, dims=["harmonic"])
    sig = apply_instrument_response(raw, wavelength_um_driver=0.8)
    np.testing.assert_array_equal(sig.coords["harmonic"].values, n)
    assert sig.attrs["al_thickness_um"] == DetectorConfig().al_thickness_um
    assert np.all(sig.values >= 0)


def test_deconvolve_second_order_cancels_known_overlap():
    """Construct a spectrum with only 1st-order content, contaminate it with
    2nd-order, deconvolve, and check that we recover the clean 1st-order."""
    n = np.arange(10, 60)
    clean = np.exp(-0.05 * n)
    contaminated = clean.copy()
    for i, ni in enumerate(n):
        j = int(np.argmin(np.abs(n - 2 * ni)))
        if abs(n[j] - 2 * ni) / (2 * ni) < 0.02 and j != i:
            contaminated[i] = contaminated[i] + float(grating_order_ratio(2 * ni, 2)) * clean[j]
    recovered = deconvolve_second_order(n, contaminated)
    # In the low-n region, 2n still falls inside the array so the deconvolution
    # has something to subtract. Recovery should be tight (<2 %).
    mask = (n >= 10) & (n <= 29)
    err = np.abs(recovered[mask] - clean[mask]) / clean[mask]
    assert err.max() < 0.02
