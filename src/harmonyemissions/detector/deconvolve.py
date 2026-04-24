"""Pipeline: raw simulated spectrum → instrument-corrected detector signal.

Applies the full S(λ) response described in the paper's eq. 1:

    S(λ) = Al · QE · G · Al₂O₃ · CH

to turn an arbitrary-units simulated spectrum (``n`` → intensity) into a
CCD photon-count estimate. This is what you would see on an Andor DV436
behind a 300 l/mm grating and an aluminium filter stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.detector.al_filter import (
    al_filter_transmission,
    harmonic_to_wavelength_nm,
    hydrocarbon_correction,
    oxide_correction,
)
from harmonyemissions.detector.grating import SECOND_ORDER, THIRD_ORDER

# Paper's Andor DV436 quantum efficiency, piecewise linear in wavelength.
_QE_WAVELENGTH_NM = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
_QE_VALUES = np.array([0.30, 0.55, 0.88, 0.80, 0.60, 0.30])
# Flat-field grating G(λ) efficiency (first-order) — rises with wavelength in
# the aluminium pass band; simplified three-point fit.
_GRATING_WAVELENGTH_NM = np.array([10.0, 20.0, 40.0, 60.0, 100.0])
_GRATING_EFFICIENCY = np.array([0.02, 0.08, 0.12, 0.09, 0.04])


@dataclass(frozen=True)
class DetectorConfig:
    al_thickness_um: float = 1.5
    al2o3_thickness_nm: float = 9.0
    ch_thickness_nm: float = 5.0
    include_second_order: bool = True
    include_third_order: bool = False


def quantum_efficiency(wavelength_nm: np.ndarray) -> np.ndarray:
    return np.clip(
        np.interp(wavelength_nm, _QE_WAVELENGTH_NM, _QE_VALUES),
        0.0, 1.0,
    )


def grating_first_order(wavelength_nm: np.ndarray) -> np.ndarray:
    return np.clip(
        np.interp(wavelength_nm, _GRATING_WAVELENGTH_NM, _GRATING_EFFICIENCY),
        0.0, 1.0,
    )


def spectral_response(
    wavelength_nm: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    """S(λ) — full first-order instrument response, as in the paper's eq. 1."""
    al = al_filter_transmission(wavelength_nm, config.al_thickness_um)
    qe = quantum_efficiency(wavelength_nm)
    g = grating_first_order(wavelength_nm)
    ox = oxide_correction(wavelength_nm, config.al2o3_thickness_nm)
    ch = hydrocarbon_correction(wavelength_nm, config.ch_thickness_nm)
    return al * qe * g * ox * ch


def apply_instrument_response(
    harmonic_spectrum: xr.DataArray,
    wavelength_um_driver: float = 0.8,
    config: DetectorConfig | None = None,
) -> xr.DataArray:
    """Apply S(λ) and grating-order overlap to a simulated harmonic spectrum.

    Returns a new DataArray with the same ``harmonic`` coord whose values
    are the photon counts recorded at the CCD (up to an overall constant).
    """
    cfg = config or DetectorConfig()
    n = harmonic_spectrum.coords["harmonic"].values.astype(float)
    wl_nm = harmonic_to_wavelength_nm(n, wavelength_um_driver)
    s_lambda = spectral_response(wl_nm, cfg)
    signal = harmonic_spectrum.values * s_lambda

    # Grating higher-order contamination: O(n log n) pairing via searchsorted
    # rather than the O(n²) np.argmin loop.  For each harmonic n_i we locate
    # the nearest tabulated harmonic to `m · n_i` (m = 2, 3) in one pass.
    sort_idx = np.argsort(n)
    n_sorted = n[sort_idx]

    def _add_order(order: int, fit_fn) -> None:
        targets = order * n
        # Insertion indices; clip to valid range before the membership check.
        ins = np.searchsorted(n_sorted, targets)
        ins = np.clip(ins, 1, len(n_sorted) - 1)
        left = n_sorted[ins - 1]
        right = n_sorted[ins]
        use_left = np.abs(targets - left) <= np.abs(targets - right)
        nearest = np.where(use_left, left, right)
        j = np.where(use_left, sort_idx[ins - 1], sort_idx[ins])
        tol = np.abs(nearest - targets) / np.maximum(1.0, targets)
        mask = (tol < 0.02) & (j != np.arange(len(n)))
        if not np.any(mask):
            return
        contributions = np.asarray(fit_fn(nearest[mask])) * \
                        harmonic_spectrum.values[j[mask]] * s_lambda[j[mask]]
        np.add.at(signal, np.where(mask)[0], contributions)

    if cfg.include_second_order:
        _add_order(2, SECOND_ORDER)
    if cfg.include_third_order:
        _add_order(3, THIRD_ORDER)

    return xr.DataArray(
        signal,
        coords={"harmonic": n},
        dims=["harmonic"],
        name="instrument_spectrum",
        attrs={
            "units": "arb. CCD counts (proportional)",
            "al_thickness_um": cfg.al_thickness_um,
            "al2o3_thickness_nm": cfg.al2o3_thickness_nm,
            "ch_thickness_nm": cfg.ch_thickness_nm,
        },
    )
