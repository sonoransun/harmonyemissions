"""Integrated hard-X-ray / γ-ray instrument response.

Chains

    S(E) = FilterStack.T(E) · DetectorConfig.ε(E)

and applies it to a simulation ``Result`` whose ``spectrum`` carries a
``photon_energy_keV`` coordinate (betatron, ICS, Bethe–Heitler,
Kramers-Maxwell bremsstrahlung, K-α).  Produces an
``instrument_spectrum`` DataArray with CCD-equivalent deposited-energy
counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from harmonyemissions.detector.hard_xray import FilterStack
from harmonyemissions.detector.scintillator import (
    DetectorConfig,
    detector_response,
)


@dataclass
class GammaDetector:
    """Composite filter-stack + active-layer detector."""

    filters: FilterStack = field(
        default_factory=lambda: FilterStack(layers=(("Al", 500.0),))
    )
    detector: DetectorConfig = field(default_factory=DetectorConfig)


def spectral_response(
    energy_keV: np.ndarray, detector: GammaDetector | None = None
) -> np.ndarray:
    """Full response S(E) = T_filter(E) · ε_detector(E)."""
    det = detector or GammaDetector()
    T = det.filters.transmission(energy_keV)
    eps = detector_response(energy_keV, det.detector)
    return T * eps


def apply_gamma_response(
    harmonic_spectrum: xr.DataArray,
    detector: GammaDetector | None = None,
) -> xr.DataArray:
    """Apply S(E) to a spectrum carrying a ``photon_energy_keV`` coord.

    Returns a DataArray indexed by the same coord with ``instrument_spectrum``
    values (photon counts × deposited-energy fraction, up to an overall
    normalisation).
    """
    if "photon_energy_keV" not in harmonic_spectrum.coords:
        raise ValueError(
            "apply_gamma_response requires a 'photon_energy_keV' coord on the "
            "input spectrum. The betatron / bremsstrahlung / ICS models all "
            "populate it; re-run the simulation if you're on an older file."
        )
    energy_keV = harmonic_spectrum.coords["photon_energy_keV"].values.astype(float)
    response = spectral_response(energy_keV, detector)
    signal = harmonic_spectrum.values * response
    det = detector or GammaDetector()
    return xr.DataArray(
        signal,
        coords=harmonic_spectrum.coords,
        dims=harmonic_spectrum.dims,
        name="instrument_spectrum",
        attrs={
            "units": "arb. (deposited-energy counts)",
            "filters": repr(det.filters.layers),
            "detector_name": det.detector.name,
            "detector_thickness_mm": det.detector.thickness_mm,
        },
    )
