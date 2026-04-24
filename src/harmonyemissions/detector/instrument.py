"""Band-aware detector dispatcher.

Picks the correct instrument response stack for a spectrum based on
its photon-energy span:

* ``xuv``       — 10 eV – 100 eV         (Al filter + Andor grating + CCD)
* ``xray-soft`` — 100 eV – 1 keV         (Mylar/Kapton + Ni/Au grating + BI-CCD)
* ``xray-hard`` — 1 keV – 100 keV        (K-edge filters + Si/CdTe absorber)
* ``gamma``     — 100 keV – 10 MeV       (filter + scintillator + Compton correction)

A user can force a band via the ``band`` argument.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from harmonyemissions.detector.deconvolve import DetectorConfig, apply_instrument_response
from harmonyemissions.detector.gamma_response import GammaDetector, apply_gamma_response
from harmonyemissions.detector.hard_xray import FilterStack
from harmonyemissions.detector.scintillator import DetectorConfig as ScintConfig
from harmonyemissions.detector.soft_xray import SoftXrayConfig, apply_soft_xray_response

BANDS: dict[str, tuple[float, float]] = {
    "xuv":       (10.0,      100.0),
    "xray-soft": (100.0,     1_000.0),
    "xray-hard": (1_000.0,   100_000.0),
    "gamma":     (100_000.0, 1.0e7),
}


def auto_band(spectrum: xr.DataArray, driver_wavelength_um: float = 0.8) -> str:
    """Pick the band whose energy range best contains the spectral peak."""
    if "photon_energy_keV" in spectrum.coords:
        energy_ev = np.asarray(spectrum.coords["photon_energy_keV"].values) * 1e3
    else:
        # Fall back to harmonic × (hc/λ) in eV; λ_μm → nm for hc=1239.84 eV·nm.
        n = np.asarray(spectrum.coords["harmonic"].values)
        energy_ev = n * (1239.84198 / (driver_wavelength_um * 1000.0))
    s = np.asarray(spectrum.values, dtype=float)
    if s.max() <= 0:
        center = float(np.median(energy_ev))
    else:
        center = float(energy_ev[int(np.argmax(s))])
    for name, (lo, hi) in BANDS.items():
        if lo <= center < hi:
            return name
    # Below the XUV band (sub-10 eV harmonics): treat as XUV; above gamma: gamma.
    lowest = min(lo for lo, _ in BANDS.values())
    return "xuv" if center < lowest else "gamma"


def apply_detector(
    spectrum: xr.DataArray,
    driver_wavelength_um: float = 0.8,
    band: str | None = None,
    *,
    xuv_config: DetectorConfig | None = None,
    soft_config: SoftXrayConfig | None = None,
    hard_filters: FilterStack | None = None,
    hard_detector: ScintConfig | None = None,
    gamma_detector: GammaDetector | None = None,
) -> xr.DataArray:
    """Dispatch to the correct band-specific response and return
    ``instrument_spectrum`` — a DataArray on the same harmonic dim.

    The XUV path is the legacy Al + grating + CCD pipeline (soft-XUV
    backwards compatibility).  Soft-X-ray is the new polyimide / Ni-Au
    grating stack.  Hard-X-ray and γ paths go through the shared
    filter + scintillator / semiconductor absorber.
    """
    resolved = auto_band(spectrum, driver_wavelength_um) if (band is None or band == "auto") else band
    if resolved == "xuv":
        return apply_instrument_response(
            spectrum, driver_wavelength_um, xuv_config
        )
    if resolved == "xray-soft":
        if "photon_energy_keV" not in spectrum.coords:
            raise ValueError(
                "xray-soft band requires a 'photon_energy_keV' coord"
            )
        return apply_soft_xray_response(spectrum, soft_config)
    if resolved in ("xray-hard", "gamma"):
        # Both share the γ pipeline; the hard-X-ray filters/detector just
        # default to different thicknesses.
        det = gamma_detector or GammaDetector(
            filters=hard_filters or FilterStack(layers=(("Al", 500.0),)),
            detector=hard_detector or ScintConfig(name="Si" if resolved == "xray-hard" else "CsI"),
        )
        return apply_gamma_response(spectrum, det)
    raise ValueError(f"unknown band {resolved!r}")


def provenance(spectrum: xr.DataArray, band: str, **kwargs: Any) -> dict[str, Any]:
    """Return a provenance dict suitable for Result.attrs / diagnostics."""
    return {
        "band": band,
        "harmonic_len": int(spectrum.sizes.get("harmonic", 0)),
        **kwargs,
    }
