"""Ross-pair differential filter isolator.

Standard hard-X-ray spectroscopy technique: two matched-thickness
filters whose K-edges straddle the energy window of interest.
Taking the *difference* of the two filtered signals subtracts the
common continuum and isolates the narrow passband between the edges.

Typical pairs
-------------
* Cu (K = 8.98 keV) / Ni (K = 8.33 keV) — Cu-K spectroscopy
* Mo (K = 20.0 keV) / Zr via Ag surrogate — Mo-K spectroscopy
* Sn (K = 29.2 keV) / Ag (K = 25.5 keV) — low-Z Compton K-windows
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.detector.filters import FilterSpec, transmission


@dataclass(frozen=True)
class RossPair:
    """High-Z / low-Z filter pair with adjacent K-edges."""

    high_z: FilterSpec
    low_z: FilterSpec
    passband_label: str | None = None


def ross_pair(
    spectrum_keV: xr.DataArray,
    pair: RossPair,
) -> dict[str, xr.DataArray | tuple[float, float]]:
    """Split a spectrum through the Ross pair and return the differential.

    Requires a ``photon_energy_keV`` coordinate on ``spectrum_keV``.

    Returns a dict with:
      * ``through_high`` — spectrum transmitted through the high-Z filter
      * ``through_low``  — spectrum transmitted through the low-Z filter
      * ``difference``   — ``through_low − through_high`` (positive in the
        Ross pair passband)
      * ``passband_ev``  — ``(E_lo, E_hi)`` energies where |difference|
        > 10 % of its peak.
    """
    if "photon_energy_keV" not in spectrum_keV.coords:
        raise ValueError(
            "ross_pair requires a 'photon_energy_keV' coord on the input spectrum"
        )
    energy_keV = np.asarray(spectrum_keV.coords["photon_energy_keV"].values, dtype=float)
    energy_ev = energy_keV * 1e3
    T_high = transmission(pair.high_z.material, energy_ev, pair.high_z.thickness_um)
    T_low = transmission(pair.low_z.material, energy_ev, pair.low_z.thickness_um)
    through_high = spectrum_keV.values * T_high
    through_low = spectrum_keV.values * T_low
    diff = through_low - through_high

    peak = float(np.abs(diff).max()) if diff.size else 0.0
    if peak > 0:
        mask = np.abs(diff) > 0.1 * peak
        pb = (float(energy_ev[mask].min()), float(energy_ev[mask].max()))
    else:
        pb = (0.0, 0.0)

    def _wrap(data: np.ndarray, label: str) -> xr.DataArray:
        da = spectrum_keV.copy(data=data)
        da.attrs["ross_pair"] = label
        return da

    return {
        "through_high": _wrap(through_high, f"high={pair.high_z.material}"),
        "through_low": _wrap(through_low, f"low={pair.low_z.material}"),
        "difference": _wrap(
            diff, f"{pair.low_z.material}−{pair.high_z.material}"
        ),
        "passband_ev": pb,
    }
