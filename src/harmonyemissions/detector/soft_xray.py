"""Soft-X-ray (100 eV – 1 keV) detector stack.

Pipeline
--------
spectrum → filter chain (Mylar/Kapton/Al) → transmission grating (Ni/Au)
          → back-illuminated CCD QE → instrument_spectrum

Everything is energy-based (not wavelength) for continuity with the
hard-X-ray / γ-band stacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.detector.filters import FilterSpec, transmission

# Back-illuminated CCD QE, tabulated from 100 eV – 1 keV (Andor Newton
# BI-CCD at soft X-ray). Values are representative, not vendor-pinned.
_BI_CCD_ENERGY_EV = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 1000.0])
_BI_CCD_QE = np.array([0.25, 0.70, 0.85, 0.90, 0.85, 0.70])


@dataclass(frozen=True)
class SoftXrayConfig:
    filters: tuple[FilterSpec, ...] = (FilterSpec("kapton", 7.0),)
    grating_material: str = "ni"          # "ni" (default) or any tabulated material
    grating_thickness_nm: float = 100.0
    use_ccd: bool = True


def _ccd_qe(energy_ev: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(energy_ev, _BI_CCD_ENERGY_EV, _BI_CCD_QE), 0.0, 1.0)


def _grating_efficiency(energy_ev: np.ndarray, material: str, thickness_nm: float) -> np.ndarray:
    """Transmission-grating 1st-order efficiency (simple attenuation proxy)."""
    try:
        absorbed = 1.0 - transmission(material, energy_ev, thickness_um=thickness_nm * 1e-3)
    except ValueError:
        # Unknown material → constant low-efficiency plateau.
        return np.full_like(energy_ev, 0.10)
    # 1st-order efficiency of a Ni transmission grating is ~10% and
    # modulated by substrate absorption; approximate with (1 − T)·0.3 to
    # reward energies the grating actually scatters.
    return 0.30 * absorbed


def apply_soft_xray_response(
    spectrum_keV: xr.DataArray,
    config: SoftXrayConfig | None = None,
) -> xr.DataArray:
    """Apply the soft-X-ray pipeline to a ``spectrum`` DataArray.

    The input is expected to carry a ``photon_energy_keV`` coordinate.
    """
    cfg = config or SoftXrayConfig()
    energy_keV = np.asarray(spectrum_keV.coords["photon_energy_keV"].values, dtype=float)
    energy_ev = energy_keV * 1e3
    T_total = np.ones_like(energy_ev)
    for f in cfg.filters:
        T_total = T_total * transmission(f.material, energy_ev, f.thickness_um)
    T_total = T_total * _grating_efficiency(
        energy_ev, cfg.grating_material, cfg.grating_thickness_nm
    )
    if cfg.use_ccd:
        T_total = T_total * _ccd_qe(energy_ev)
    sig = spectrum_keV.values * T_total
    out = spectrum_keV.copy(data=sig)
    out.attrs["band"] = "xray-soft"
    out.attrs["filter_chain"] = "|".join(
        f"{f.material}-{f.thickness_um:g}um" for f in cfg.filters
    )
    out.attrs["grating"] = f"{cfg.grating_material}-{cfg.grating_thickness_nm:g}nm"
    out.attrs["energy_ev_low"] = float(energy_ev.min())
    out.attrs["energy_ev_high"] = float(energy_ev.max())
    return out
