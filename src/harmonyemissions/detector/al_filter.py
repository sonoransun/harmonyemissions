"""Aluminium filter transmission in the XUV.

Real transmission is measured from Henke tables (the paper's ref. 47).
Here we use a simplified analytical approximation that captures:

- Near-zero transmission below the aluminium L-edge (17 nm).
- A pass-band between ~17 nm and ~70 nm with slowly rising transmission
  as absorption drops.
- Exponential attenuation length scaling linearly with Al thickness.

For quantitative work against real measurements, override this module
with a table-based lookup.
"""

from __future__ import annotations

import numpy as np

# Tabulated attenuation lengths (in μm) at selected XUV wavelengths.
# Source: Henke et al. 1993 (paper ref. 47), fitted roughly.
_REFERENCE_WAVELENGTHS_NM = np.array([17.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0])
_ATTENUATION_LENGTHS_UM = np.array([0.02, 0.10, 0.25, 0.40, 0.55, 0.62, 0.50, 0.30])


def _attenuation_length_um(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Interpolated attenuation length (1/e in amplitude) in μm."""
    w = np.asarray(wavelength_nm, dtype=float)
    return np.interp(w, _REFERENCE_WAVELENGTHS_NM, _ATTENUATION_LENGTHS_UM)


def al_filter_transmission(
    wavelength_nm: float | np.ndarray,
    thickness_um: float,
) -> np.ndarray:
    """Return the Al filter amplitude transmission T(λ) for a given thickness.

    Below the L-edge (~17 nm) transmission is ~0; above ~80 nm the Al
    becomes opaque again at the plasma frequency. We clamp those regimes.
    """
    w = np.asarray(wavelength_nm, dtype=float)
    T = np.zeros_like(w)
    in_band = (w >= _REFERENCE_WAVELENGTHS_NM[0]) & (w <= 80.0)
    if np.any(in_band):
        lengths = _attenuation_length_um(w[in_band])
        T[in_band] = np.exp(-thickness_um / np.maximum(lengths, 1e-6))
    return T


def harmonic_to_wavelength_nm(n: float | np.ndarray, wavelength_um_driver: float) -> np.ndarray:
    """Convert harmonic order n to wavelength in nm."""
    return 1000.0 * wavelength_um_driver / np.asarray(n, dtype=float)


def oxide_correction(wavelength_nm: float | np.ndarray, oxide_thickness_nm: float = 9.0) -> np.ndarray:
    """Al₂O₃ transmission correction. Thickness ~9 nm (paper's measurement)."""
    # Very approximate: similar exp-attenuation but with shorter lengths.
    w = np.asarray(wavelength_nm, dtype=float)
    atten_nm = np.interp(w, _REFERENCE_WAVELENGTHS_NM, _ATTENUATION_LENGTHS_UM * 1000.0)
    return np.exp(-oxide_thickness_nm / np.maximum(atten_nm, 1e-3))


def hydrocarbon_correction(wavelength_nm: float | np.ndarray, thickness_nm: float = 5.0) -> np.ndarray:
    """Crude CH-contaminant correction — exponential attenuation, gentler."""
    w = np.asarray(wavelength_nm, dtype=float)
    # CH has longer attenuation lengths → weaker correction.
    atten_nm = np.interp(w, _REFERENCE_WAVELENGTHS_NM, _ATTENUATION_LENGTHS_UM * 2000.0)
    return np.exp(-thickness_nm / np.maximum(atten_nm, 1e-3))
