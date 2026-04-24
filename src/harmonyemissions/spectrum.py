"""Spectrum and pulse-synthesis helpers shared by all time-domain models.

Conventions:
- Time grids are in units of the laser period T₀; so frequencies returned are
  harmonic orders n = ω/ω₀.
- Fourier transforms are normalized so that ``|FFT(E)|²`` has units of
  spectral intensity per unit harmonic order.
- "Attosecond pulse" = time-domain field after bandpass-filtering to the
  harmonic window (n_low, n_high).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def field_to_spectrum(t_over_T0: np.ndarray, field: np.ndarray) -> xr.DataArray:
    """Compute the harmonic spectrum |FFT(E)|² from a time-domain field.

    Returns a DataArray indexed by positive harmonic order ``n``.
    """
    t = np.asarray(t_over_T0, dtype=float)
    e = np.asarray(field, dtype=float)
    if t.size != e.size:
        raise ValueError("time and field arrays must be the same length")
    dt = float(t[1] - t[0])
    n_samples = t.size
    freqs = np.fft.rfftfreq(n_samples, d=dt)  # in units of 1/T₀ = n
    amps = np.fft.rfft(e) / n_samples
    spec = np.abs(amps) ** 2
    # Drop DC for plotting cleanliness; keep from n=1 upwards.
    mask = freqs >= 0.5
    return xr.DataArray(
        spec[mask],
        coords={"harmonic": freqs[mask]},
        dims=["harmonic"],
        name="spectrum",
    )


def bandpass_field(
    t_over_T0: np.ndarray,
    field: np.ndarray,
    n_low: float,
    n_high: float,
    rolloff: float = 0.1,
) -> np.ndarray:
    """Return the field bandpass-filtered to harmonics [n_low, n_high].

    Uses a smooth cosine rolloff (width ``rolloff`` in harmonic-order units)
    on both edges of the mask to suppress Gibbs ringing when synthesizing
    attosecond pulses.
    """
    t = np.asarray(t_over_T0, dtype=float)
    e = np.asarray(field, dtype=float)
    dt = float(t[1] - t[0])
    n_samples = t.size
    spec = np.fft.rfft(e)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    mask = _cosine_mask(freqs, n_low, n_high, rolloff)
    return np.fft.irfft(spec * mask, n=n_samples)


def _cosine_mask(freqs: np.ndarray, n_low: float, n_high: float, rolloff: float) -> np.ndarray:
    mask = np.zeros_like(freqs)
    lo_core = n_low + rolloff
    hi_core = n_high - rolloff
    in_core = (freqs >= lo_core) & (freqs <= hi_core)
    mask[in_core] = 1.0
    lo_ramp = (freqs >= n_low) & (freqs < lo_core)
    mask[lo_ramp] = 0.5 * (1.0 - np.cos(np.pi * (freqs[lo_ramp] - n_low) / rolloff))
    hi_ramp = (freqs > hi_core) & (freqs <= n_high)
    mask[hi_ramp] = 0.5 * (1.0 + np.cos(np.pi * (freqs[hi_ramp] - hi_core) / rolloff))
    return mask


def attosecond_pulse(
    t_over_T0: np.ndarray,
    field: np.ndarray,
    harmonic_window: tuple[float, float] | None,
) -> xr.DataArray | None:
    """Synthesize an attosecond pulse by bandpass-filtering to a harmonic window."""
    if harmonic_window is None:
        return None
    n_low, n_high = harmonic_window
    filtered = bandpass_field(t_over_T0, field, n_low, n_high)
    return xr.DataArray(
        filtered,
        coords={"t_over_T0": np.asarray(t_over_T0, dtype=float)},
        dims=["t_over_T0"],
        name="attosecond_pulse",
    )


def time_field(t_over_T0: np.ndarray, field: np.ndarray) -> xr.DataArray:
    """Wrap a raw time-domain field into an xarray.DataArray."""
    return xr.DataArray(
        np.asarray(field, dtype=float),
        coords={"t_over_T0": np.asarray(t_over_T0, dtype=float)},
        dims=["t_over_T0"],
        name="time_field",
    )
