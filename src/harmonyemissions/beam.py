"""2D spatial laser-beam profiles and Fraunhofer propagation.

Laser focal spots in high-power systems are rarely pure Gaussians. The
Timmis 2026 paper emphasises that a **super-Gaussian / top-hat** focal
profile (common when the amplifier chain is saturated) is essential to
the Coherent Harmonic Focus: its radial "wings" at the 5–10 % intensity
level produce the high-spatial-frequency components that, after
ponderomotive denting, imprint the curved wavefront on the reflected
harmonic beam.

This module provides:

1. Factory helpers for canonical spatial profiles:
   Gaussian, super-Gaussian (order p), flat-top / top-hat, and jinc (the
   ideal diffraction-limited spot from a circular aperture).
2. A 2-D Fraunhofer propagator ``fraunhofer(u0, dx, wavelength, z)``
   that returns the far-field amplitude on a matching grid.
3. An inverse propagator ``inverse_fraunhofer`` so users can round-trip
   between near- and far-field.

All grids are square, complex-valued ``np.ndarray`` of shape ``(N, N)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

ProfileName = Literal["gaussian", "super_gaussian", "top_hat", "jinc"]


@dataclass(frozen=True)
class SpatialGrid:
    """Square x/y grid in metres, centred on zero."""

    n: int
    dx: float  # metre

    @property
    def extent(self) -> float:
        return self.n * self.dx

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        axis = (np.arange(self.n) - self.n // 2) * self.dx
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return x, y


def gaussian_spot(grid: SpatialGrid, fwhm_m: float) -> np.ndarray:
    """Gaussian spot: E(r) = exp(−r²/(2σ²)), FWHM of intensity = ``fwhm_m``."""
    x, y = grid.coords()
    r2 = x * x + y * y
    sigma = fwhm_m / (2.0 * np.sqrt(np.log(2.0)))
    return np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.complex128)


def super_gaussian_spot(grid: SpatialGrid, fwhm_m: float, order: int = 8) -> np.ndarray:
    """Super-Gaussian: E(r) = exp(−(r/w)^p), FWHM of intensity = ``fwhm_m``.

    Order ``order=2`` is a plain Gaussian; higher orders approach a top-hat.
    High-power amplifier chains typically produce p ≈ 6–10.
    """
    if order < 2:
        raise ValueError("super-Gaussian order must be ≥ 2")
    x, y = grid.coords()
    r = np.sqrt(x * x + y * y)
    # Solve 0.5 = exp(−2 (fwhm/2/w)^p) for w so intensity FWHM = fwhm_m.
    half = fwhm_m / 2.0
    w = half * (np.log(2.0) / 2.0) ** (-1.0 / order)
    return np.exp(-((r / w) ** order)).astype(np.complex128)


def top_hat_spot(grid: SpatialGrid, diameter_m: float) -> np.ndarray:
    """Ideal top-hat (circular aperture) amplitude of given diameter."""
    x, y = grid.coords()
    r = np.sqrt(x * x + y * y)
    return (r <= 0.5 * diameter_m).astype(np.complex128)


def jinc_spot(grid: SpatialGrid, fwhm_m: float) -> np.ndarray:
    """Jinc(r) = 2 J_1(κ r)/(κ r) — ideal diffraction-limited spot."""
    from scipy.special import j1

    x, y = grid.coords()
    r = np.sqrt(x * x + y * y)
    # Airy FWHM ≈ 1.028 · κ/λ_f; pick κ so FWHM(intensity) = fwhm_m.
    # Equivalently: κ · r_fwhm/2 = 1.6163 (first root of the jinc²).
    kappa = 2.0 * 1.6163 / fwhm_m
    with np.errstate(divide="ignore", invalid="ignore"):
        u = kappa * r
        out = np.where(u == 0, 1.0, 2.0 * j1(u) / u)
    return out.astype(np.complex128)


_PROFILES: dict[str, Callable[[SpatialGrid, float, int], np.ndarray]] = {
    "gaussian": lambda g, f, _: gaussian_spot(g, f),
    "super_gaussian": lambda g, f, p: super_gaussian_spot(g, f, p or 8),
    "top_hat": lambda g, f, _: top_hat_spot(g, f),
    "jinc": lambda g, f, _: jinc_spot(g, f),
}


def build_profile(
    name: ProfileName,
    grid: SpatialGrid,
    fwhm_m: float,
    super_gaussian_order: int = 8,
) -> np.ndarray:
    """Dispatch to the named profile factory."""
    try:
        return _PROFILES[name](grid, fwhm_m, super_gaussian_order)
    except KeyError as exc:
        raise ValueError(
            f"Unknown spatial profile {name!r}; options: {sorted(_PROFILES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Fraunhofer propagation (2-D).
# ---------------------------------------------------------------------------


def fraunhofer(u0: np.ndarray, dx: float, wavelength: float, z: float) -> tuple[np.ndarray, float]:
    """Propagate a near-field amplitude u0 to the far field at distance z.

    Returns ``(u_far, dx_far)`` where ``dx_far = λ z / (N dx)`` is the far-field
    pixel size (inverse relationship, standard Fraunhofer FFT result).

    The amplitude is normalized so that ``Σ |u_far|² · dx_far²`` ≈ ``Σ |u0|² · dx²``
    (energy conserved up to numerical FFT accuracy). FFT is dispatched
    through :mod:`harmonyemissions.accel.fft` — cupy on GPU arrays, pyfftw
    with cached plans, or threaded scipy.fft.
    """
    from harmonyemissions.accel.fft import fft2 as _fft2

    if u0.ndim != 2 or u0.shape[0] != u0.shape[1]:
        raise ValueError("u0 must be a square 2-D array")
    n = u0.shape[0]
    spec = np.fft.fftshift(_fft2(np.fft.ifftshift(u0))) * (dx * dx)
    dx_far = wavelength * z / (n * dx)
    u_far = spec / (dx_far * dx_far) ** 0.5
    power0 = np.sum(np.abs(u0) ** 2) * dx * dx
    power_far = np.sum(np.abs(u_far) ** 2) * dx_far * dx_far
    if power_far > 0:
        u_far = u_far * np.sqrt(power0 / power_far)
    return u_far, dx_far


def inverse_fraunhofer(
    u_far: np.ndarray, dx_far: float, wavelength: float, z: float
) -> tuple[np.ndarray, float]:
    """Inverse of :func:`fraunhofer`. Returns ``(u0, dx)``."""
    from harmonyemissions.accel.fft import ifft2 as _ifft2

    if u_far.ndim != 2 or u_far.shape[0] != u_far.shape[1]:
        raise ValueError("u_far must be a square 2-D array")
    n = u_far.shape[0]
    spec = np.fft.fftshift(_ifft2(np.fft.ifftshift(u_far))) * (dx_far * dx_far) * (n * n)
    dx = wavelength * z / (n * dx_far)
    u0 = spec / (dx * dx) ** 0.5
    power_far = np.sum(np.abs(u_far) ** 2) * dx_far * dx_far
    power0 = np.sum(np.abs(u0) ** 2) * dx * dx
    if power0 > 0:
        u0 = u0 * np.sqrt(power_far / power0)
    return u0, dx


def intensity(u: np.ndarray) -> np.ndarray:
    """|u|² of a complex amplitude."""
    return (u * np.conj(u)).real


def peak_intensity(u: np.ndarray) -> float:
    """Max of |u|²."""
    return float(np.max(intensity(u)))


def fwhm_spot_size(u: np.ndarray, dx: float) -> float:
    """Approximate FWHM diameter of the intensity distribution."""
    i = intensity(u)
    peak = i.max()
    if peak <= 0:
        return float("nan")
    mask = i >= 0.5 * peak
    if not np.any(mask):
        return 0.0
    ys, xs = np.nonzero(mask)
    return dx * max(xs.max() - xs.min(), ys.max() - ys.min())
