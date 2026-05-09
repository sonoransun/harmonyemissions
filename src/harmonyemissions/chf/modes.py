"""Structured-light beam modes for the chf3d Phase C kernel.

Each helper returns a multiplicative profile (or, for vector modes, a
``(2, N, N)`` Jones field) that can be applied to a driver near-field
``u0`` before far-field propagation. See ``docs/chf3d.md`` § "Structured
modes" for the math.

Modes:

- ``lg``        — Laguerre–Gauss LG_pℓ; carries OAM ℓ.
- ``bessel``    — Bessel-Gauss approximation J_m(k_r r) · exp(-(r/w)²).
- ``radial``    — radially polarized cylindrical-vector beam.
- ``azimuthal`` — azimuthally polarized cylindrical-vector beam.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from harmonyemissions.beam import SpatialGrid


def _polar_coords(grid: SpatialGrid) -> tuple[np.ndarray, np.ndarray]:
    x, y = grid.coords()
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return r, phi


def lg_profile(grid: SpatialGrid, w0_m: float, oam_l: int, p: int) -> np.ndarray:
    """Laguerre–Gauss LG_pℓ amplitude on ``grid`` with waist ``w0_m``.

    Carries OAM ℓ (``oam_l``) via the ``exp(i ℓ φ)`` factor; ``p`` is
    the radial quantum number (0 = simple vortex). Returns a complex
    (N, N) array.
    """
    from scipy.special import eval_genlaguerre

    r, phi = _polar_coords(grid)
    rho = (r / w0_m) ** 2
    radial = (np.sqrt(2.0) * r / w0_m) ** abs(oam_l) * eval_genlaguerre(
        p, abs(oam_l), 2.0 * rho
    )
    return (radial * np.exp(-rho) * np.exp(1j * oam_l * phi)).astype(np.complex128)


def bessel_profile(
    grid: SpatialGrid,
    kr_per_k: float,
    order: int = 0,
    wavelength_m: float = 0.8e-6,
    waist_m: float | None = None,
) -> np.ndarray:
    """Bessel–Gauss approximation J_m(k_r r) · exp(-(r/w)²)."""
    from scipy.special import jv

    r, phi = _polar_coords(grid)
    k = 2.0 * np.pi / wavelength_m
    kr = kr_per_k * k
    envelope = (
        np.exp(-((r / waist_m) ** 2)) if waist_m is not None else np.ones_like(r)
    )
    return (jv(order, kr * r) * envelope * np.exp(1j * order * phi)).astype(np.complex128)


def radial_vector_profile(grid: SpatialGrid, fwhm_m: float) -> np.ndarray:
    """Radially polarized cylindrical-vector beam.

    Returns a complex ``(2, N, N)`` array — the in-plane Jones components
    (x, y). The amplitude envelope is a Laguerre–Gauss-1 (donut) so that
    it has a centre null compatible with vector polarization.
    """
    r, phi = _polar_coords(grid)
    sigma = fwhm_m / (2.0 * np.sqrt(np.log(2.0)))
    envelope = (r / sigma) * np.exp(-0.5 * (r / sigma) ** 2)
    fx = envelope * np.cos(phi)
    fy = envelope * np.sin(phi)
    return np.stack([fx, fy], axis=0).astype(np.complex128)


def azimuthal_vector_profile(grid: SpatialGrid, fwhm_m: float) -> np.ndarray:
    """Azimuthally polarized cylindrical-vector beam (same envelope as radial)."""
    r, phi = _polar_coords(grid)
    sigma = fwhm_m / (2.0 * np.sqrt(np.log(2.0)))
    envelope = (r / sigma) * np.exp(-0.5 * (r / sigma) ** 2)
    fx = -envelope * np.sin(phi)
    fy = envelope * np.cos(phi)
    return np.stack([fx, fy], axis=0).astype(np.complex128)


def apply_structured_mode(
    u0: np.ndarray,
    grid: SpatialGrid,
    mode: str | None,
    params: dict[str, Any] | None,
    *,
    fwhm_m: float,
    wavelength_m: float,
) -> np.ndarray:
    """Apply a scalar structured-light profile to ``u0`` (multiplicative).

    Vector modes (radial / azimuthal) are handled at the polarization
    layer in :mod:`chf.geometry`, not here. This helper only returns a
    scalar amplitude profile for the LG / Bessel cases.
    """
    if mode is None:
        return u0
    params = params or {}
    if mode == "lg":
        oam_l = int(params["l"])
        p = int(params["p"])
        w0 = float(params.get("w0_m", fwhm_m / (2.0 * np.sqrt(np.log(2.0)))))
        return u0 * lg_profile(grid, w0, oam_l, p)
    if mode == "bessel":
        return u0 * bessel_profile(
            grid,
            kr_per_k=float(params["kr_per_k"]),
            order=int(params.get("order", 0)),
            wavelength_m=wavelength_m,
            waist_m=params.get("waist_m"),
        )
    if mode in ("radial", "azimuthal"):
        # Vector modes don't reshape u0 directly — the per-beam Jones
        # vector ε_i in the geometry layer handles the polarization
        # rotation. Returning u0 unchanged keeps the scalar amplitude.
        return u0
    raise ValueError(f"unknown structured mode {mode!r}")
