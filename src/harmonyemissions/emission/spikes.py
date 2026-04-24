"""Relativistic-spikes spatial filter S(n, a₀).

Reference
---------
Gordienko, Pukhov, Shorokhov & Baeva, *Phys. Rev. Lett.* **94**, 103903 (2005)
("Coherent focusing of high harmonics: a new way towards the extreme
intensities") — the paper's ref. 10. Baeva, Gordienko & Pukhov, *Phys.
Rev. E* **74**, 046404 (2006) for the companion n^(−8/3) scaling.

What it does
------------
``S(n, a₀)`` is the spatial spectral filter applied to the driving laser
beam profile in the Timmis 2026 pipeline (eqs. 8, 12):

    U_0(x', y', n) ~ √S(n, a_0(x', y')) · U_0(x', y')

It multiplies the far-field driver amplitude per harmonic. Its two
essential features are:

1. A universal n^(−8/3) energy scaling in the plateau.
2. A sharp roll-off at a cutoff harmonic n_c ≈ 4 √(2 α) · a₀³.

We encode the roll-off with a logistic (so that the filter is smooth and
differentiable for optimisation), with width set by ``sharpness``. Taking
``sharpness → ∞`` reproduces the paper's sharp-step caricature.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

# Fine-structure constant — sets the BGP cutoff prefactor.
ALPHA_FS = 1.0 / 137.036
# BGP cutoff prefactor: n_c = 4 √(2 α) · a₀³.
CUTOFF_PREFACTOR = 4.0 * math.sqrt(2.0 * ALPHA_FS)


class CutoffMode(str, Enum):
    """How the cutoff drops off beyond n_c."""

    LOGISTIC = "logistic"     # smooth roll-off (good for numerics)
    EXPONENTIAL = "exponential"  # exp(−(n/n_c)^{2/3}), BGP-style
    SHARP = "sharp"           # Heaviside step — paper's caricature


def spikes_cutoff_harmonic(a0: float | np.ndarray) -> float | np.ndarray:
    """Cutoff harmonic n_c = 4 √(2 α) · a₀³ (BGP)."""
    return CUTOFF_PREFACTOR * np.asarray(a0) ** 3


def universal_envelope(n: np.ndarray, a0: float) -> np.ndarray:
    """Plateau envelope |S|² with n^(−8/3) and an exponential cutoff."""
    n_c = float(CUTOFF_PREFACTOR * a0 ** 3)
    return (np.asarray(n) ** (-8.0 / 3.0)) * np.exp(
        -(np.asarray(n) / max(n_c, 1e-9)) ** (2.0 / 3.0)
    )


def relativistic_spikes_filter(
    n: int | np.ndarray,
    a0: float | np.ndarray,
    mode: CutoffMode = CutoffMode.LOGISTIC,
    sharpness: float = 6.0,
) -> np.ndarray:
    """Return S(n, a₀) — harmonic-dependent spatial filter.

    ``n`` and ``a0`` broadcast normally. For a 2-D spatial map of a₀, pass
    a 2-D array and ``n`` as a scalar; the result has the same shape as
    ``a0``.

    ``mode`` picks the roll-off shape:

    * ``LOGISTIC`` — smooth, midpoint at n = n_c; slope set by ``sharpness``.
      Best choice for differentiable workflows.
    * ``EXPONENTIAL`` — exp(−(n/n_c)^{2/3}): the BGP form.
    * ``SHARP`` — Heaviside(n_c − n) · (n/n_c)^(−8/3): the paper's caricature.
    """
    n_arr = np.asarray(n, dtype=float)
    a0_arr = np.asarray(a0, dtype=float)
    n_c = CUTOFF_PREFACTOR * np.maximum(a0_arr, 1e-12) ** 3

    # Plateau value: S = (n/n_c)^{−8/3} gives the BGP |S(n)|² ∝ n^{−8/3} when
    # evaluated below cutoff. We cast to ratio so the plateau is O(1) near
    # n = n_c.
    ratio = n_arr / np.maximum(n_c, 1e-9)
    plateau = ratio ** (-8.0 / 3.0)

    if mode is CutoffMode.LOGISTIC:
        # Clip the exponent so very-far-above-cutoff points saturate to 0
        # without blowing up (and very-far-below saturate to 1).
        arg = np.clip(sharpness * (ratio - 1.0), -50.0, 50.0)
        roll = 1.0 / (1.0 + np.exp(arg))
    elif mode is CutoffMode.EXPONENTIAL:
        roll = np.exp(-ratio ** (2.0 / 3.0))
    elif mode is CutoffMode.SHARP:
        roll = (ratio <= 1.0).astype(float)
    else:  # pragma: no cover
        raise ValueError(f"unknown cutoff mode {mode!r}")

    return plateau * roll
