"""Flat-field grating higher-order-diffraction contamination.

The Timmis 2026 paper uses a 300 line/mm flat-field grating (SHIMADZU
L0300-20-80). Its 2nd- and 3rd-order efficiencies rise sharply with
harmonic order and must be deconvolved from the raw spectrum; the paper
fits eq. 6,

    R_{s_{2,3}/s_1}(n) = ν / (1 + exp(−k (n − n₀))) + b,

with logistic parameters

    2nd order: ν=0.92, n₀=40, k=0.37, b=0.12
    3rd order: ν=0.79, n₀=44, k=0.44, b=0.13.

This module exposes those fits and applies them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GratingFit:
    nu: float
    n0: float
    k: float
    b: float

    def __call__(self, n: float | np.ndarray) -> np.ndarray:
        n_arr = np.asarray(n, dtype=float)
        return self.nu / (1.0 + np.exp(-self.k * (n_arr - self.n0))) + self.b


# Paper's eq. 6 coefficients.
SECOND_ORDER = GratingFit(nu=0.92, n0=40.0, k=0.37, b=0.12)
THIRD_ORDER = GratingFit(nu=0.79, n0=44.0, k=0.44, b=0.13)


def grating_order_ratio(n: float | np.ndarray, order: int = 2) -> np.ndarray:
    """Return R_{s_m/s_1}(n) — ratio of mth-order to first-order efficiency."""
    if order == 2:
        return SECOND_ORDER(n)
    if order == 3:
        return THIRD_ORDER(n)
    raise ValueError(f"only 2nd and 3rd orders parametrised; got order={order}")


def deconvolve_second_order(
    n: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Remove the 2nd-order overlap from a 1-st-order signal.

    Measured intensity at position of harmonic n is approximately
        S_measured(n) = S_1(n) + R_{s_2/s_1}(2n) · S_1(2n).
    Iterating from the highest n downward lets us recover S_1(n) knowing
    S_1(2n) already.
    """
    n = np.asarray(n, dtype=float)
    s = np.asarray(signal, dtype=float).copy()
    order_idx = np.argsort(n)[::-1]  # descending n
    for idx in order_idx:
        ni = n[idx]
        n2 = 2.0 * ni
        j = int(np.argmin(np.abs(n - n2)))
        if j != idx and np.abs(n[j] - n2) / n2 < 0.02:
            # s[j] is now the already-deconvolved S_1(2n); subtract its
            # 2nd-order contribution from s[idx] to recover S_1(n).
            s[idx] = s[idx] - grating_order_ratio(n2, 2) * s[j]
    return np.maximum(s, 0.0)
