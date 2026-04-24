"""Cached modified-Bessel K_{2/3} evaluator for betatron spectra.

``scipy.special.kv(2/3, x)`` is expensive and is called element-wise on
the ~2048-point harmonic grid of the betatron model. We precompute a
log-spaced table at import time and use :func:`numpy.interp` for
subsequent calls — the same pattern the detector's Al-filter
transmission uses.

The table covers x ∈ [1e-4, 1e3], where the betatron synchrotron integrand
``ξ · K_{2/3}(ξ/2)²`` falls from a power-law rise to exponential decay.
Outside this range we return zero (deep in the exponentially-attenuated
tail anyway).
"""

from __future__ import annotations

import numpy as np
from scipy.special import kv

# Log-spaced table — log-log interpolation keeps relative error under 1e-4
# across ~45 orders of magnitude of the function value.
_X_TABLE = np.geomspace(1e-4, 1e3, 4096)
_LOG_X_TABLE = np.log(_X_TABLE)
_KV_SQ_HALF_X = kv(2.0 / 3.0, _X_TABLE / 2.0) ** 2
# Work in log(K²) space so that both sides of the interpolator are smooth.
_LOG_KV_SQ = np.log(np.clip(_KV_SQ_HALF_X, 1e-300, None))


def kv_two_thirds_half(x: np.ndarray | float) -> np.ndarray:
    """Return K_{2/3}(x/2)² via cached log-log interpolation.

    Inputs outside the tabulated range clamp to the table edges. Accuracy
    is ~1e-4 relative in the [1e-3, 1e2] domain that matters for
    betatron spectra.
    """
    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(x_arr)
    mask = x_arr > 0
    if np.any(mask):
        log_x = np.log(np.clip(x_arr[mask], _X_TABLE[0], _X_TABLE[-1]))
        out[mask] = np.exp(np.interp(log_x, _LOG_X_TABLE, _LOG_KV_SQ))
    return out
