"""Array-namespace dispatcher and optional-dependency feature flags.

The dispatcher lets a single function body run on both NumPy and CuPy
arrays: call ``xp = get_xp(x)`` and then use ``xp.zeros_like(x)``,
``xp.exp(...)``, etc.  We ship the simplest possible shim rather than
pulling in ``array_api_compat`` so the hot paths stay explicit.

Feature flags are plain booleans so callers can short-circuit without
catching ImportError every time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional-backend detection
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised via HARMONY_TEST_CUPY
    import cupy as _cp  # type: ignore[import-not-found]
    HAS_CUPY = True
except Exception:  # noqa: BLE001
    _cp = None
    HAS_CUPY = False

try:
    import numba as _numba  # noqa: F401
    HAS_NUMBA = True
except Exception:  # noqa: BLE001
    HAS_NUMBA = False

try:
    import pyfftw  # noqa: F401
    HAS_PYFFTW = True
except Exception:  # noqa: BLE001
    HAS_PYFFTW = False


def get_xp(x: Any) -> Any:
    """Return the array-namespace module that owns ``x`` (numpy or cupy)."""
    if HAS_CUPY and _cp is not None and isinstance(x, _cp.ndarray):  # type: ignore[union-attr]
        return _cp
    return np


def asarray_numpy(x: Any) -> np.ndarray:
    """Convert an array (NumPy or CuPy) to a NumPy array on the host."""
    if HAS_CUPY and _cp is not None and isinstance(x, _cp.ndarray):  # type: ignore[union-attr]
        return _cp.asnumpy(x)
    return np.asarray(x)
