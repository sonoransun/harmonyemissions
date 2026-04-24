"""Unified 2-D FFT wrapper.

Dispatch strategy:

- CuPy arrays → ``cupy.fft.fft2`` (stays on GPU).
- NumPy arrays with pyfftw available → plan-cached ``pyfftw.builders.fft2``.
- Otherwise → ``scipy.fft.fft2`` with ``workers=-1`` (threaded, uses all cores).

All three backends are bit-compatible in output, so the rest of the code
can be agnostic to which one ran.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from scipy import fft as _spfft

from harmonyemissions.accel.backend import HAS_CUPY, HAS_PYFFTW, get_xp

if HAS_PYFFTW:
    import pyfftw  # type: ignore[import-not-found]


_PYFFTW_PLAN_CACHE: dict[tuple[int, int, str], Any] = {}


def _pyfftw_plan(shape: tuple[int, int], kind: str) -> Any:
    key = (shape[0], shape[1], kind)
    cached = _PYFFTW_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    buffer = pyfftw.empty_aligned(shape, dtype="complex128")
    builder = pyfftw.builders.fft2 if kind == "fwd" else pyfftw.builders.ifft2
    plan = builder(buffer, threads=-1)
    _PYFFTW_PLAN_CACHE[key] = plan
    return plan


def fft2(u: Any, workers: int | None = -1) -> Any:
    """2-D forward FFT.  ``workers`` is passed to scipy.fft (ignored by cupy/pyfftw)."""
    xp = get_xp(u)
    if HAS_CUPY and xp is not np:
        return xp.fft.fft2(u)
    if HAS_PYFFTW and u.dtype.kind == "c":
        plan = _pyfftw_plan(u.shape, "fwd")
        plan.input_array[...] = u
        return plan().copy()
    return _spfft.fft2(u, workers=workers)


def ifft2(u: Any, workers: int | None = -1) -> Any:
    """2-D inverse FFT, mirror of :func:`fft2`."""
    xp = get_xp(u)
    if HAS_CUPY and xp is not np:
        return xp.fft.ifft2(u)
    if HAS_PYFFTW and u.dtype.kind == "c":
        plan = _pyfftw_plan(u.shape, "inv")
        plan.input_array[...] = u
        return plan().copy()
    return _spfft.ifft2(u, workers=workers)


@cache
def _rfft_freqs(n: int, d: float) -> np.ndarray:
    """Cached 1-D rfft frequency grid."""
    return np.fft.rfftfreq(n, d=d)
