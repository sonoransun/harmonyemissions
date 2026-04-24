"""Acceleration abstractions — CPU (numba / pyfftw / scipy-workers) and GPU (cupy).

All members are safe to import in environments that have only NumPy; the
optional backends fall back to the stdlib path when their runtime
dependency is missing.
"""

from harmonyemissions.accel.backend import (
    HAS_CUPY,
    HAS_NUMBA,
    HAS_PYFFTW,
    asarray_numpy,
    get_xp,
)
from harmonyemissions.accel.bessel import kv_two_thirds_half
from harmonyemissions.accel.fft import fft2, ifft2
from harmonyemissions.accel.jit import njit

__all__ = [
    "HAS_CUPY",
    "HAS_NUMBA",
    "HAS_PYFFTW",
    "asarray_numpy",
    "fft2",
    "get_xp",
    "ifft2",
    "kv_two_thirds_half",
    "njit",
]
