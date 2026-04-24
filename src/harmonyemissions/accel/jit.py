"""Numba ``njit`` fallback.

Decorating a function with :func:`njit` compiles it via numba when numba
is installed, or leaves it as plain Python otherwise.  This keeps the
surface code identical on any platform while giving a 10–50× speedup
when numba is available.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harmonyemissions.accel.backend import HAS_NUMBA

if HAS_NUMBA:
    import numba  # type: ignore[import-not-found]


def njit(fn: Callable[..., Any] | None = None, /, **kwargs: Any) -> Any:
    """Decorator: ``numba.njit`` when available, no-op otherwise.

    Usage::

        @njit(cache=True)
        def hot(x):
            ...

    Or without arguments::

        @njit
        def hot(x):
            ...
    """
    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        if HAS_NUMBA:
            return numba.njit(**kwargs)(f)
        return f

    if fn is not None and callable(fn):
        return _wrap(fn)
    return _wrap
