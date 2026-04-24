"""Property-based tests via Hypothesis.

Strategies explore realistic laser / target parameter ranges and assert
model invariants (finite spectra, plateau monotonicity, Parseval on
Fraunhofer transforms, logistic midpoint of the spikes filter). Kept
small via ``max_examples=25`` so the suite stays well under the fast-CI
time budget.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.accel.backend import HAS_NUMBA
from harmonyemissions.beam import SpatialGrid, fraunhofer, gaussian_spot, intensity
from harmonyemissions.emission.spikes import (
    CutoffMode,
    relativistic_spikes_filter,
    spikes_cutoff_harmonic,
)

_fast = settings(max_examples=25, deadline=None)


@_fast
@given(a0=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_bgp_spectrum_is_finite_and_positive(a0: float):
    r = simulate(Laser(a0=a0), Target.overdense(100.0), model="bgp")
    s = r.spectrum.values
    assert np.all(np.isfinite(s))
    assert np.all(s > 0)


@_fast
@given(
    a0=st.floats(min_value=3.0, max_value=50.0),
    ncc=st.floats(min_value=50.0, max_value=500.0),
    L=st.floats(min_value=0.01, max_value=0.3),
)
def test_bgp_plateau_monotone(a0: float, ncc: float, L: float):
    """In the BGP plateau, I(n) must decrease with n."""
    r = simulate(Laser(a0=a0), Target.overdense(ncc, L), model="bgp")
    n = r.spectrum.coords["harmonic"].values
    s = r.spectrum.values
    nc = r.diagnostics["n_cutoff"]
    upper = min(0.3 * nc, 40.0)
    mask = (n >= 3) & (n <= upper)
    if mask.sum() < 2:
        return  # plateau too narrow for this sample
    assert s[mask][0] > s[mask][-1]


@_fast
@given(a0=st.floats(min_value=0.1, max_value=50.0))
def test_spikes_logistic_midpoint(a0: float):
    """S(n_c, a₀) is exactly 0.5 for the logistic cutoff, for any a₀ > 0."""
    n_c = float(spikes_cutoff_harmonic(a0))
    s = float(relativistic_spikes_filter(n_c, a0, mode=CutoffMode.LOGISTIC))
    assert abs(s - 0.5) < 1e-10


@_fast
@given(
    a0=st.floats(min_value=0.5, max_value=50.0),
    scale=st.floats(min_value=1.1, max_value=3.0),
)
def test_cutoff_prefactor_universal(a0: float, scale: float):
    """n_c(scale · a₀) = scale³ · n_c(a₀) regardless of a₀."""
    nc1 = spikes_cutoff_harmonic(a0)
    nc2 = spikes_cutoff_harmonic(a0 * scale)
    assert nc2 == pytest_approx(nc1 * scale ** 3)


def pytest_approx(value, rel: float = 1e-12):
    # Local wrapper so we can avoid importing pytest at module scope.
    import pytest
    return pytest.approx(value, rel=rel)


@_fast
@given(
    n_pix=st.integers(min_value=32, max_value=96),
    fwhm_um=st.floats(min_value=0.5, max_value=5.0),
)
def test_fraunhofer_parseval_energy_conservation(n_pix: int, fwhm_um: float):
    """Σ |u₀|² Δx² == Σ |U_far|² Δx_far² up to numerical tolerance."""
    grid = SpatialGrid(n=n_pix, dx=0.1e-6)
    u0 = gaussian_spot(grid, fwhm_um * 1e-6)
    u_far, dx_far = fraunhofer(u0, grid.dx, 8e-7, 1.0)
    power0 = float(np.sum(intensity(u0)) * grid.dx ** 2)
    power_far = float(np.sum(intensity(u_far)) * dx_far ** 2)
    assert abs(power_far - power0) / power0 < 1e-5


@_fast
@given(a0=st.floats(min_value=3.0, max_value=30.0))
def test_spikes_monotone_in_harmonic(a0: float):
    """For fixed a₀ (above ~n_cutoff > 1), S(n, a₀) decreases in n."""
    nc = float(spikes_cutoff_harmonic(a0))
    n = np.linspace(1.0, 2.0 * nc, 32)
    s = relativistic_spikes_filter(n, a0, mode=CutoffMode.EXPONENTIAL)
    # Below-cutoff harmonics should dominate the far tail.
    head = s[n < 0.3 * nc]
    if head.size == 0:
        return
    assert head.min() > s[-1]


def test_numba_flag_matches_environment():
    """The JIT decorator must agree with the runtime numba availability."""
    from harmonyemissions.accel.jit import njit as _njit

    @_njit(cache=True)
    def f(x):
        return x * 2 + 1

    assert f(3.0) == 7.0
    # If HAS_NUMBA, the decorated function should have a __wrapped__ attribute
    # or a numba Dispatcher type.
    if HAS_NUMBA:
        import numba
        assert isinstance(f, numba.core.dispatcher.Dispatcher)
