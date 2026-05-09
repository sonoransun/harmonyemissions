"""Per-beam delays and phase optimisation for chf3d.

Implements:

- ``geometric_delays(beam, r_focus)`` — Δt_i = ‖r_focus − r_i‖ / c (in fs)
  so beam wavefronts cross the focus simultaneously regardless of the
  per-driver launch radius.
- ``analytic_phase_lock(beam, A, n, λ)`` — closed-form per-beam phase that
  cancels both the per-beam complex amplitude phase and the propagation
  phase to the focus, producing constructive interference at r_focus.
- ``optimise_phases(method)`` — dispatcher; iterative methods
  (``scipy_lbfgs`` / ``gerchberg_saxton``) are deferred to Phase D and
  raise ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np

from harmonyemissions.chf.geometry import BeamArray

C_M_PER_S = 2.99792458e8


def geometric_delays(
    beam: BeamArray, r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """Return per-beam delays Δt_i [fs] = ‖r_focus − r_i‖ / c.

    Arranges per-beam timing so the launched pulses arrive at ``r_focus``
    simultaneously. Useful default when the user has not specified
    ``relative_delay_fs`` explicitly.
    """
    rf = np.asarray(r_focus, dtype=float).reshape(1, 3)
    delta_r = beam.positions - rf  # (N, 3)
    distance_m = np.linalg.norm(delta_r, axis=1)
    return distance_m / C_M_PER_S * 1e15  # → fs


def analytic_phase_lock(
    beam: BeamArray,
    per_beam_amplitudes: np.ndarray,
    harmonic_n: int,
    wavelength_m: float,
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Closed-form phase set that maximises focal intensity at ``r_focus``.

    Implements ``φ_i* = −arg(A_i) − k_n · n̂_i · (r_focus − r_i)`` so
    that the per-beam contribution ``A_i exp(i φ_i) exp(i k_n n̂_i·(r_focus−r_i))``
    becomes ``|A_i|`` — purely real and positive — at the focus.
    """
    A = np.asarray(per_beam_amplitudes, dtype=complex).reshape(beam.n_beams)
    rf = np.asarray(r_focus, dtype=float).reshape(1, 3)
    lam_n = wavelength_m / float(harmonic_n)
    k_n = 2.0 * np.pi / lam_n
    delta = rf - beam.positions  # (N, 3) — vector from r_i to r_focus
    propagation_phase = k_n * np.einsum("ij,ij->i", beam.directions, delta)
    return -np.angle(A) - propagation_phase


def optimise_phases(
    beam: BeamArray,
    per_beam_amplitudes: np.ndarray,
    harmonic_n: int,
    wavelength_m: float,
    method: str = "analytic",
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Dispatch on phase-optimisation method.

    Phase D will add ``scipy_lbfgs`` (gradient ascent on focal intensity
    over a band of harmonics) and ``gerchberg_saxton`` (iterative
    far-field/near-field projection). For now they raise to keep callers
    honest about which path is live.
    """
    if method == "analytic":
        return analytic_phase_lock(
            beam, per_beam_amplitudes, harmonic_n, wavelength_m, r_focus
        )
    if method in ("scipy_lbfgs", "gerchberg_saxton"):
        raise NotImplementedError(
            f"phase_optimiser={method!r} lands in Phase D; use 'analytic' for now"
        )
    raise ValueError(f"unknown phase optimiser {method!r}")
