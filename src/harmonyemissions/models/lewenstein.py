"""Gas-phase HHG via the Lewenstein strong-field approximation (simplified).

Reference
---------
Lewenstein, Balcou, Ivanov, L'Huillier, Corkum, *Phys. Rev. A* 49, 2117 (1994).
Corkum three-step model: ionization → propagation → recombination.

Implementation
--------------
Rather than evaluating the full Lewenstein integral (which requires the
saddle-point method and complex trajectories), this module simulates a
representative single-atom dipole via the classical Corkum picture and
post-processes it through :mod:`harmonyemissions.spectrum`:

1. At each phase φ = ω₀ t, compute the ADK ionization rate and accumulate
   ionized population.
2. For each born-time t', propagate the freed electron classically in the
   laser field ``a(t) = a₀ envelope(t) cos(ωt + φ_CEP)``.
3. Recollision at t'' gives a dipole acceleration contribution
   ``d̈(t'') ∝ √(rate(t')) · E(t'') · exp(i(S(t'',t') − I_p (t''−t')/ħ))``.
4. Sum over born-times, take the real part, FFT → spectrum.

The Corkum cutoff ħω_max ≈ I_p + 3.17 U_p is reproduced by the classical
recollision energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from harmonyemissions.accel.jit import njit
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.spectrum import attosecond_pulse, field_to_spectrum, time_field
from harmonyemissions.target import Target, ionization_potential
from harmonyemissions.units import (
    E_CHARGE,
    HBAR,
    a0_to_intensity,
    ponderomotive_energy_ev,
)


@njit(cache=True)
def _accumulate_dipole(
    t: np.ndarray,
    cum_a: np.ndarray,
    cum_xa: np.ndarray,
    rate: np.ndarray,
    ip_over_photon: float,
    n: int,
) -> np.ndarray:
    """Born-time loop hot path — compiled with numba when available.

    Iterates born times j with stride 4, finds the first zero crossing of
    the post-birth displacement, and accumulates a cos-phase kernel onto
    the dipole array at the recollision index.
    """
    dipole = np.zeros(n)
    two_pi = 2.0 * math.pi
    for j in range(0, n - 1, 4):
        if rate[j] < 1e-8:
            continue
        sign_prev = 0.0
        k_rel = -1
        cum_a_j = cum_a[j]
        cum_xa_j = cum_xa[j]
        t_j = t[j]
        # Manual zero-crossing search of disp[k] for k in [j+1, n-1].
        for k in range(j + 1, n):
            dt_k = t[k] - t_j
            disp = -(cum_a[k + 1] - cum_a_j) * dt_k \
                   + (cum_xa[k + 1] - cum_xa_j - cum_a_j * dt_k)
            sign_cur = 1.0 if disp > 0.0 else -1.0 if disp < 0.0 else 0.0
            if sign_prev != 0.0 and sign_cur != 0.0 and sign_cur != sign_prev:
                k_rel = k
                break
            if sign_prev == 0.0:
                sign_prev = sign_cur
        if k_rel < 0 or k_rel >= n:
            continue
        k = k_rel
        dt_k = t[k] - t_j
        v_rec = -(cum_a[k + 1] - cum_a_j)
        phase = (cum_xa[k + 1] - cum_xa_j - cum_a_j * dt_k) - ip_over_photon * dt_k * two_pi
        amp = math.sqrt(rate[j]) * v_rec
        dipole[k] += amp * math.cos(phase)
    return dipole


@dataclass
class LewensteinModel:
    name: str = "lewenstein"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        ip_ev = ionization_potential(target)
        intensity = a0_to_intensity(laser.a0, laser.wavelength_um)
        up_ev = ponderomotive_energy_ev(intensity, laser.wavelength_um)
        # Corkum cutoff in units of harmonic order ħω / ħω₀.
        photon_ev = 1239.84198 / laser.wavelength_um  # ħω₀ in eV
        n_cutoff_corkum = (ip_ev + 3.17 * up_ev) / photon_ev

        t = laser.time_grid(numerics.n_periods, numerics.samples_per_period)
        a = laser.field(t)  # drive in units of a₀·env·cos phase

        # Classical electron trajectory: x''(t) = -E(t)/m (units: normalized).
        # In our normalized system E(t) ≡ a(t) (up to constants we absorb).
        # Build dipole d(t) = Σ_{t'} √(rate(t')) · x(t) · window(t,t')
        # where x(t) solves x'(t) = -∫_{t'}^{t} a(τ) dτ with x(t')=0, v(t')=0.
        n = t.size
        a_arr = a
        dt = float(t[1] - t[0])
        # Cumulative integrals for fast born-time propagation.
        cum_a = np.concatenate([[0.0], np.cumsum(a_arr) * dt])   # ∫₀^t a
        cum_xa = np.concatenate([[0.0], np.cumsum(cum_a[1:]) * dt])  # ∫₀^t (∫₀^τ a)

        # ADK-ish ionization rate: w(t) ∝ exp(-2(2 I_p)^{3/2} / (3|E|)).
        # We work in scaled units; |E(t)| ∝ |a(t)|, and define an effective
        # (I_p / U_p)^{3/2} exponent so the rate peaks at field maxima.
        # Normalize so that the rate integrates to 1 over the pulse — this
        # way the dipole amplitude is a pure shape, not a cross section.
        eps = 1e-6
        exponent = 2.0 * (2.0 * ip_ev / max(up_ev, eps)) ** 1.5 / 3.0
        rate = np.exp(-exponent / (np.abs(a_arr) + eps))
        rate /= max(rate.sum(), eps)

        # Sum of short-trajectory contributions — accelerated via numba
        # when available (see `_accumulate_dipole` above).
        dipole = _accumulate_dipole(
            t, cum_a, cum_xa, rate, float(ip_ev / photon_ev), n,
        )

        # Second derivative → acceleration spectrum.
        accel = np.gradient(np.gradient(dipole, dt), dt)
        spec = field_to_spectrum(t, accel)

        # Trim spectrum to a reasonable harmonic range for presentation.
        max_n = max(2 * n_cutoff_corkum, 200.0)
        spec = spec.where(spec.coords["harmonic"] <= max_n, drop=True)

        tf = time_field(t, accel)
        pulse = attosecond_pulse(t, accel, numerics.harmonic_window)

        return Result(
            spectrum=spec,
            time_field=tf,
            attosecond_pulse=pulse,
            diagnostics={
                "ionization_potential_eV": float(ip_ev),
                "ponderomotive_eV": float(up_ev),
                "n_cutoff_corkum": float(n_cutoff_corkum),
                "photon_energy_eV": float(photon_ev),
            },
            provenance={
                "model": "lewenstein",
                "reference": "Lewenstein et al., PRA 49, 2117 (1994)",
                "simplification": "classical 3-step with phase weighting",
            },
        )


# Keep HBAR/E_CHARGE referenced so they don't get dead-code-pruned by linters
# — they document that the amplitudes are in SI-friendly units.
_ = (HBAR, E_CHARGE)
