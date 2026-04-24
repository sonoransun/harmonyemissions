"""Relativistic Oscillating Mirror model of surface HHG.

Design
------
The ROM model has two jobs:

1. Compute the time-domain reflected field E_r(t) via the Doppler-compressed
   retarded-time map, so that users can synthesize attosecond pulses and
   inspect the field shape.
2. Report a harmonic spectrum that matches the canonical ROM scaling —
   a plateau falling as n^(−8/3) up to a cutoff n_c ∼ 2 γ_peak² (Doppler)
   or ∼ γ_peak³ (BGP universal) depending on regime.

Rather than FFT a coarsely-sampled time-domain field (which caps at the
Nyquist harmonic and smears the cutoff), we combine:

- a parametric mirror motion β_s(t) → x_s(t) → t_r(t) → E_r(t) for the
  time-domain outputs, and
- the BGP universal envelope I(n) ∝ n^(−8/3) exp(−(n/n_c)^(2/3)) for the
  spectrum, anchored by the mirror's γ_peak.

This follows Bulanov et al. (1994) / Gonoskov et al. (2011) in spirit: a
reduced single-degree-of-freedom picture of the plasma surface. Going
beyond it (e.g. tracking the full Lorentz equation with radiation reaction,
multi-mode surface oscillations, CSE nanobunching) is the job of the PIC
backend — see :mod:`harmonyemissions.backends.smilei`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.models.bgp import BGP_CUTOFF_PREFACTOR
from harmonyemissions.spectrum import attosecond_pulse, time_field
from harmonyemissions.target import Target
from harmonyemissions.units import gamma_from_a0


@dataclass
class ROMModel:
    name: str = "rom"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        # ---- mirror motion -------------------------------------------------
        t = laser.time_grid(numerics.n_periods, numerics.samples_per_period)
        env = laser.envelope_value(t)
        phase = 2.0 * math.pi * t + laser.cep

        # Mirror peak γ: in the ultra-relativistic limit γ_mirror ≈ a₀,
        # reduced by a gradient-dependent factor that reflects how well the
        # plasma can sustain coherent surface motion.
        gradient_factor = 1.0 / (1.0 + 5.0 * target.gradient_L_over_lambda ** 2)
        density_factor = 1.0 if target.n_over_nc >= 10.0 else (target.n_over_nc / 10.0)
        gamma_peak = max(1.0, laser.a0 * gradient_factor * density_factor)
        beta_peak = math.sqrt(max(0.0, 1.0 - 1.0 / gamma_peak**2))
        beta_peak = float(min(beta_peak, 0.9995))

        # Sharpen the motion toward the "spike" regime: higher a₀ → sharper
        # surface transits via the tanh compressor.
        compressor = min(8.0, 0.5 + laser.a0)
        beta_s = beta_peak * (env ** 2) * np.tanh(compressor * np.sin(phase))
        dt = float(t[1] - t[0])
        x_s = np.cumsum(beta_s) * dt

        # ---- reflected field (time domain) --------------------------------
        t_r = t.copy()
        for _ in range(4):
            t_r = t - 2.0 * np.interp(t_r, t, x_s)
        e_reflected = laser.field(t_r) * env

        # ---- harmonic spectrum: BGP universal envelope anchored at γ_peak -
        n_cutoff = BGP_CUTOFF_PREFACTOR * gamma_peak**3
        n = np.arange(1.0, 3.0 * n_cutoff + 1.0)
        spectrum_vals = n ** (-8.0 / 3.0) * np.exp(-((n / n_cutoff) ** (2.0 / 3.0)))
        spec_da = xr.DataArray(
            spectrum_vals,
            coords={"harmonic": n},
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (BGP-anchored ROM envelope)"},
        )

        # ---- package ------------------------------------------------------
        tf = time_field(t, e_reflected)
        pulse = attosecond_pulse(t, e_reflected, numerics.harmonic_window)

        return Result(
            spectrum=spec_da,
            time_field=tf,
            attosecond_pulse=pulse,
            diagnostics={
                "gamma_max_single_particle": float(gamma_from_a0(laser.a0)),
                "beta_s_peak": beta_peak,
                "gamma_mirror_peak": float(gamma_peak),
                "n_cutoff": float(n_cutoff),
                "gradient_factor": float(gradient_factor),
                "density_factor": float(density_factor),
                "slope_theory": -8.0 / 3.0,
            },
            provenance={
                "model": "rom",
                "variant": "reduced-parametric-mirror + BGP-envelope",
                "reference": "Bulanov (1994); Gonoskov (2011); Baeva et al. (2006)",
            },
        )
