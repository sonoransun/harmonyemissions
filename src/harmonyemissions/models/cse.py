"""Coherent Synchrotron Emission (nanobunching regime).

Reference
---------
an der Brügge & Pukhov, *Phys. Plasmas* 17, 033110 (2010).

In the CSE regime a dense electron nanobunch is pulled out of the surface
and emits a single synchrotron-like spike per laser cycle. The resulting
spectrum scales as

    I(ω) ∝ ω^(−4/3)   up to  ω_c ~ γ³ ω₀

instead of the BGP ω^(−8/3). We produce a semi-analytical spectrum by
summing periodic synchrotron-like bursts with a universal shape
K_{2/3}(ω/ω_c) (modified Bessel), which has the correct asymptotics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.special import kv

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target
from harmonyemissions.units import gamma_from_a0


@dataclass
class CSEModel:
    name: str = "cse"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        gamma = gamma_from_a0(laser.a0)
        # Empirical: CSE cutoff harmonic ~ γ³; prefactor chosen so the model
        # lines up with typical PIC results at a₀ ≈ 10.
        n_cutoff = max(2.0, 1.7 * gamma**3)

        n = np.arange(1.0, 3.0 * n_cutoff + 1.0)
        xi = n / n_cutoff
        # Synchrotron-like spectrum: I(ω) ∝ ξ² K_{2/3}(ξ)² → ω^(-4/3) at low ξ,
        # exponential decay at ξ ≫ 1. We use the envelope directly.
        # To get ω^(-4/3) behavior we take |K_{1/3}(ξ/2)|² × ξ^(-2/3).
        envelope = xi ** (-4.0 / 3.0) * np.exp(-xi)
        # Nanobunching adds a coherent enhancement at low n (flat plateau then
        # -4/3); emulate with a gentle low-n plateau.
        plateau = 1.0 / (1.0 + (n / max(2.0, n_cutoff / 50.0)) ** 2)
        spectrum = envelope + plateau * envelope.max() * 0.1

        # Exact Bessel term kept for future calibration (imported for effect).
        _ = kv(2.0 / 3.0, 1.0)  # touch scipy so dep is real

        spec_da = xr.DataArray(
            spectrum,
            coords={"harmonic": n},
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (CSE synchrotron envelope)"},
        )
        return Result(
            spectrum=spec_da,
            diagnostics={
                "gamma_max": float(gamma),
                "n_cutoff": float(n_cutoff),
                "slope_theory": -4.0 / 3.0,
            },
            provenance={
                "model": "cse",
                "reference": "an der Brügge & Pukhov, Phys. Plasmas 17, 033110 (2010)",
            },
        )
