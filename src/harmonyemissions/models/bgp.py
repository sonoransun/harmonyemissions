"""Baeva–Gordienko–Pukhov universal spectrum for surface HHG.

Reference
---------
Baeva, Gordienko, Pukhov, *Phys. Rev. E* 74, 046404 (2006).

Key result
----------
In the ultra-relativistic limit (a₀ ≫ 1) the spectrum of harmonics emitted
from a relativistically oscillating overdense surface follows the universal
power law

    I(n) ∝ n^(-8/3) · F(n / n_c)

where n = ω/ω₀ is the harmonic order, n_c ≈ √(8 α) · γ_max³ is the cutoff
harmonic, and F is a smooth rolloff (approximated here by an exponential in
(n/n_c)^(2/3) — the standard closed-form envelope).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target
from harmonyemissions.units import gamma_from_a0

BGP_ALPHA = 1.0 / 137.036  # fine-structure constant
BGP_CUTOFF_PREFACTOR = math.sqrt(8.0 * BGP_ALPHA) * 2.0
# The factor of 2 absorbs Doppler upshift; the original BGP paper writes
# n_c ≃ 4√(2α) γ_max³ — used directly here.


@dataclass
class BGPModel:
    name: str = "bgp"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        # γ_max is the relativistic Lorentz factor of the oscillating surface,
        # taken to be the single-electron estimate in the incident field.
        gamma_max = gamma_from_a0(laser.a0, polarization="linear")
        n_cutoff = BGP_CUTOFF_PREFACTOR * gamma_max**3

        n = np.arange(1.0, 3.0 * n_cutoff + 1.0)
        # Universal power law with a smooth cutoff:
        #     I(n) = C · n^(-8/3) · exp(-(n/n_c)^(2/3))
        spectrum = n ** (-8.0 / 3.0) * np.exp(-((n / n_cutoff) ** (2.0 / 3.0)))

        spec_da = xr.DataArray(
            spectrum,
            coords={"harmonic": n},
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (universal BGP envelope)"},
        )
        return Result(
            spectrum=spec_da,
            diagnostics={
                "gamma_max": float(gamma_max),
                "n_cutoff": float(n_cutoff),
                "slope_theory": -8.0 / 3.0,
            },
            provenance={
                "model": "bgp",
                "reference": "Baeva, Gordienko, Pukhov PRE 74 046404 (2006)",
            },
        )
