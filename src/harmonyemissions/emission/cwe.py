"""Coherent Wake Emission (CWE) — sub-relativistic surface HHG.

Reference
---------
Quéré, F. et al. *Coherent wake emission of high-order harmonics from
overdense plasmas.* Phys. Rev. Lett. **96**, 125004 (2006). Also ref. 44
of Timmis 2026.

Physics
-------
For driver intensities below the relativistic threshold (a₀ < 1, I <
10¹⁸ W/cm²) SHHG is still produced but via a different mechanism:
Brunel electrons pulled out into vacuum and sent back into the plasma
excite a plasma-frequency wakefield that radiates coherently. The
signature is a **cutoff at the plasma frequency** of the target
(ω ≤ ω_p) rather than the ~γ³ cutoff of ROM/BGP — so CWE saturates at
harmonic n_p = √(n_e/n_c), independent of a₀.

Model
-----
We produce an analytical CWE spectrum

    I(n) = (n/n_p)^(−4) · Θ(n_p − n)     for n ≤ n_p
         = I(n_p) · exp(−(n − n_p)²/Δ²)  beyond (smooth tail)

This envelope has the three correct qualitative features: a power-law
plateau with steeper slope than ROM (~n^{−4} vs ~n^{−8/3}), a sharp
cutoff at the plasma frequency harmonic, and no a₀ dependence of the
cutoff position.

CWE is a surface-HHG model but uses a different ``Target`` regime
(overdense plasmas with any gradient) — most useful at mildly relativistic
intensities where ROM/BGP over-predicts emission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target


@dataclass
class CWEModel:
    name: str = "cwe"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        # Plasma cutoff harmonic: n_p = √(n_e / n_c).
        n_p = math.sqrt(max(target.n_over_nc, 1e-9))

        n = np.arange(1.0, max(2 * n_p, 50.0))
        spectrum = np.empty_like(n)
        plateau = (n / n_p) ** (-4.0)
        tail_width = max(2.0, 0.2 * n_p)
        tail = (n_p / n_p) ** (-4.0) * np.exp(-((n - n_p) ** 2) / tail_width ** 2)
        spectrum = np.where(n <= n_p, plateau, tail)
        spectrum = np.maximum(spectrum, 1e-40)

        spec_da = xr.DataArray(
            spectrum,
            coords={"harmonic": n},
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (CWE envelope)"},
        )
        return Result(
            spectrum=spec_da,
            diagnostics={
                "n_plasma_cutoff": float(n_p),
                "slope_theory": -4.0,
                "mechanism": 1.0,  # 1 = Brunel / coherent wake
            },
            provenance={
                "model": "cwe",
                "reference": "Quéré et al., Phys. Rev. Lett. 96, 125004 (2006)",
            },
        )
