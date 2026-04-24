"""Hot-electron bremsstrahlung continuum from overdense targets.

Reference
---------
Kramers (1923) thick-target formula; Wilks et al., *Phys. Rev. Lett.*
69, 1383 (1992) for the ponderomotive hot-electron temperature.

Physics
-------
Laser-driven fast electrons penetrating the bulk of a solid target emit
bremsstrahlung with an approximately Kramers-law single-electron
spectrum ``dN/dE ∝ (E_max/E − 1)`` up to ``E_max = E_e``.  Averaging
over a Maxwell-Jüttner hot-electron distribution with temperature T_hot
yields the closed form

    dI/dE ∝ E1(E / T_hot)

with ``E1`` the exponential integral, giving a slope ≈ ``-1/T_hot`` in
semilog at ``E ≳ T_hot`` and a soft divergence at low E.  T_hot is
derived inline from Wilks ponderomotive scaling unless
``target.hot_electron_temp_keV`` overrides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.special import exp1

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target
from harmonyemissions.units import (
    hot_electron_temperature_keV,
    keV_per_harmonic,
)


def continuum_fn(energy_keV: np.ndarray, T_hot_keV: float) -> np.ndarray:
    """Maxwell-averaged Kramers continuum dI/dE (unnormalised) [arb/keV]."""
    x = np.asarray(energy_keV, dtype=float) / float(T_hot_keV)
    # exp1(x) → −γ_Euler − ln(x) + x − ... at small x (diverges logarithmically);
    # clip the low-E end so the spectrum is finite on grid boundaries.
    x = np.clip(x, 1e-6, None)
    return exp1(x)


def _resolve_T_hot(laser: Laser, target: Target) -> float:
    if target.hot_electron_temp_keV is not None:
        return float(target.hot_electron_temp_keV)
    return float(hot_electron_temperature_keV(laser.a0, laser.wavelength_um))


@dataclass
class BremsstrahlungModel:
    name: str = "bremsstrahlung"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        T_hot = _resolve_T_hot(laser, target)
        # Span 0.1·T_hot to 10·T_hot — captures the low-E logarithmic rise
        # and the high-E exponential tail.
        energy_keV = np.geomspace(0.1 * T_hot, 10.0 * T_hot, 4096)
        spectrum = continuum_fn(energy_keV, T_hot)

        # Map back to harmonic-order coord for Result-shape consistency.
        keV_per_n = keV_per_harmonic(laser.wavelength_um)
        harmonic = energy_keV / keV_per_n

        spec_da = xr.DataArray(
            spectrum,
            coords={
                "harmonic": harmonic,
                "photon_energy_keV": ("harmonic", energy_keV),
            },
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (Kramers-Maxwell bremsstrahlung)"},
        )

        # Efficiency prefactor: ~η = (4/3)·α·Z/(π·m_ec²)·T_hot is ~1e-4 for Z=14–29.
        efficiency = 1.4e-4 * (target.n_over_nc / 100.0)

        return Result(
            spectrum=spec_da,
            diagnostics={
                "hot_electron_temp_keV": float(T_hot),
                "photon_energy_keV_efolding": float(T_hot),
                "photon_energy_keV_peak": float(energy_keV[int(np.argmax(energy_keV * spectrum))]),
                "efficiency_proxy": float(efficiency),
            },
            provenance={
                "model": "bremsstrahlung",
                "reference": "Kramers 1923; Wilks et al., PRL 69 1383 (1992)",
                "T_hot_source": "target override" if target.hot_electron_temp_keV else "Wilks scaling",
            },
        )
