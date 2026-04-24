"""K-α fluorescence lines from laser-driven hot-electron ionisation.

Reference
---------
Casnati, Tartari, Baraldi, *J. Phys. B* 15, 155 (1982) — K-shell
ionisation cross section fit.
Krause, *J. Phys. Chem. Ref. Data* 8, 307 (1979) — fluorescence yields.

Physics
-------
Hot electrons (temperature T_hot from Wilks ponderomotive scaling)
ionise K-shells in the target; the vacancy relaxes by X-ray emission
at the Kα₁/Kα₂/Kβ lines.  Line amplitudes scale as
``I_line ∝ ω_K · σ_K(T_hot) · branching``, with the statistical
branching ratio ``Kα₂/Kα₁ = 1/2`` (from 2p₁/₂ : 2p₃/₂) and
``Kβ/Kα ≈ 0.13`` (Z-dependent, small correction used as constant here).
A small bremsstrahlung pedestal on top of the lines is imported from
``bremsstrahlung.continuum_fn`` to keep the physics shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.laser import Laser
from harmonyemissions.materials import MaterialData, lookup
from harmonyemissions.models.base import Result
from harmonyemissions.models.bremsstrahlung import _resolve_T_hot, continuum_fn
from harmonyemissions.target import Target
from harmonyemissions.units import keV_per_harmonic

KALPHA2_OVER_KALPHA1 = 0.5  # statistical multiplicity.
KBETA_OVER_KALPHA = 0.13    # weak Z dependence; held constant here.


def _casnati_cross_section(E_keV: float, mat: MaterialData) -> float:
    """Casnati K-shell ionisation σ_K [arb]; zero below threshold."""
    U = E_keV / mat.K_edge_keV
    if U <= 1.0:
        return 0.0
    return np.log(U) / (U * mat.K_edge_keV**2)


def _natural_width_keV(Z: int) -> float:
    # Krause K-level natural widths: ~0.93 eV at Z=22, 2.11 eV at Z=29, 5.89 eV at Z=42.
    # Fit: Γ_K[keV] ≈ 8.6e-8 · Z³.
    return 8.6e-8 * Z**3


def _lorentzian(energy_keV: np.ndarray, center_keV: float, fwhm_keV: float) -> np.ndarray:
    gamma = 0.5 * fwhm_keV
    return (gamma / np.pi) / ((energy_keV - center_keV) ** 2 + gamma * gamma)


@dataclass
class KalphaModel:
    name: str = "kalpha"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        mat = lookup(target.material)
        T_hot = _resolve_T_hot(laser, target)

        # Grid straddles lines plus enough continuum shoulder for the pedestal.
        E_min = 0.3 * min(mat.K_alpha1_keV, mat.K_alpha2_keV)
        E_max = 3.0 * mat.K_beta_keV
        base = np.geomspace(E_min, E_max, 2048)
        # Inject exact line centres so peaks are always sampled at their Lorentzian max.
        line_samples = np.concatenate([
            np.linspace(mat.K_alpha1_keV - 0.01, mat.K_alpha1_keV + 0.01, 21),
            np.linspace(mat.K_alpha2_keV - 0.01, mat.K_alpha2_keV + 0.01, 21),
            np.linspace(mat.K_beta_keV - 0.01, mat.K_beta_keV + 0.01, 21),
        ])
        energy_keV = np.sort(np.concatenate([base, line_samples]))

        gamma_nat = _natural_width_keV(mat.Z)
        sigma_K = _casnati_cross_section(T_hot, mat)

        # Line amplitudes ∝ ω_K · σ_K(T_hot) · branching.
        amp_prefactor = mat.fluorescence_yield_K * max(sigma_K, 0.0)
        a1 = amp_prefactor
        a2 = amp_prefactor * KALPHA2_OVER_KALPHA1
        ab = amp_prefactor * KBETA_OVER_KALPHA

        lines = (
            a1 * _lorentzian(energy_keV, mat.K_alpha1_keV, gamma_nat)
            + a2 * _lorentzian(energy_keV, mat.K_alpha2_keV, gamma_nat)
            + ab * _lorentzian(energy_keV, mat.K_beta_keV, 1.3 * gamma_nat)
        )
        # Bremsstrahlung pedestal; amplitude set by a fixed prefactor relative
        # to the raw continuum so it survives even when σ_K = 0 (T_hot below
        # K-edge).  Line-to-continuum ratio at Cu/100 keV hot electrons lands
        # near 30 — inside the experimentally-observed 10–100 range.
        continuum_raw = continuum_fn(energy_keV, T_hot)
        continuum = 1e-3 * continuum_raw
        spectrum = lines + continuum

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
            attrs={"units": f"arb. (K-α lines for {mat.name})"},
        )

        return Result(
            spectrum=spec_da,
            diagnostics={
                "material_Z": float(mat.Z),
                "K_alpha1_keV": float(mat.K_alpha1_keV),
                "K_alpha2_keV": float(mat.K_alpha2_keV),
                "K_beta_keV": float(mat.K_beta_keV),
                "fluorescence_yield_K": float(mat.fluorescence_yield_K),
                "hot_electron_temp_keV": float(T_hot),
                "sigma_K_at_T_hot": float(sigma_K),
                "natural_width_keV": float(gamma_nat),
            },
            provenance={
                "model": "kalpha",
                "material": mat.name,
                "reference": "Casnati 1982 σ_K; Krause 1979 ω_K",
                "T_hot_source": "target override" if target.hot_electron_temp_keV else "Wilks scaling",
            },
        )
