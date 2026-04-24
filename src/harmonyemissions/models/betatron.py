"""Betatron radiation from an electron oscillating in an LWFA ion bubble.

Reference
---------
Kostyukov, Pukhov, Kiselev, *Phys. Plasmas* **11**, 5256 (2004);
Corde et al., *Rev. Mod. Phys.* **85**, 1 (2013).  Quantum-χ correction
from Niel et al., *Phys. Rev. E* **97**, 043209 (2018).

Physics
-------
Inside the bubble, the transverse restoring force is linear and the
electron executes betatron oscillations at

    ω_β = ω_p / √(2γ)

where γ is the electron Lorentz factor and ω_p is the background plasma
frequency. The classical spectrum follows the universal synchrotron
envelope

    dI/dω ∝ (ω/ω_c) · K_{2/3}(ω/2ω_c)²

with critical frequency

    ω_c = (3/2) γ³ ω_β² r_β / c = (3/4) γ² K ω_β

and K = γ k_β r_β the wiggler strength.

Extreme-energy regime
---------------------
At multi-GeV beam energies the quantum nonlinearity parameter

    χ_e = γ · |E_⊥| / E_Schwinger

becomes comparable to unity.  Radiation reaction suppresses photon
emission beyond a fraction of the classical cutoff; we multiply the
classical envelope by the Gaunt factor g(χ_e) from
:mod:`harmonyemissions.gamma.radiation_reaction`.

The result carries both the harmonic-order and absolute-photon-energy
coordinates so users can feed the spectrum into either the XUV pipeline
(``detector.apply_instrument_response``) or the hard-X-ray / γ pipeline
(``detector.apply_gamma_response``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.accel.bessel import kv_two_thirds_half
from harmonyemissions.gamma.angular import (
    brightness_estimate,
    divergence_cone_fwhm_mrad,
)
from harmonyemissions.gamma.radiation_reaction import (
    betatron_field_estimate_V_per_m,
    chi_e_parameter,
    quantum_synchrotron_suppression,
)
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target
from harmonyemissions.units import (
    E_CHARGE,
    M_E,
    C,
    keV_per_harmonic,
    omega_from_wavelength,
    plasma_frequency,
)


@dataclass
class BetatronModel:
    name: str = "betatron"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        lambda_m = laser.wavelength_um * 1e-6
        omega0 = omega_from_wavelength(lambda_m)

        n_c = 8.8541878128e-12 * M_E * omega0**2 / E_CHARGE**2
        n_e = target.n_over_nc * n_c
        omega_p = plasma_frequency(n_e)

        gamma_e = target.electron_energy_mev / 0.511 + 1.0
        omega_b = omega_p / math.sqrt(2.0 * gamma_e)       # betatron frequency
        r_b = target.betatron_amplitude_um * 1e-6          # oscillation amplitude
        k_b = omega_b / C
        K = gamma_e * k_b * r_b                            # wiggler strength

        omega_c = 0.75 * gamma_e**2 * K * omega_b          # critical angular frequency

        # Harmonic-order grid: log-spaced from a thousandth of the cutoff
        # to twice the cutoff — spans the 1/3-power rise, the peak, and
        # the first decade of exponential fall.
        n_c_harm = omega_c / omega0
        n_min = max(1.0, n_c_harm * 1e-3)
        n_max = max(2.0 * n_c_harm, 50.0)
        n = np.geomspace(n_min, n_max, 2048)
        xi = n * (omega0 / omega_c)
        env = xi * kv_two_thirds_half(xi)
        env[~np.isfinite(env)] = 0.0

        # --- Quantum χ correction (extreme-energy regime) ---------------
        E_transverse = betatron_field_estimate_V_per_m(gamma_e, omega_b, r_b)
        chi_e = float(chi_e_parameter(gamma_e, E_transverse))
        # Apply the Gaunt factor per photon energy, so the correction
        # only bites the high-ω tail.  χ-per-photon rises ∝ ω/ω_c:
        chi_per_photon = chi_e * xi
        suppression = quantum_synchrotron_suppression(chi_per_photon)
        env = env * suppression

        keV_per_n = keV_per_harmonic(laser.wavelength_um)
        photon_energy_keV = n * keV_per_n
        E_crit_keV = n_c_harm * keV_per_n
        E_peak_keV = 0.29 * E_crit_keV                      # ξ ≈ 0.29 peak

        # --- Photon-number estimate (rough) ------------------------------
        photons_per_pulse_estimate = 5.6e-3 * gamma_e * gamma_e * K * K

        # --- Angular diagnostics ----------------------------------------
        divergence_mrad = divergence_cone_fwhm_mrad(gamma_e, K)
        brightness = brightness_estimate(
            n_photons_per_pulse=photons_per_pulse_estimate,
            divergence_mrad=divergence_mrad,
            bandwidth_percent=0.1,
            source_size_um=target.betatron_amplitude_um,
            rep_rate_hz=1.0,
        )

        spec_da = xr.DataArray(
            env,
            coords={
                "harmonic": n,
                "photon_energy_keV": ("harmonic", photon_energy_keV),
            },
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (betatron synchrotron envelope w/ quantum χ)"},
        )

        return Result(
            spectrum=spec_da,
            diagnostics={
                "gamma_e": float(gamma_e),
                "omega_p_over_omega0": float(omega_p / omega0),
                "omega_b_over_omega0": float(omega_b / omega0),
                "K_wiggler": float(K),
                "omega_c_over_omega0": float(n_c_harm),
                "photon_energy_keV_critical": float(E_crit_keV),
                "photon_energy_keV_peak": float(E_peak_keV),
                "photon_energy_keV_at_cutoff": float(E_crit_keV),
                "photons_per_pulse_estimate": float(photons_per_pulse_estimate),
                # Extreme-energy enhancements:
                "chi_e": chi_e,
                "quantum_suppression_at_cutoff": float(suppression[-1]),
                "transverse_field_V_per_m": float(E_transverse),
                "divergence_FWHM_mrad": float(divergence_mrad),
                "brightness_peak_ph_s_mm2_mrad2_0p1bw": float(brightness),
            },
            provenance={
                "model": "betatron",
                "reference": (
                    "Kostyukov et al., Phys. Plasmas 11, 5256 (2004); "
                    "Corde et al., Rev. Mod. Phys. 85, 1 (2013); "
                    "Niel et al., PRE 97, 043209 (2018)"
                ),
            },
        )
