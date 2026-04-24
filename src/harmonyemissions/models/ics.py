"""Inverse Compton scattering (ICS) source.

Model
-----
An electron bunch — either from an LWFA (``target.kind == "underdense"``)
or externally injected (``target.kind == "electron_beam"``) — collides
head-on with a counter-propagating scattering laser whose parameters
come from :class:`harmonyemissions.laser.Laser` (the ``Laser`` argument
is treated as the *scattering* pump in this model).

The scattered-photon spectrum is built from

  1.  The classical cutoff energy E_γ^max (``gamma.compton.compton_max_photon_energy_keV``).
  2.  A sharp-edge plateau with a 1/E² tail above the cutoff (collection-
      angle proxy, see ``gamma.compton.ics_photon_spectrum_keV``).
  3.  Quantum-χ Gaunt-factor suppression at the highest energies (see
      ``gamma.radiation_reaction``).
  4.  An absolute photon yield estimate from σ_KN × bunch charge ×
      laser pulse photon count.

The Klein–Nishina recoil enters twice: in the kinematic cutoff formula
(denominator ``1 + 4 γ E_L / m_e c²``) and via the rest-frame total
cross section that sets the absolute yield.

References
----------
- Sarri, G. et al., *Phys. Rev. Lett.* **113**, 224801 (2014) — all-
  optical Compton γ-source demonstration.
- Khrennikov, K. et al., *Phys. Rev. Lett.* **114**, 195003 (2015) —
  ICS brightness measurement.
- Corde, S. et al., *Rev. Mod. Phys.* **85**, 1 (2013) §V — theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from harmonyemissions.gamma.compton import (
    compton_max_photon_energy_keV,
    ics_photon_spectrum_keV,
    klein_nishina_total_cross_section,
)
from harmonyemissions.gamma.radiation_reaction import (
    chi_e_parameter,
    quantum_synchrotron_suppression,
)
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target
from harmonyemissions.units import (
    HC_KEV_NM,
    M_E_C2_KEV,
    a0_to_intensity,
    default_xray_energy_grid,
    keV_per_harmonic,
)


def _gamma_e_from_target(target: Target) -> float:
    if target.kind == "electron_beam":
        E_mev = target.beam_energy_mev
    else:
        E_mev = target.electron_energy_mev
    return float(E_mev / 0.511 + 1.0)


def _beam_charge_pC(target: Target) -> float:
    if target.kind == "electron_beam":
        return float(target.beam_charge_pc)
    # LWFA default bunch charge estimate — rough, scales with a₀ of the
    # drive laser elsewhere, but we take a 100 pC fiducial since the
    # ``Target.underdense`` factory doesn't carry explicit charge.
    return 100.0


def _laser_photons_per_pulse(laser: Laser) -> float:
    """Rough count: I·πw² / (ħω) integrated over pulse duration."""
    intensity = a0_to_intensity(laser.a0, laser.wavelength_um)  # W/cm²
    spot_area_cm2 = np.pi * (0.5 * laser.spot_fwhm_um * 1e-4) ** 2
    duration_s = laser.duration_fs * 1e-15
    energy_J = intensity * spot_area_cm2 * duration_s
    photon_eV = 1.2398419843320025 / laser.wavelength_um * 1e3  # hc/λ
    return float(energy_J / (photon_eV * 1.602176634e-19))


@dataclass
class ICSModel:
    name: str = "ics"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        gamma_e = _gamma_e_from_target(target)
        E_max_keV = compton_max_photon_energy_keV(gamma_e, laser.wavelength_um)
        # Photon-energy grid: decade below cutoff to one decade above.
        energy_keV = default_xray_energy_grid(
            E_min_keV=max(1.0, E_max_keV * 1e-3),
            E_max_keV=max(100.0, E_max_keV * 10.0),
            n_points=2048,
        )
        _, spectrum = ics_photon_spectrum_keV(
            gamma_e, laser.wavelength_um, energy_keV
        )

        # Apply quantum-χ suppression at the high-energy tail.  The
        # rest-frame photon energy seen by the electron in a head-on
        # geometry is ~2 γ · E_L, so the field is effectively boosted.
        E_L_keV = HC_KEV_NM * 1e-3 / laser.wavelength_um
        rest_frame_keV = 2.0 * gamma_e * E_L_keV
        # Normalise the transverse field of the laser: E_L [V/m] from a₀
        # by a₀ = eE_L / (m_e c ω_L).
        omega_L = 2.0 * np.pi * 2.99792458e8 / (laser.wavelength_um * 1e-6)
        E_L_field = laser.a0 * 9.1093837015e-31 * 2.99792458e8 * omega_L / 1.602176634e-19
        chi = float(chi_e_parameter(gamma_e, 2.0 * gamma_e * E_L_field))
        # Per-energy χ proxy: scale by E/E_max so suppression kicks in at
        # the cutoff where recoil dominates.
        chi_per_E = chi * energy_keV / max(E_max_keV, 1.0)
        spectrum = spectrum * quantum_synchrotron_suppression(chi_per_E)

        # Absolute photon yield.  σ_KN evaluated at the rest-frame photon
        # energy gives the cross section per interaction; multiply by
        # electron count × laser-photon count × geometric factor (~1/area).
        N_e = _beam_charge_pC(target) * 1e-12 / 1.602176634e-19
        N_L = _laser_photons_per_pulse(laser)
        sigma = float(klein_nishina_total_cross_section(rest_frame_keV))
        spot_area_m2 = np.pi * (0.5 * laser.spot_fwhm_um * 1e-6) ** 2
        N_gamma_total = N_e * N_L * sigma / max(spot_area_m2, 1e-30)

        # Harmonic coord for Result uniformity.
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
            attrs={"units": "arb. (ICS spectrum with KN suppression)"},
        )

        return Result(
            spectrum=spec_da,
            diagnostics={
                "gamma_e": gamma_e,
                "photon_energy_keV_cutoff": float(E_max_keV),
                "photon_energy_keV_peak": float(E_max_keV * 0.9),
                "rest_frame_photon_keV": float(rest_frame_keV),
                "chi_e_nominal": float(chi),
                "klein_nishina_cross_section_m2": float(sigma),
                "n_photons_per_pulse": float(N_gamma_total),
                "recoil_parameter_x": float(rest_frame_keV / M_E_C2_KEV),
            },
            provenance={
                "model": "ics",
                "reference": (
                    "Sarri et al., PRL 113, 224801 (2014); "
                    "Khrennikov et al., PRL 114, 195003 (2015); "
                    "Corde et al., RMP 85, 1 (2013)"
                ),
            },
        )
