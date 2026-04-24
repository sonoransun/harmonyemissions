"""Hard-X-ray and gamma-ray physics primitives.

- `compton`  — Klein–Nishina cross section and inverse-Compton-scattering
  (ICS) energy/spectrum helpers.
- `bremsstrahlung` — Bethe–Heitler bremsstrahlung from an electron beam
  impinging on a thin high-Z converter (distinct from the hot-electron
  Kramers–Maxwell continuum in ``models/bremsstrahlung.py``).
- `radiation_reaction` — quantum nonlinearity parameter χ_e and the
  Gaunt-factor-style correction to the synchrotron envelope.
- `angular` — synchrotron emission cone, divergence, and brightness.
"""

from harmonyemissions.gamma.angular import (
    brightness_estimate,
    divergence_cone_fwhm_mrad,
    synchrotron_angular_pattern,
)
from harmonyemissions.gamma.bremsstrahlung import (
    bethe_heitler_spectrum,
    converter_photon_yield,
    radiation_length_g_per_cm2,
)
from harmonyemissions.gamma.compton import (
    compton_max_photon_energy_keV,
    ics_photon_spectrum_keV,
    klein_nishina_total_cross_section,
    thomson_cross_section,
)
from harmonyemissions.gamma.radiation_reaction import (
    chi_e_parameter,
    quantum_synchrotron_suppression,
)

__all__ = [
    "bethe_heitler_spectrum",
    "brightness_estimate",
    "chi_e_parameter",
    "compton_max_photon_energy_keV",
    "converter_photon_yield",
    "divergence_cone_fwhm_mrad",
    "ics_photon_spectrum_keV",
    "klein_nishina_total_cross_section",
    "quantum_synchrotron_suppression",
    "radiation_length_g_per_cm2",
    "synchrotron_angular_pattern",
    "thomson_cross_section",
]
