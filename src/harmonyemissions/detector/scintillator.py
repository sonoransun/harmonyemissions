"""Scintillator / semiconductor detector efficiency for 1 keV – 10 MeV.

We approximate the detector absorption efficiency as the attenuation
*inside* the active layer of the detector — same μ/ρ tables as the
filter module but with a "deposit fraction" that folds in photoelectric
absorption vs Compton scattering at MeV energies.

Detector menu:

- ``NaI(Tl)``  — sodium iodide scintillator, ρ = 3.67 g/cm³, Z_eff = 50.
- ``CsI(Tl)``  — caesium iodide, ρ = 4.51, Z_eff = 54.
- ``LYSO``     — lutetium-yttrium orthosilicate, ρ = 7.1, Z_eff = 66.
- ``CdTe``     — cadmium telluride semiconductor, ρ = 6.2, Z_eff = 50.
- ``HPGe``     — hyper-pure germanium, ρ = 5.32, Z = 32.
- ``Si``       — silicon photodiode, ρ = 2.33, Z = 14.
- ``YAG(Ce)``  — yttrium aluminium garnet, ρ = 4.55, Z_eff = 32.
- ``IP``       — phosphor imaging plate (BaFBr:Eu²⁺), ρ = 3.0, Z_eff = 47.

Numbers here are representative, not certification-grade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harmonyemissions.detector.hard_xray import mass_attenuation_cm2_per_g

_DETECTOR_DENSITY_G_CM3 = {
    "NaI":  3.67,
    "CsI":  4.51,
    "LYSO": 7.1,
    "CdTe": 6.2,
    "HPGe": 5.32,
    "Si":   2.33,
    "YAG":  4.55,
    "IP":   3.0,
}

# Reuse the μ/ρ tables from the filter module for the element that
# dominates each composite; a more careful implementation would compose
# Bragg additivity across compounds.
_DETECTOR_PROXY_ELEMENT = {
    "NaI":  "Pb",   # high-Z-dominated; Pb is closest element tabulated
    "CsI":  "Pb",
    "LYSO": "Pb",
    "CdTe": "Cu",
    "HPGe": "Cu",
    "Si":   "Al",
    "YAG":  "Al",
    "IP":   "Cu",
}


@dataclass(frozen=True)
class DetectorConfig:
    name: str = "CsI"
    thickness_mm: float = 5.0
    # Fraction of deposited Compton energy that contributes to the
    # recorded signal (photoelectric = 1; Compton varies with energy).
    compton_efficiency: float = 0.3

    @property
    def density_g_cm3(self) -> float:
        return _DETECTOR_DENSITY_G_CM3[self.name]

    @property
    def proxy_element(self) -> str:
        return _DETECTOR_PROXY_ELEMENT[self.name]


def detector_absorption(
    energy_keV: np.ndarray, config: DetectorConfig
) -> np.ndarray:
    """Absorption fraction F(E) of photons inside the active layer.

    ``F = 1 − exp(−(μ/ρ)·ρ·t)`` with μ/ρ from the filter tables for the
    dominant element.  Does **not** include the Compton escape
    correction — that's handled by :func:`detector_response`.
    """
    mu_rho = mass_attenuation_cm2_per_g(energy_keV, config.proxy_element)
    t_cm = config.thickness_mm * 0.1
    return 1.0 - np.exp(-mu_rho * config.density_g_cm3 * t_cm)


def detector_response(
    energy_keV: np.ndarray, config: DetectorConfig
) -> np.ndarray:
    """Total detector efficiency ``ε(E)`` — absorption × deposit fraction.

    Above ~500 keV, Compton scattering dominates; a fraction of the
    scattered photon escapes the active volume.  We approximate the
    deposited-energy efficiency as

        ε(E) = F(E) · [PE_fraction + (1 − PE_fraction) · compton_eff]

    where ``PE_fraction`` falls from 1 below ~100 keV to ~0.1 at 1 MeV.
    """
    F = detector_absorption(energy_keV, config)
    # Photoelectric-vs-Compton cross-over sigmoid (not a physics fit —
    # a plausible shape that integrates correctly).
    E = np.asarray(energy_keV, dtype=float)
    pe_frac = 1.0 / (1.0 + (E / 200.0) ** 2)
    return F * (pe_frac + (1.0 - pe_frac) * config.compton_efficiency)
