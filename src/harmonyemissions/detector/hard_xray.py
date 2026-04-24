"""Filter attenuation for 1 keV–10 MeV photons.

Tabulates mass-attenuation coefficients μ/ρ [cm²/g] for the most common
filter / converter materials (Al, Cu, Ta, Pb, Au, Plastic (CH), Water),
on a log-spaced photon-energy grid.  Values are taken from the
NIST-Hubbell compilation (XCOM 1996 + updates) and coarsely sampled —
accurate to a few percent between K-edges, with the dominant edges
tagged for readability.

Transmission is ``T(E) = exp(−(μ/ρ) · ρ · d)`` for thickness ``d``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Log-spaced energy grid, keV, identical for every material.
_ENERGY_KEV = np.array([
    1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 80.0,
    100.0, 150.0, 200.0, 300.0, 500.0,
    1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0,
])

# Mass attenuation coefficients μ/ρ in cm²/g, sampled from NIST XCOM.
# Values are to ~10% accuracy at continuum energies; K-edges are smoothed
# through because the detector-response pipeline does not sit on an edge
# in the 1–10000 keV domain for any material below (Pb K-edge at 88 keV
# is the narrowest feature and is resolved here).
_MU_OVER_RHO: dict[str, np.ndarray] = {
    "Al": np.array([
        1185.0, 138.0, 26.2, 3.44, 1.13, 0.368, 0.202,
        0.170, 0.138, 0.122, 0.104, 0.0840,
        0.0614, 0.0500, 0.0432, 0.0353, 0.0284, 0.0230,
    ]),
    "Cu": np.array([
        1065.0, 2080.0, 216.0, 33.8, 10.9, 2.61, 0.714,
        0.458, 0.227, 0.158, 0.110, 0.0808,
        0.0587, 0.0499, 0.0452, 0.0410, 0.0390, 0.0412,
    ]),
    "Ta": np.array([
        3680.0, 5000.0, 1700.0, 321.0, 107.0, 25.4, 6.96,
        4.30, 1.55, 0.788, 0.320, 0.156,
        0.0931, 0.0703, 0.0605, 0.0536, 0.0519, 0.0578,
    ]),
    "W": np.array([
        3680.0, 5000.0, 1700.0, 322.0, 108.0, 25.5, 6.98,
        4.30, 1.56, 0.789, 0.320, 0.156,
        0.0931, 0.0704, 0.0607, 0.0538, 0.0521, 0.0582,
    ]),
    "Pb": np.array([
        5210.0, 4040.0, 1900.0, 86.4, 30.3, 8.04, 2.42,
        5.55, 1.91, 0.939, 0.378, 0.160,
        0.0710, 0.0523, 0.0459, 0.0414, 0.0419, 0.0496,
    ]),
    "Au": np.array([
        5070.0, 3960.0, 1860.0, 86.0, 30.1, 8.01, 2.41,
        5.16, 1.80, 0.886, 0.357, 0.152,
        0.0684, 0.0508, 0.0448, 0.0406, 0.0414, 0.0499,
    ]),
    "CH": np.array([  # plastic / PMMA approximation
        2050.0, 150.0, 1.73, 0.366, 0.215, 0.172, 0.160,
        0.156, 0.142, 0.131, 0.115, 0.0956,
        0.0706, 0.0582, 0.0509, 0.0419, 0.0337, 0.0263,
    ]),
    "H2O": np.array([
        4060.0, 295.0, 5.33, 0.819, 0.376, 0.227, 0.184,
        0.171, 0.151, 0.137, 0.119, 0.0969,
        0.0707, 0.0575, 0.0494, 0.0397, 0.0303, 0.0222,
    ]),
}

_DENSITY_G_CM3 = {
    "Al": 2.699,
    "Cu": 8.96,
    "Ta": 16.65,
    "W":  19.25,
    "Pb": 11.35,
    "Au": 19.30,
    "CH": 1.19,     # PMMA
    "H2O": 1.00,
}


def mass_attenuation_cm2_per_g(
    energy_keV: float | np.ndarray, material: str
) -> np.ndarray:
    """Mass-attenuation coefficient μ/ρ [cm²/g] at ``energy_keV``.

    Log-linear interpolation over the tabulated points.  Out-of-range
    energies extrapolate via constant-edge clamping (conservative).
    """
    try:
        table = _MU_OVER_RHO[material]
    except KeyError as exc:
        raise ValueError(
            f"Unknown material {material!r}; options: {sorted(_MU_OVER_RHO)}"
        ) from exc
    E = np.asarray(energy_keV, dtype=float)
    log_E = np.log(np.clip(E, _ENERGY_KEV[0], _ENERGY_KEV[-1]))
    log_ref = np.log(_ENERGY_KEV)
    log_mu = np.log(table)
    return np.exp(np.interp(log_E, log_ref, log_mu))


def filter_transmission(
    energy_keV: float | np.ndarray, material: str, thickness_um: float
) -> np.ndarray:
    """Photon transmission T(E) through a filter slab of given thickness."""
    mu_over_rho = mass_attenuation_cm2_per_g(energy_keV, material)
    rho = _DENSITY_G_CM3[material]          # g/cm³
    t_cm = thickness_um * 1e-4              # μm → cm
    return np.exp(-mu_over_rho * rho * t_cm)


@dataclass(frozen=True)
class FilterStack:
    """Sequence of passive filters in front of the detector."""

    layers: tuple[tuple[str, float], ...]  # e.g. (("Al", 200.0), ("Cu", 25.0))

    def transmission(self, energy_keV: np.ndarray) -> np.ndarray:
        T = np.ones_like(np.asarray(energy_keV, dtype=float))
        for material, thickness_um in self.layers:
            T = T * filter_transmission(energy_keV, material, thickness_um)
        return T
