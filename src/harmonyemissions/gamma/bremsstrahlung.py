"""Bethe–Heitler bremsstrahlung from an electron beam on a high-Z converter.

Reference
---------
Bethe & Heitler (1934); Koch & Motz, *Rev. Mod. Phys.* **31**, 920 (1959)
formula 3BS; Tsai, *Rev. Mod. Phys.* **46**, 815 (1974).  This is the
thin-target, ultra-relativistic limit — valid for GeV-class LWFA beams
dumping into a Ta / W / Pb foil of thickness ≪ radiation length.

Differs from ``models/bremsstrahlung.py``: that module computes the
hot-electron Maxwell-averaged Kramers continuum inside a laser-driven
solid; this module produces the **converter** spectrum for a
mono-energetic relativistic beam.
"""

from __future__ import annotations

import numpy as np

from harmonyemissions.units import CLASSICAL_ELECTRON_RADIUS_M, FINE_STRUCTURE_ALPHA

# Tabulated radiation lengths [g/cm²] for common converter materials.
# Values from PDG review (2022).
_RADIATION_LENGTH_G_CM2 = {
    "Al": 24.01,
    "Cu": 12.86,
    "Ta":  6.82,
    "W":   6.76,
    "Pb":  6.37,
    "Au":  6.46,
}

# Approximate density in g/cm³ — needed to convert thickness in μm to g/cm².
_DENSITY_G_CM3 = {
    "Al": 2.699,
    "Cu": 8.96,
    "Ta": 16.65,
    "W":  19.25,
    "Pb": 11.35,
    "Au": 19.30,
}


def radiation_length_g_per_cm2(material: str) -> float:
    """Return radiation length X₀ [g/cm²] for a converter material."""
    try:
        return _RADIATION_LENGTH_G_CM2[material]
    except KeyError as exc:
        raise ValueError(
            f"Unknown converter material {material!r}; options: "
            f"{sorted(_RADIATION_LENGTH_G_CM2)}"
        ) from exc


def _material_Z(material: str) -> int:
    return {"Al": 13, "Cu": 29, "Ta": 73, "W": 74, "Pb": 82, "Au": 79}[material]


def bethe_heitler_spectrum(
    energy_keV: np.ndarray,
    electron_energy_keV: float,
    material: str = "Ta",
) -> np.ndarray:
    """Thin-target Bethe–Heitler dN/dE at fixed electron energy [arb].

    Formula (3BS from Koch–Motz, high-energy limit):

        dσ/dk = 4 α r_e² Z² · (1/k) · [ (4/3) − (4/3) y + y² ] · L_rad

    where ``y = k / E_e`` and ``L_rad ≈ ln(183 Z^-1/3)`` is the
    logarithmic radiator factor absorbed into the Z² prefactor here as a
    constant.  Returns dN/dE up to ``E_e``.
    """
    E_e = float(electron_energy_keV)
    E = np.asarray(energy_keV, dtype=float)
    y = np.clip(E / E_e, 0.0, 1.0)
    shape = (4.0 / 3.0) - (4.0 / 3.0) * y + y * y
    prefactor = 4.0 * FINE_STRUCTURE_ALPHA * CLASSICAL_ELECTRON_RADIUS_M**2 * _material_Z(material) ** 2
    dsigma_dk = prefactor * shape / np.maximum(E, 1e-9)
    # Enforce hard endpoint at E = E_e.
    return np.where(E <= E_e, dsigma_dk, 0.0)


def converter_photon_yield(
    electron_energy_keV: float,
    beam_charge_pC: float,
    material: str = "Ta",
    thickness_um: float = 100.0,
) -> float:
    """Rough total photon yield per shot (number of γ above 10 keV).

    Uses the approximate thick-target fraction ``N_γ ≈ N_e · t/X₀ · L_rad``
    with ``L_rad ≈ ln(183 Z^{-1/3})`` for the logarithmic factor.
    Thin-target regime; saturates near 1 radiation length, which is
    clamped here.
    """
    N_e = beam_charge_pC * 1e-12 / 1.602176634e-19
    X0_g = radiation_length_g_per_cm2(material)
    rho = _DENSITY_G_CM3[material]
    t_g_cm2 = rho * thickness_um * 1e-4  # μm → cm
    t_over_X0 = min(t_g_cm2 / X0_g, 1.0)  # clamp at 1 rad-len
    Z = _material_Z(material)
    L_rad = np.log(183.0 / Z ** (1.0 / 3.0))
    return float(N_e * t_over_X0 * L_rad)
