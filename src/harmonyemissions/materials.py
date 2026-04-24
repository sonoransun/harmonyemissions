"""Target-side atomic data for K-shell emission.

Used by the K-α model; instrument-side attenuation tables (Al filter,
Be window, Si CCD, etc.) live in ``detector/filters.py`` and are kept
separate because their role is different and their data sources
(Henke / NIST XCOM) have different provenance.

K-shell line energies and fluorescence yields are from Krause (1979)
/ Thompson & Vaughan X-ray Data Booklet; fluorescence yields ω_K are
the Bambynek review values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialData:
    """K-shell line data for one target material."""

    name: str
    Z: int
    K_edge_keV: float
    K_alpha1_keV: float
    K_alpha2_keV: float
    K_beta_keV: float
    fluorescence_yield_K: float
    L_edge_keV: float | None = None


MATERIAL_REGISTRY: dict[str, MaterialData] = {
    "Al": MaterialData("Al", 13, 1.5596, 1.4867, 1.4863, 1.5575, 0.039, 0.0728),
    "Si": MaterialData("Si", 14, 1.8389, 1.7400, 1.7394, 1.8359, 0.050, 0.0997),
    # SiO2 is a compound; the K-α emission is effectively silicon's.
    "SiO2": MaterialData("SiO2", 14, 1.8389, 1.7400, 1.7394, 1.8359, 0.050, 0.0997),
    "Ti": MaterialData("Ti", 22, 4.9664, 4.5108, 4.5049, 4.9318, 0.219, 0.4555),
    "Fe": MaterialData("Fe", 26, 7.1120, 6.4038, 6.3908, 7.0580, 0.350, 0.7088),
    "Cu": MaterialData("Cu", 29, 8.9789, 8.0478, 8.0278, 8.9053, 0.440, 0.9327),
    "Mo": MaterialData("Mo", 42, 20.000, 17.4793, 17.3743, 19.6083, 0.770, 2.5202),
    "Ag": MaterialData("Ag", 47, 25.5140, 22.1629, 21.9903, 24.9424, 0.833, 3.3511),
}


def lookup(material: str) -> MaterialData:
    """Case-insensitive lookup against :data:`MATERIAL_REGISTRY`."""
    if material in MATERIAL_REGISTRY:
        return MATERIAL_REGISTRY[material]
    for key, val in MATERIAL_REGISTRY.items():
        if key.lower() == material.lower():
            return val
    raise ValueError(
        f"No material data for {material!r}. Known: {sorted(MATERIAL_REGISTRY)}"
    )
