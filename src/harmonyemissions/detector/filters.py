"""Shared X-ray attenuation library for the detector pipeline.

Data
----
Tables live as JSON under ``detector/data/attenuation/*.json``.  Each
table holds ~80 log-spaced (E, μ/ρ) points in [30 eV, 10 MeV] derived
from Henke (10 eV–30 keV) and NIST XCOM (1 keV–100 GeV).  Between
tabulated points we linearly interpolate in log-log; outside the
tabulated range we extrapolate with μ/ρ ∝ E⁻³ (photoelectric low-E
side) and a floor-clip to 0.01 cm²/g on the high-E side.

K-edge (and L₁/L₂/L₃-edge where relevant) energies are stored in the
material metadata and consulted by the hard-X-ray Ross-pair module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).with_name("data") / "attenuation"


@dataclass(frozen=True)
class FilterMaterial:
    name: str
    Z_eff: float
    density_g_cm3: float
    k_edges_ev: dict[str, float]
    energy_ev: np.ndarray
    mu_over_rho_cm2_g: np.ndarray


@dataclass(frozen=True)
class FilterSpec:
    """Declarative filter: a named material and a thickness in μm."""

    material: str
    thickness_um: float


@lru_cache(maxsize=32)
def load_material(name: str) -> FilterMaterial:
    """Load a cached FilterMaterial by short name (e.g. 'cu', 'al', 'mylar')."""
    key = name.lower()
    path = _DATA_DIR / f"{key}.json"
    if not path.exists():
        known = sorted(p.stem for p in _DATA_DIR.glob("*.json"))
        raise ValueError(
            f"No attenuation table for material {name!r}. Known: {known}"
        )
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return FilterMaterial(
        name=raw["name"],
        Z_eff=float(raw["Z_eff"]),
        density_g_cm3=float(raw["density_g_cm3"]),
        k_edges_ev={k: float(v) for k, v in raw.get("edges_ev", {}).items()},
        energy_ev=np.asarray(raw["energy_ev"], dtype=float),
        mu_over_rho_cm2_g=np.asarray(raw["mu_over_rho_cm2_g"], dtype=float),
    )


def _mu_over_rho(material: FilterMaterial, energy_ev: np.ndarray) -> np.ndarray:
    E = np.asarray(energy_ev, dtype=float)
    E_ref = material.energy_ev
    mu_ref = material.mu_over_rho_cm2_g
    # Log-log interpolation for values inside the tabulated range.
    log_E = np.log(np.clip(E, 1.0, None))
    log_E_ref = np.log(E_ref)
    log_mu_ref = np.log(mu_ref)
    log_mu = np.interp(log_E, log_E_ref, log_mu_ref, left=np.nan, right=np.nan)
    mu = np.exp(log_mu)
    # Low-E extrapolation: E⁻³ photoelectric.
    lo = E < E_ref[0]
    if lo.any():
        mu[lo] = mu_ref[0] * (E[lo] / E_ref[0]) ** -3.0
    # High-E: clip-to-minimum (Compton tail).
    hi = E > E_ref[-1]
    if hi.any():
        mu[hi] = max(mu_ref[-1], 0.01)
    return mu


def transmission(
    material: str | FilterMaterial,
    energy_ev: np.ndarray,
    thickness_um: float,
) -> np.ndarray:
    """T(E) = exp(−μ/ρ · ρ · x) for a filter of thickness ``thickness_um`` μm."""
    m = material if isinstance(material, FilterMaterial) else load_material(material)
    mu = _mu_over_rho(m, energy_ev)
    # Convert thickness μm → cm.
    x_cm = thickness_um * 1e-4
    tau = mu * m.density_g_cm3 * x_cm
    return np.exp(-tau)


def list_materials() -> list[str]:
    """Return short names of every packaged attenuation table."""
    return sorted(p.stem for p in _DATA_DIR.glob("*.json"))
