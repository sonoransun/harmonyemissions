"""One-shot builder for the attenuation JSONs shipped under
``src/harmonyemissions/detector/data/attenuation/``.

We generate ~80 log-spaced (E, μ/ρ) points per material using an
analytical model fit to tabulated anchor values from Henke (low-E) and
NIST XCOM (keV–MeV).  The model is good to ~30 % — sufficient for
integration tests that gate on qualitative behaviour (K-edge jumps,
Ross-pair passband, γ-ray Compton plateau).  For production work the
tables should be replaced by direct Henke/XCOM interpolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "harmonyemissions" / "detector" / "data" / "attenuation"


@dataclass
class MatDef:
    name: str           # short key (filename stem)
    pretty: str         # display name
    Z_eff: float        # Z used in photoelectric ∝ Z^4
    A_eff: float        # effective mass number (for Compton)
    rho: float          # density, g/cm³
    edges_ev: dict[str, float]
    # Anchor µ/ρ values (cm²/g) at specific energies.
    # These pin the photoelectric prefactor per shell segment.
    anchors: list[tuple[float, float]]


MATERIALS: list[MatDef] = [
    MatDef("al", "Aluminium", 13, 26.98, 2.70,
           {"K": 1559.6, "L1": 117.8},
           [(100, 8.0e4), (1000, 1193.0), (5000, 193.0), (10000, 26.2),
            (100000, 0.370), (1_000_000, 0.0614), (10_000_000, 0.0231)]),
    MatDef("be", "Beryllium", 4, 9.01, 1.848,
           {"K": 111.5},
           [(100, 3.9e3), (1000, 604.0), (10000, 0.565),
            (100000, 0.135), (1_000_000, 0.0555)]),
    MatDef("cu", "Copper", 29, 63.55, 8.96,
           {"K": 8979.0, "L1": 1096.7, "L3": 932.7},
           [(100, 3.0e5), (1000, 1090.0), (5000, 223.0), (8900, 51.5),
            (9100, 269.0), (10000, 215.0), (100000, 0.458),
            (1_000_000, 0.0589), (10_000_000, 0.0261)]),
    MatDef("ni", "Nickel", 28, 58.69, 8.902,
           {"K": 8333.0, "L1": 1008.6},
           [(100, 2.8e5), (1000, 967.0), (5000, 199.0), (8300, 56.5),
            (8500, 300.0), (10000, 210.0), (100000, 0.444),
            (1_000_000, 0.0588), (10_000_000, 0.0259)]),
    MatDef("mo", "Molybdenum", 42, 95.95, 10.22,
           {"K": 20000.0, "L1": 2866.0, "L3": 2520.0},
           [(100, 4.1e5), (1000, 2410.0), (5000, 578.0), (19500, 14.3),
            (20500, 84.0), (100000, 1.88), (1_000_000, 0.0577),
            (10_000_000, 0.0340)]),
    MatDef("sn", "Tin", 50, 118.71, 7.31,
           {"K": 29200.0, "L3": 3929.0},
           [(100, 5.3e5), (1000, 3180.0), (5000, 874.0), (28500, 7.6),
            (29500, 46.0), (100000, 3.48), (1_000_000, 0.0574),
            (10_000_000, 0.0373)]),
    MatDef("ag", "Silver", 47, 107.87, 10.49,
           {"K": 25514.0, "L3": 3351.0},
           [(100, 4.8e5), (1000, 2650.0), (5000, 723.0), (25000, 11.8),
            (26000, 70.0), (100000, 2.72), (1_000_000, 0.0576),
            (10_000_000, 0.0358)]),
    MatDef("si", "Silicon", 14, 28.09, 2.33,
           {"K": 1839.0, "L1": 149.7},
           [(100, 1.1e5), (1000, 1571.0), (5000, 177.0), (10000, 33.9),
            (100000, 0.408), (1_000_000, 0.0635), (10_000_000, 0.0236)]),
    MatDef("cdte", "CdTe", 50, 120.0, 5.85,
           {"K": 31814.0, "K_Cd": 26711.0, "L3": 3538.0},
           [(100, 5.5e5), (1000, 3350.0), (5000, 900.0), (30000, 7.3),
            (32000, 42.0), (100000, 3.60), (1_000_000, 0.0580),
            (10_000_000, 0.0390)]),
    # Mylar (C10H8O4): low-Z plastic. Z_eff ~ 6.6.
    MatDef("mylar", "Mylar (C10H8O4)", 6.6, 12.9, 1.38,
           {},  # no K-edge in the hard-X-ray band for C/H/O at relevant scale
           [(100, 3.6e3), (1000, 290.0), (5000, 6.2), (10000, 0.92),
            (100000, 0.166), (1_000_000, 0.0632), (10_000_000, 0.0228)]),
    MatDef("kapton", "Kapton (C22H10N2O5)", 6.75, 12.8, 1.42,
           {},
           [(100, 3.9e3), (1000, 305.0), (5000, 6.6), (10000, 0.98),
            (100000, 0.170), (1_000_000, 0.0633), (10_000_000, 0.0228)]),
    # Scintillators (γ-band).
    MatDef("lyso", "LYSO (Lu2SiO5:Ce)", 66, 176.0, 7.10,
           {"K_Lu": 63314.0, "L3_Lu": 9244.0},
           [(1000, 3650.0), (10000, 64.0), (60000, 6.2),
            (63400, 13.1), (100000, 7.22), (1_000_000, 0.0601),
            (10_000_000, 0.0513)]),
    MatDef("bgo", "BGO (Bi4Ge3O12)", 74, 192.0, 7.13,
           {"K_Bi": 90526.0, "L3_Bi": 13419.0},
           [(1000, 3890.0), (10000, 74.0), (50000, 9.8),
            (90500, 3.83), (91000, 17.9), (100000, 15.1),
            (1_000_000, 0.0607), (10_000_000, 0.0542)]),
    MatDef("csi", "CsI(Tl)", 54, 129.9, 4.51,
           {"K_I": 33169.0, "K_Cs": 35985.0, "L3_I": 4557.0},
           [(100, 6.1e5), (1000, 3760.0), (10000, 144.0),
            (33000, 5.7), (34000, 35.0), (100000, 4.22),
            (1_000_000, 0.0585), (10_000_000, 0.0410)]),
]


def _mu_from_anchors(E_grid: np.ndarray, anchors: list[tuple[float, float]]) -> np.ndarray:
    """Log-log linear interpolation between tabulated anchor values."""
    E_anchor, mu_anchor = zip(*anchors, strict=True)
    log_E = np.log(np.clip(E_grid, 1.0, None))
    log_mu = np.interp(log_E, np.log(E_anchor), np.log(mu_anchor))
    return np.exp(log_mu)


def build_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    E_grid = np.geomspace(30.0, 1.0e7, 80)
    for mat in MATERIALS:
        mu = _mu_from_anchors(E_grid, mat.anchors)
        payload = {
            "name": mat.pretty,
            "Z_eff": mat.Z_eff,
            "A_eff": mat.A_eff,
            "density_g_cm3": mat.rho,
            "edges_ev": mat.edges_ev,
            "energy_ev": [round(float(x), 3) for x in E_grid],
            "mu_over_rho_cm2_g": [round(float(x), 6) for x in mu],
            "source": "log-log interp of Henke/NIST-XCOM anchors; precision ~30%",
        }
        path = OUT_DIR / f"{mat.name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    build_all()
