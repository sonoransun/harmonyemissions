"""Beam-array geometry materialisation for the chf3d Phase C kernel.

Turns a :class:`harmonyemissions.config.LaserArrayConfig` into a frozen
:class:`BeamArray` carrying everything the multi-beam pipeline needs:

- ``directions``  — inward-pointing unit vectors n̂_i (one per driver),
- ``positions``   — beam launch points r_i on a sphere of radius
  ``focal_radius_m`` centred on the focus,
- ``polarization``— complex Jones vectors ε_i (orthogonal to n̂_i),
- merged user-supplied ``relative_phase_rad`` / ``relative_delay_fs`` /
  ``per_beam_a0_scale`` (defaulted to zero / unity when absent).

The Platonic face/vertex tables are this module's canonical source of
truth — the integer counts in :mod:`harmonyemissions.config` are only used
for schema arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from harmonyemissions.config import LaserArrayConfig

PHI = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio


# ---------------------------------------------------------------------------
# Platonic face & vertex tables.
# ---------------------------------------------------------------------------


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return v / norms


def _tetrahedron_vertices() -> np.ndarray:
    # Four vertices of a regular tetrahedron inscribed in a cube.
    v = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=float,
    )
    return _normalize_rows(v)


def _tetrahedron_faces() -> np.ndarray:
    # The four faces of a regular tetrahedron point opposite to its vertices.
    return -_tetrahedron_vertices()


def _cube_vertices() -> np.ndarray:
    v = np.array(
        [[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)],
        dtype=float,
    )
    return _normalize_rows(v)


def _cube_faces() -> np.ndarray:
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )


def _octahedron_vertices() -> np.ndarray:
    return _cube_faces()  # duals: octahedron's vertices = cube's face centres


def _octahedron_faces() -> np.ndarray:
    # Eight face centres of a regular octahedron lie at the vertices of a cube.
    return _cube_vertices()


def _icosahedron_vertices() -> np.ndarray:
    v = np.array(
        [
            [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
            [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
            [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1],
        ],
        dtype=float,
    )
    return _normalize_rows(v)


def _icosahedron_faces() -> np.ndarray:
    # Sum the three vertices of each triangular face.  The 20 face centres
    # are the vertices of the dodecahedron — built directly here as a
    # closed-form set (golden-ratio coordinates of the regular dodecahedron).
    a = 1.0
    b = 1.0 / PHI
    c = PHI
    v = np.array(
        # 8 cube vertices × (±1, ±1, ±1)
        [[s1 * a, s2 * a, s3 * a]
         for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
        # 4 in (0, ±1/φ, ±φ)
        + [[0, s2 * b, s3 * c] for s2 in (-1, 1) for s3 in (-1, 1)]
        # 4 in (±1/φ, ±φ, 0)
        + [[s1 * b, s2 * c, 0] for s1 in (-1, 1) for s2 in (-1, 1)]
        # 4 in (±φ, 0, ±1/φ)
        + [[s1 * c, 0, s3 * b] for s1 in (-1, 1) for s3 in (-1, 1)],
        dtype=float,
    )
    return _normalize_rows(v)


def _dodecahedron_vertices() -> np.ndarray:
    # Same closed-form set as icosahedron_faces (the two are duals).
    return _icosahedron_faces()


def _dodecahedron_faces() -> np.ndarray:
    # Dual of the icosahedron's faces ⇒ icosahedron's vertex set.
    return _icosahedron_vertices()


_PLATONIC_TABLE = {
    ("tetrahedral",  "faces"):    _tetrahedron_faces,
    ("tetrahedral",  "vertices"): _tetrahedron_vertices,
    ("cubic",        "faces"):    _cube_faces,
    ("cubic",        "vertices"): _cube_vertices,
    ("octahedral",   "faces"):    _octahedron_faces,
    ("octahedral",   "vertices"): _octahedron_vertices,
    ("dodecahedral", "faces"):    _dodecahedron_faces,
    ("dodecahedral", "vertices"): _dodecahedron_vertices,
    ("icosahedral",  "faces"):    _icosahedron_faces,
    ("icosahedral",  "vertices"): _icosahedron_vertices,
}


# ---------------------------------------------------------------------------
# Generic distributions.
# ---------------------------------------------------------------------------


def _ring(n: int, normal: tuple[float, float, float] = (0, 0, 1)) -> np.ndarray:
    """``n`` points equispaced on a great circle perpendicular to ``normal``."""
    if n <= 0:
        raise ValueError("n must be > 0 for a ring")
    nz = np.array(normal, dtype=float)
    nz = nz / max(np.linalg.norm(nz), 1e-30)
    # Pick any unit vector orthogonal to nz.
    helper = np.array([1.0, 0.0, 0.0]) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(nz, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(nz, e1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack(
        [np.cos(a) * e1 + np.sin(a) * e2 for a in angles], axis=0
    )


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` points quasi-uniformly distributed on the unit sphere."""
    if n <= 0:
        raise ValueError("n must be > 0 for a fibonacci sphere")
    indices = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * indices  # golden-angle stride
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Polarization resolution.
# ---------------------------------------------------------------------------


def _resolve_polarization(
    mode: str,
    directions: np.ndarray,
    explicit: list[tuple[float, float, float]] | None,
) -> np.ndarray:
    """Return complex Jones vectors ε_i (each orthogonal to n̂_i)."""
    n = directions.shape[0]
    if mode == "explicit":
        if explicit is None or len(explicit) != n:
            raise ValueError("explicit polarization needs len == n_beams")
        eps = np.asarray(explicit, dtype=complex)
        return eps

    # Build an in-plane radial axis (project x̂ into the plane normal to n̂).
    # Falls back to ŷ when n̂ is parallel to x̂.
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])

    eps = np.zeros((n, 3), dtype=complex)
    for i, n_hat in enumerate(directions):
        helper = x_hat if abs(np.dot(n_hat, x_hat)) < 0.9 else y_hat
        radial = helper - np.dot(helper, n_hat) * n_hat
        radial = radial / max(np.linalg.norm(radial), 1e-30)
        azimuthal = np.cross(n_hat, radial)
        azimuthal = azimuthal / max(np.linalg.norm(azimuthal), 1e-30)
        if mode in ("uniform_p", "radial"):
            eps[i] = radial.astype(complex)
        elif mode in ("uniform_s", "azimuthal"):
            eps[i] = azimuthal.astype(complex)
        elif mode == "circular_alternating":
            handedness = 1.0 if (i % 2 == 0) else -1.0
            eps[i] = (radial + 1j * handedness * azimuthal) / np.sqrt(2.0)
        else:
            raise ValueError(f"unknown polarization mode {mode!r}")
    return eps


# ---------------------------------------------------------------------------
# BeamArray dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamArray:
    """Materialised multi-driver geometry for the chf3d coherent kernel."""

    n_beams: int
    geometry: str
    placement: str
    polarization_mode: str
    structured_mode: str | None
    structured_mode_params: dict[str, Any] | None
    focal_radius_m: float
    directions: np.ndarray            # (N, 3) inward unit vectors
    positions: np.ndarray             # (N, 3) launch points (m)
    polarization: np.ndarray          # (N, 3) complex Jones vectors
    relative_phase_rad: np.ndarray    # (N,)
    relative_delay_fs: np.ndarray     # (N,)
    a0_scale: np.ndarray              # (N,) per-beam amplitude scale (default 1)


def build_beam_array(
    cfg: LaserArrayConfig, focal_radius_m: float = 0.01
) -> BeamArray:
    """Resolve a config into a frozen :class:`BeamArray`."""
    geom = cfg.geometry
    if geom in ("tetrahedral", "cubic", "octahedral", "dodecahedral", "icosahedral"):
        dirs = _PLATONIC_TABLE[(geom, cfg.placement)]()
    elif geom == "ring":
        if cfg.directions is not None:
            dirs = np.asarray(cfg.directions, dtype=float)
        else:
            assert cfg.n_beams is not None  # validator guarantee
            dirs = _ring(cfg.n_beams)
    elif geom == "fibonacci_sphere":
        if cfg.directions is not None:
            dirs = np.asarray(cfg.directions, dtype=float)
        else:
            assert cfg.n_beams is not None
            dirs = _fibonacci_sphere(cfg.n_beams)
    elif geom == "explicit":
        assert cfg.directions is not None  # validator guarantee
        dirs = np.asarray(cfg.directions, dtype=float)
    else:
        raise ValueError(f"unknown geometry {geom!r}")

    dirs = _normalize_rows(np.asarray(dirs, dtype=float))
    n = dirs.shape[0]

    # Beam launch points: place each driver on the launch sphere, opposite
    # the inward direction n̂_i (so r_i = -focal_radius_m * n̂_i and the
    # beam propagates *toward* the origin).
    positions = -focal_radius_m * dirs

    polarization = _resolve_polarization(
        cfg.polarization_mode, dirs, cfg.polarization_vectors
    )

    relative_phase = (
        np.asarray(cfg.relative_phase_rad, dtype=float)
        if cfg.relative_phase_rad is not None
        else np.zeros(n, dtype=float)
    )
    relative_delay = (
        np.asarray(cfg.relative_delay_fs, dtype=float)
        if cfg.relative_delay_fs is not None
        else np.zeros(n, dtype=float)
    )
    a0_scale = (
        np.asarray(cfg.per_beam_a0_scale, dtype=float)
        if cfg.per_beam_a0_scale is not None
        else np.ones(n, dtype=float)
    )

    return BeamArray(
        n_beams=n,
        geometry=geom,
        placement=cfg.placement,
        polarization_mode=cfg.polarization_mode,
        structured_mode=cfg.structured_mode,
        structured_mode_params=dict(cfg.structured_mode_params)
            if cfg.structured_mode_params is not None
            else None,
        focal_radius_m=float(focal_radius_m),
        directions=dirs,
        positions=positions,
        polarization=polarization,
        relative_phase_rad=relative_phase,
        relative_delay_fs=relative_delay,
        a0_scale=a0_scale,
    )


# ---------------------------------------------------------------------------
# JSON-round-trippable record (used in Result.beam_array_geometry).
# ---------------------------------------------------------------------------


def to_record(arr: BeamArray) -> dict[str, Any]:
    """Lossless JSON-serialisable form of a :class:`BeamArray`."""
    return {
        "n_beams": int(arr.n_beams),
        "geometry": arr.geometry,
        "placement": arr.placement,
        "polarization_mode": arr.polarization_mode,
        "structured_mode": arr.structured_mode,
        "structured_mode_params": arr.structured_mode_params,
        "focal_radius_m": float(arr.focal_radius_m),
        "directions": arr.directions.tolist(),
        "positions": arr.positions.tolist(),
        "polarization_real": arr.polarization.real.tolist(),
        "polarization_imag": arr.polarization.imag.tolist(),
        "relative_phase_rad": arr.relative_phase_rad.tolist(),
        "relative_delay_fs": arr.relative_delay_fs.tolist(),
        "a0_scale": arr.a0_scale.tolist(),
    }


def from_record(rec: dict[str, Any]) -> BeamArray:
    pol = (np.asarray(rec["polarization_real"], dtype=float)
           + 1j * np.asarray(rec["polarization_imag"], dtype=float))
    return BeamArray(
        n_beams=int(rec["n_beams"]),
        geometry=str(rec["geometry"]),
        placement=str(rec["placement"]),
        polarization_mode=str(rec["polarization_mode"]),
        structured_mode=rec.get("structured_mode"),
        structured_mode_params=rec.get("structured_mode_params"),
        focal_radius_m=float(rec["focal_radius_m"]),
        directions=np.asarray(rec["directions"], dtype=float),
        positions=np.asarray(rec["positions"], dtype=float),
        polarization=pol,
        relative_phase_rad=np.asarray(rec["relative_phase_rad"], dtype=float),
        relative_delay_fs=np.asarray(rec["relative_delay_fs"], dtype=float),
        a0_scale=np.asarray(rec["a0_scale"], dtype=float),
    )


def with_phases(arr: BeamArray, phases: np.ndarray) -> BeamArray:
    """Return a copy of ``arr`` with ``relative_phase_rad`` replaced."""
    phases = np.asarray(phases, dtype=float)
    if phases.shape != (arr.n_beams,):
        raise ValueError(
            f"phases shape {phases.shape} != (n_beams,) = ({arr.n_beams},)"
        )
    return replace(arr, relative_phase_rad=phases)


def with_delays(arr: BeamArray, delays_fs: np.ndarray) -> BeamArray:
    """Return a copy of ``arr`` with ``relative_delay_fs`` replaced."""
    delays_fs = np.asarray(delays_fs, dtype=float)
    if delays_fs.shape != (arr.n_beams,):
        raise ValueError(
            f"delays shape {delays_fs.shape} != (n_beams,) = ({arr.n_beams},)"
        )
    return replace(arr, relative_delay_fs=delays_fs)
