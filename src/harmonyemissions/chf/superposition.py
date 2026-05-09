"""Coherent-superposition kernel for chf3d.

Streams per-beam complex far-field amplitudes A_i(n) through

    E(r, n) = Σ_i w_i ε_i A_i(n) exp(i [φ_i + k_n n̂_i·(r−r_focus) + ω_n Δt_i])

on a 3-D focal-volume grid (or a single voxel — the "point" fast path).
The accumulator is the inner loop of ``surface_pipeline._run_multi_beam``
for the homogeneous (Phase C) case and of ``coherent_sum_heterogeneous``
for the Extreme-Power overlay.

Approximation: each beam contributes a *scalar* on-axis far-field
amplitude A_i(n) — interpolated at the centre pixel of its own 2-D
``far_amp`` stack — modulated by the propagation phase across the focal
voxels. The voxel-scale beam-amplitude curvature is dropped (the focal
volume is 1 μm-class and the beams are paraxial near the focus).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from harmonyemissions.chf.geometry import BeamArray

C_M_PER_S = 2.99792458e8


# ---------------------------------------------------------------------------
# Focal-volume grid.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FocalVolume:
    """Cube of voxels centred on ``origin_m`` with edge ``extent_m``."""

    n: int
    extent_m: float
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def voxel_size_m(self) -> float:
        return self.extent_m / max(self.n, 1)

    def coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.n <= 1:
            ox, oy, oz = self.origin_m
            return (
                np.array([ox], dtype=float),
                np.array([oy], dtype=float),
                np.array([oz], dtype=float),
            )
        axis = (np.arange(self.n) - self.n // 2) * self.voxel_size_m
        x = axis + self.origin_m[0]
        y = axis + self.origin_m[1]
        z = axis + self.origin_m[2]
        return x, y, z

    def grid_xyz(self) -> np.ndarray:
        """Return a (n, n, n, 3) array of voxel positions in metres."""
        x, y, z = self.coords()
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)


# ---------------------------------------------------------------------------
# Focal-volume accumulator.
# ---------------------------------------------------------------------------


def _on_axis_amplitude(far_amp: np.ndarray) -> complex:
    """Return the central pixel of a 2-D far-field amplitude array."""
    n = far_amp.shape[-1]
    return complex(far_amp[..., n // 2, n // 2])


def _build_phase_grid(
    volume: FocalVolume, n_hat: np.ndarray, k_n: float, r_focus: np.ndarray
) -> np.ndarray:
    """Return ``exp(i k_n n̂ · (r − r_focus))`` on the voxel grid."""
    grid = volume.grid_xyz() - r_focus  # (n, n, n, 3)
    phase = k_n * np.einsum("xyzc,c->xyz", grid, n_hat)
    return np.exp(1j * phase)


@dataclass
class FocalVolumeAccumulator:
    """Streams beams into a Jones-vector focal-volume cube.

    Indexing convention: the accumulator stores a ``(H, 3, n, n, n)``
    complex field where the first axis enumerates diagnostic harmonics
    and the second is the Cartesian Jones component.
    """

    volume: FocalVolume
    diag_harmonics: np.ndarray         # (H,)
    wavelength_m: float
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self._field: np.ndarray = np.zeros(
            (self.diag_harmonics.size, 3, self.volume.n, self.volume.n, self.volume.n),
            dtype=np.complex128,
        )
        self._rfocus = np.asarray(self.r_focus, dtype=float)

    # ----- streaming API -------------------------------------------------

    def add_beam(
        self,
        beam: BeamArray,
        beam_index: int,
        far_amp_per_diag: np.ndarray,
        weight: complex = 1.0,
    ) -> None:
        """Accumulate one driver's contribution to the focal-volume field.

        Parameters
        ----------
        beam
            Geometry struct (provides n̂_i, ε_i, φ_i, Δt_i, a0_scale[i]).
        beam_index
            Row index ``i`` into ``beam.directions`` etc.
        far_amp_per_diag
            ``(H, N, N)`` complex far-field amplitude at the diagnostic
            harmonics, in the beam's own frame.
        weight
            Optional global complex weight (default 1.0).
        """
        i = beam_index
        n_hat = beam.directions[i]
        eps = beam.polarization[i]
        phi_static = float(beam.relative_phase_rad[i])
        delay_fs = float(beam.relative_delay_fs[i])
        amp_scale = float(beam.a0_scale[i])

        for h_idx, n in enumerate(self.diag_harmonics):
            n = int(n)
            lam_n = self.wavelength_m / max(n, 1)
            k_n = 2.0 * np.pi / lam_n
            omega_n = 2.0 * np.pi * C_M_PER_S / lam_n
            # ω_n Δt with Δt in fs (×1e-15 → s)
            delay_phase = omega_n * delay_fs * 1e-15

            A_i = _on_axis_amplitude(far_amp_per_diag[h_idx])
            scalar = (
                weight
                * amp_scale
                * A_i
                * np.exp(1j * (phi_static + delay_phase))
            )

            phase_grid = _build_phase_grid(self.volume, n_hat, k_n, self._rfocus)
            contribution = scalar * phase_grid  # (n, n, n)
            # Distribute across the three Jones components.
            for c in range(3):
                self._field[h_idx, c] += eps[c] * contribution

    # ----- read-out ------------------------------------------------------

    def field(self) -> np.ndarray:
        """Return the accumulated (H, 3, n, n, n) complex Jones field."""
        return self._field

    def intensity(self) -> np.ndarray:
        """Return I(r, h) = Σ_c |E_c|² with shape (H, n, n, n)."""
        return np.sum(np.abs(self._field) ** 2, axis=1)

    def peak_intensity(self) -> np.ndarray:
        """Return max over voxels per harmonic, shape (H,)."""
        return self.intensity().reshape(self.diag_harmonics.size, -1).max(axis=1)


# ---------------------------------------------------------------------------
# Homogeneous coherent sum (Phase C).
# ---------------------------------------------------------------------------


def coherent_sum_homogeneous(
    beam: BeamArray,
    far_amp_per_beam: Sequence[np.ndarray],
    diag_harmonics: np.ndarray,
    wavelength_m: float,
    volume: FocalVolume,
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FocalVolumeAccumulator:
    """Sum N drivers (homogeneous, single-λ) into a focal volume.

    ``far_amp_per_beam[i]`` has shape ``(H, N, N)`` and is the complex
    diagnostic-harmonic far-field amplitude for beam ``i`` in its own
    frame. The accumulator output uses the shared harmonic axis directly.
    """
    if len(far_amp_per_beam) != beam.n_beams:
        raise ValueError(
            f"got {len(far_amp_per_beam)} per-beam far_amp arrays for n_beams={beam.n_beams}"
        )
    acc = FocalVolumeAccumulator(
        volume=volume,
        diag_harmonics=np.asarray(diag_harmonics, dtype=int),
        wavelength_m=wavelength_m,
        r_focus=r_focus,
    )
    for i in range(beam.n_beams):
        acc.add_beam(beam, i, far_amp_per_beam[i])
    return acc


# ---------------------------------------------------------------------------
# Heterogeneous frequency-axis resampling (Extreme-Power overlay).
# ---------------------------------------------------------------------------


def build_omega_grid(
    per_beam_omega_min: Sequence[float],
    per_beam_omega_max: Sequence[float],
    n_points: int = 4096,
    pad: float = 1.05,
) -> np.ndarray:
    """Build a log-spaced ω-grid spanning the union of per-beam ranges.

    Parameters
    ----------
    per_beam_omega_min
        Lowest physical angular frequency reached by each beam (rad/s).
    per_beam_omega_max
        Highest physical angular frequency reached by each beam (rad/s).
    n_points
        Number of grid points.
    pad
        Multiplicative padding around the union range.
    """
    omega_lo = float(min(per_beam_omega_min)) / pad
    omega_hi = float(max(per_beam_omega_max)) * pad
    if omega_lo <= 0 or omega_hi <= omega_lo:
        raise ValueError("invalid omega range")
    return np.geomspace(omega_lo, omega_hi, int(n_points))


def resample_to_omega(
    A_per_n: np.ndarray,
    omega_per_n: np.ndarray,
    omega_grid: np.ndarray,
) -> np.ndarray:
    """Resample a per-harmonic complex amplitude array onto a common ω-grid.

    ``A_per_n`` may be 1-D ``(n_harmonics,)`` or 2-D ``(n_harmonics, K)``.
    Interpolation is linear in log ω (separately for real and imaginary
    parts). Out-of-range points return zero.
    """
    omega = np.asarray(omega_per_n, dtype=float)
    grid = np.asarray(omega_grid, dtype=float)
    log_om = np.log(omega)
    log_grid = np.log(grid)
    A = np.asarray(A_per_n)
    if A.ndim == 1:
        re = np.interp(log_grid, log_om, A.real, left=0.0, right=0.0)
        im = np.interp(log_grid, log_om, A.imag, left=0.0, right=0.0)
        return (re + 1j * im).astype(np.complex128)
    if A.ndim == 2:
        out = np.zeros((grid.size, A.shape[1]), dtype=np.complex128)
        for k in range(A.shape[1]):
            out[:, k] = (
                np.interp(log_grid, log_om, A[:, k].real, left=0.0, right=0.0)
                + 1j * np.interp(log_grid, log_om, A[:, k].imag, left=0.0, right=0.0)
            )
        return out
    raise ValueError(f"unsupported A_per_n.ndim={A.ndim}")


def coherent_sum_heterogeneous(
    beam: BeamArray,
    on_axis_amp_per_omega: Sequence[np.ndarray],
    omega_grid: np.ndarray,
    volume: FocalVolume,
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sum heterogeneous-λ drivers on a common physical-frequency axis.

    Parameters
    ----------
    beam
        Geometry struct (used for n̂_i, ε_i, φ_i, Δt_i, a0_scale[i]).
    on_axis_amp_per_omega
        ``len(...) == n_beams``; each entry is a complex ``(M,)`` array
        of on-axis amplitudes already resampled onto ``omega_grid``.
    omega_grid
        ``(M,)`` physical angular frequencies in rad/s.

    Returns
    -------
    spectrum_omega
        Coherent-sum on-axis intensity ``(M,)`` at the focal-volume
        centre (Σ_c |Σ_i …|²).
    focal_volume_intensity
        ``(M, n, n, n)`` intensity Σ_c |E_c|² (one cube per ω). For
        memory parity with the homogeneous path, callers may downselect
        to a smaller diagnostic-ω subset before invoking this — see
        ``coherent_sum_heterogeneous_diag`` below.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    M = omega_grid.size
    rfocus = np.asarray(r_focus, dtype=float)
    field = np.zeros((M, 3, volume.n, volume.n, volume.n), dtype=np.complex128)
    for i in range(beam.n_beams):
        n_hat = beam.directions[i]
        eps = beam.polarization[i]
        phi_static = float(beam.relative_phase_rad[i])
        delay_fs = float(beam.relative_delay_fs[i])
        amp_scale = float(beam.a0_scale[i])
        A_om = np.asarray(on_axis_amp_per_omega[i], dtype=np.complex128)
        if A_om.shape[0] != M:
            raise ValueError(
                f"beam {i} amplitude shape {A_om.shape} != omega_grid size {M}"
            )
        for m, omega in enumerate(omega_grid):
            k_m = omega / C_M_PER_S
            phase_grid = _build_phase_grid(volume, n_hat, k_m, rfocus)
            delay_phase = omega * delay_fs * 1e-15
            scalar = (
                amp_scale
                * A_om[m]
                * np.exp(1j * (phi_static + delay_phase))
            )
            for c in range(3):
                field[m, c] += eps[c] * scalar * phase_grid
    intensity_cube = np.sum(np.abs(field) ** 2, axis=1)  # (M, n, n, n)
    centre = volume.n // 2
    spectrum_omega = intensity_cube[:, centre, centre, centre]
    return spectrum_omega, intensity_cube


def coherent_sum_heterogeneous_diag(
    beam: BeamArray,
    on_axis_amp_per_omega: Sequence[np.ndarray],
    omega_grid: np.ndarray,
    diag_omega_indices: Sequence[int],
    volume: FocalVolume,
    r_focus: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`coherent_sum_heterogeneous`, but only build cubes at
    selected diagnostic ω-indices. Returns ``(spectrum_omega_full,
    focal_volume_intensity_diag)`` so the full ω-spectrum is preserved
    (for plotting / detector dispatch) while the volumetric output
    stays memory-bounded."""
    omega_grid = np.asarray(omega_grid, dtype=float)
    diag = np.asarray(list(diag_omega_indices), dtype=int)
    full_centre_intensity = np.zeros(omega_grid.size, dtype=float)
    rfocus = np.asarray(r_focus, dtype=float)
    diag_field = np.zeros(
        (diag.size, 3, volume.n, volume.n, volume.n), dtype=np.complex128
    )
    centre_field = np.zeros((omega_grid.size, 3), dtype=np.complex128)
    for i in range(beam.n_beams):
        n_hat = beam.directions[i]
        eps = beam.polarization[i]
        phi_static = float(beam.relative_phase_rad[i])
        delay_fs = float(beam.relative_delay_fs[i])
        amp_scale = float(beam.a0_scale[i])
        A_om = np.asarray(on_axis_amp_per_omega[i], dtype=np.complex128)
        for m, omega in enumerate(omega_grid):
            scalar_static = (
                amp_scale
                * A_om[m]
                * np.exp(1j * (phi_static + omega * delay_fs * 1e-15))
            )
            # Centre-pixel propagation phase at r_focus is unity, so the
            # centre contribution is just scalar_static · ε.
            centre_field[m] += eps * scalar_static
        for d_idx, m in enumerate(diag):
            omega = omega_grid[int(m)]
            k_m = omega / C_M_PER_S
            phase_grid = _build_phase_grid(volume, n_hat, k_m, rfocus)
            scalar = (
                amp_scale
                * A_om[int(m)]
                * np.exp(1j * (phi_static + omega * delay_fs * 1e-15))
            )
            for c in range(3):
                diag_field[d_idx, c] += eps[c] * scalar * phase_grid
    full_centre_intensity = np.sum(np.abs(centre_field) ** 2, axis=1)
    diag_intensity_cube = np.sum(np.abs(diag_field) ** 2, axis=1)
    return full_centre_intensity, diag_intensity_cube
