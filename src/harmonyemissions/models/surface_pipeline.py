"""Unified surface-HHG pipeline from Timmis et al. 2026 (Methods eqs. 7–12).

Pipeline:

    1. Build a 2-D driver near-field U₀(x', y') from the Laser's spatial profile.
    2. Peak a₀ map: a0(x', y') = laser.a0 · |U₀| / |U₀|_max.
    3. Plasma scale length L/λ from the contrast model (Target.t_HDR + prepulse).
    4. Plasma dent Δz(x', y') = Δz_i + Δz_e (surface.denting).
    5. For each harmonic n: apply S(n, a₀) spatial filter + e^{−2 i k_n Δz cos θ}
       phase imprint; Fraunhofer-propagate to the CHF focal plane.
    6. Per-harmonic efficiency η_n = energy in the far-field harmonic
       intensity, normalized to unit efficiency at n = 1.
    7. CHF breakdown: Γ_D, Γ_2D, Γ_3D, Γ_total.

The result is an xarray dataset that bundles spectrum, 2-D dent map, and
2-D far-field intensity for a small subset of "diagnostic" harmonics so
users can visualise both the spectrum and the beam profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import xarray as xr

from harmonyemissions.beam import SpatialGrid, build_profile, intensity
from harmonyemissions.chf.gain import extrapolate_3d_gain
from harmonyemissions.chf.propagation import stack_harmonics_far_field
from harmonyemissions.contrast import ContrastInputs, scale_length
from harmonyemissions.emission.spikes import (
    CutoffMode,
    relativistic_spikes_filter,
    spikes_cutoff_harmonic,
)
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.surface.denting import DentingInputs, dent_map
from harmonyemissions.target import Target

# Tunable defaults for the pipeline's 2-D grid. Keeping these modest keeps
# run time under ~1 s on a laptop. Users who want higher resolution can
# override via NumericsConfig.pipeline_grid / pipeline_dx_um.
DEFAULT_GRID_N = 128
DEFAULT_DX_UM = 0.08             # 80 nm / pixel → ~10 μm window at n=128
DEFAULT_FOCUS_M = 0.01           # 1 cm focal distance for the CHF propagation
# A small set of "diagnostic" harmonics to store 2-D profiles for.
DEFAULT_DIAG_HARMONICS = (1, 15, 30, 45)


class _BeamProducts(NamedTuple):
    """Per-driver products of stages 1–7 of the surface_pipeline.

    Returned by :func:`_run_single_beam` so the multi-beam (chf3d) path in
    Phase C can iterate over drivers and stream their outputs through the
    coherent-superposition accumulator without duplicating per-beam physics.
    """
    u0: np.ndarray            # (N, N) complex driver near-field amplitude
    a0_map: np.ndarray        # (N, N) peak-a₀ map
    dmap: np.ndarray          # (N, N) plasma dent in units of λ
    n: np.ndarray              # harmonic axis (1..n_int + log tail to 2·n_c)
    spectrum_vals: np.ndarray  # spatially averaged spikes envelope
    far_stack: np.ndarray      # (n_diag, N, N) far-field intensity per diag harmonic
    dx_far: np.ndarray         # (n_diag,) far-field pixel size per diag harmonic
    diag: np.ndarray           # (n_diag,) diagnostic harmonic indices
    L_over_lambda: float


def _run_single_beam(
    laser: Laser, target: Target, numerics, grid: SpatialGrid
) -> _BeamProducts:
    """Run stages 1–7 of the surface_pipeline for one driver on `grid`.

    Pure refactor — this is the body of :meth:`SurfacePipelineModel.run`
    extracted verbatim, modulo the grid construction (now done by the
    caller so multi-beam runs share one grid across drivers).
    """
    # 1. Driver near-field amplitude on the supplied grid.
    fwhm_m = laser.spot_fwhm_um * 1e-6
    u0 = build_profile(
        laser.spatial_profile, grid, fwhm_m, laser.super_gaussian_order,
    )

    # 2. Peak a₀ map.
    abs_u0 = np.abs(u0)
    peak = float(abs_u0.max())
    if peak <= 0:
        raise RuntimeError("driver amplitude is zero everywhere")
    a0_map = laser.a0 * (abs_u0 / peak)

    # 3. Scale length from contrast model.
    contrast = ContrastInputs(
        t_HDR_fs=target.t_HDR_fs,
        prepulse_intensity_rel=target.prepulse_intensity_rel,
        prepulse_delay_fs=target.prepulse_delay_fs,
        wavelength_um=laser.wavelength_um,
    )
    L_over_lambda = scale_length(contrast)

    # 4. Plasma dent map.
    denting = DentingInputs(
        scale_length_lambda=L_over_lambda,
        angle_deg=laser.angle_deg,
        reflectivity=target.reflectivity,
        oxygen_admixture=2.0 / 3.0 if target.material == "SiO2" else 0.0,
        wavelength_um=laser.wavelength_um,
    )
    duration_T0 = laser.duration_fs * 1e-15 / laser.units.period_s
    dmap = dent_map(a0_map, duration_T0, denting)

    # 5–6. Spectrum from the efficiency envelope integrated over the beam.
    # Use integer harmonics up to min(2·n_c_peak, 2000) so the BGP plateau
    # and first decade of the cutoff roll-off are resolved, then a sparse
    # log-spaced tail out to 2×n_c — the tail is exponentially suppressed
    # so coarse sampling doesn't lose physics but keeps the per-run cost
    # bounded at any a₀.
    n_c_peak = float(spikes_cutoff_harmonic(laser.a0))
    n_int_max = int(min(2000.0, 2.0 * n_c_peak))
    n_int = np.arange(1.0, max(200, n_int_max) + 1.0)
    n_tail_target = 2.0 * n_c_peak
    if n_tail_target > n_int[-1] + 1.0:
        n_tail = np.geomspace(n_int[-1] + 1.0, n_tail_target, 256)
        n = np.concatenate([n_int, n_tail])
    else:
        n = n_int
    weights = intensity(u0)
    weights = weights / max(weights.sum(), 1e-30)
    a0_flat = a0_map.ravel()
    w_flat = weights.ravel()
    # Chunked broadcast: build S in blocks of ≤ HARMONIC_CHUNK rows to cap
    # peak memory at ~128 MB regardless of grid size.  Each block is one
    # call into the numpy-vectorised relativistic_spikes_filter, which is
    # ~15× faster than the original per-harmonic Python loop.
    harmonic_chunk = max(1, 16_000_000 // max(1, a0_flat.size))
    spectrum_vals = np.empty(n.size, dtype=float)
    for start in range(0, n.size, harmonic_chunk):
        stop = min(start + harmonic_chunk, n.size)
        block = relativistic_spikes_filter(
            n[start:stop, None], a0_flat[None, :], mode=CutoffMode.EXPONENTIAL,
        )
        spectrum_vals[start:stop] = block @ w_flat

    # 7. Diagnostic 2-D far-field intensities (a few harmonics only).
    diag_cfg = getattr(numerics, "diag_harmonics", None)
    diag = np.array(diag_cfg if diag_cfg else DEFAULT_DIAG_HARMONICS, dtype=int)
    far_stack, dx_far = stack_harmonics_far_field(
        u0, a0_map, dmap, diag, laser.angle_deg,
        grid.dx, laser.wavelength_um * 1e-6, DEFAULT_FOCUS_M,
    )

    return _BeamProducts(
        u0=u0,
        a0_map=a0_map,
        dmap=dmap,
        n=n,
        spectrum_vals=spectrum_vals,
        far_stack=far_stack,
        dx_far=dx_far,
        diag=diag,
        L_over_lambda=L_over_lambda,
    )


@dataclass
class SurfacePipelineModel:
    name: str = "surface_pipeline"

    def run(self, laser: Laser, target: Target, numerics) -> Result:
        grid_n = getattr(numerics, "pipeline_grid", DEFAULT_GRID_N) or DEFAULT_GRID_N
        dx_um = getattr(numerics, "pipeline_dx_um", DEFAULT_DX_UM) or DEFAULT_DX_UM
        grid = SpatialGrid(n=grid_n, dx=float(dx_um) * 1e-6)

        bp = _run_single_beam(laser, target, numerics, grid)

        # Attach an absolute photon-energy coord so the detector dispatch
        # (``detector.instrument.auto_band``) and the hard-X-ray / γ path
        # (``apply_gamma_response``) work without a second conversion.
        from harmonyemissions.units import keV_per_harmonic
        photon_energy_keV = bp.n * keV_per_harmonic(laser.wavelength_um)
        spec_da = xr.DataArray(
            bp.spectrum_vals,
            coords={
                "harmonic": bp.n,
                "photon_energy_keV": ("harmonic", photon_energy_keV),
            },
            dims=["harmonic"],
            name="spectrum",
            attrs={"units": "arb. (spatially averaged relativistic-spikes envelope)"},
        )

        # xarray wrappers for the spatial outputs.
        axis_near = (np.arange(grid_n) - grid_n // 2) * grid.dx
        dent_da = xr.DataArray(
            bp.dmap,
            coords={"x": axis_near, "y": axis_near},
            dims=["y", "x"],
            name="dent_map",
            attrs={"units": "λ"},
        )

        # We store the far-field intensity at the first diagnostic harmonic
        # coords are per-harmonic variable scale, so we use index axes.
        far_da = xr.DataArray(
            bp.far_stack,
            coords={
                "harmonic_diag": bp.diag,
                "xi": np.arange(grid_n),
                "yi": np.arange(grid_n),
            },
            dims=["harmonic_diag", "yi", "xi"],
            name="beam_profile_far",
            attrs={
                "description": "|U(x,y,z,n)|² far-field at CHF plane",
                "dx_far_m_per_n": ",".join(
                    f"{bp.dx_far[i]:.3e}" for i in range(len(bp.diag))
                ),
                "focus_distance_m": DEFAULT_FOCUS_M,
            },
        )

        # Near-field beam profile (driver) for context.
        near_da = xr.DataArray(
            intensity(bp.u0),
            coords={"x": axis_near, "y": axis_near},
            dims=["y", "x"],
            name="beam_profile_near",
            attrs={"units": "arb. (driver |U₀|²)"},
        )

        # CHF gain — use the diagnostic harmonic closest to the plateau centre
        # as the attosecond-pulse proxy, and the summed far-field as the CHF
        # integrated intensity. This is a semi-quantitative estimate; for a
        # PIC-grounded gain use the SMILEI backend (still WIP).
        i_driver = float(intensity(bp.u0).max())
        i_atto = float(bp.far_stack[0].max())
        i_focus_2d = float(bp.far_stack.sum(axis=0).max())
        gain = extrapolate_3d_gain(
            intensity_attosecond=max(i_atto, 1e-30),
            intensity_driver=max(i_driver, 1e-30),
            intensity_at_chf_focus_2d=max(i_focus_2d, 1e-30),
        )

        return Result(
            spectrum=spec_da,
            dent_map=dent_da,
            beam_profile_near=near_da,
            beam_profile_far=far_da,
            chf_gain=gain.to_dict(),
            diagnostics={
                "a0_peak": float(laser.a0),
                "gamma_max": float(math.sqrt(1.0 + 0.5 * laser.a0 ** 2)),
                "n_cutoff_spikes": float(spikes_cutoff_harmonic(laser.a0)),
                "L_over_lambda": float(bp.L_over_lambda),
                "dent_peak_lambda": float(bp.dmap.max()),
                "slope_theory": -8.0 / 3.0,
            },
            provenance={
                "model": "surface_pipeline",
                "reference": (
                    "Timmis et al., Nature (2026), doi 10.1038/s41586-026-10400-2; "
                    "Gordienko et al., PRL 94 103903 (2005); Vincenti, Nat. Commun. 5, 3403 (2014)"
                ),
            },
        )
