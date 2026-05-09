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
from harmonyemissions.chf.geometry import (
    BeamArray,
    build_beam_array,
    to_record,
    with_delays,
    with_phases,
)
from harmonyemissions.chf.modes import apply_structured_mode
from harmonyemissions.chf.propagation import stack_harmonics_far_amplitude
from harmonyemissions.chf.superposition import (
    FocalVolume,
    coherent_sum_homogeneous,
)
from harmonyemissions.chf.timing import analytic_phase_lock, geometric_delays
from harmonyemissions.config import LaserArrayConfig
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

    Returned by :func:`_run_single_beam` so the multi-beam (chf3d) path
    can iterate over drivers and stream their outputs through the
    coherent-superposition accumulator without duplicating per-beam physics.
    ``far_amp`` is the complex far-field amplitude (load-bearing for the
    coherent sum); ``far_stack`` = ``|far_amp|²`` is kept for the legacy
    single-beam Result tail.
    """
    u0: np.ndarray            # (N, N) complex driver near-field amplitude
    a0_map: np.ndarray        # (N, N) peak-a₀ map
    dmap: np.ndarray          # (N, N) plasma dent in units of λ
    n: np.ndarray              # harmonic axis (1..n_int + log tail to 2·n_c)
    spectrum_vals: np.ndarray  # spatially averaged spikes envelope
    far_amp: np.ndarray        # (n_diag, N, N) complex far-field amplitude
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

    # 7. Diagnostic 2-D far-field amplitudes + intensities (a few harmonics only).
    diag_cfg = getattr(numerics, "diag_harmonics", None)
    diag = np.array(diag_cfg if diag_cfg else DEFAULT_DIAG_HARMONICS, dtype=int)
    far_amp, dx_far = stack_harmonics_far_amplitude(
        u0, a0_map, dmap, diag, laser.angle_deg,
        grid.dx, laser.wavelength_um * 1e-6, DEFAULT_FOCUS_M,
    )
    far_stack = (far_amp.conj() * far_amp).real

    return _BeamProducts(
        u0=u0,
        a0_map=a0_map,
        dmap=dmap,
        n=n,
        spectrum_vals=spectrum_vals,
        far_amp=far_amp,
        far_stack=far_stack,
        dx_far=dx_far,
        diag=diag,
        L_over_lambda=L_over_lambda,
    )


def _build_single_beam_result(
    laser: Laser, bp: _BeamProducts, grid: SpatialGrid,
) -> Result:
    """Wrap a single-driver ``_BeamProducts`` into a legacy single-beam Result.

    Pure refactor — extracted verbatim from the historical
    ``SurfacePipelineModel.run`` body so the chf3d multi-beam path can
    bypass it and reuse only ``_run_single_beam`` per driver.
    """
    grid_n = grid.n
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

    axis_near = (np.arange(grid_n) - grid_n // 2) * grid.dx
    dent_da = xr.DataArray(
        bp.dmap,
        coords={"x": axis_near, "y": axis_near},
        dims=["y", "x"],
        name="dent_map",
        attrs={"units": "λ"},
    )
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
    near_da = xr.DataArray(
        intensity(bp.u0),
        coords={"x": axis_near, "y": axis_near},
        dims=["y", "x"],
        name="beam_profile_near",
        attrs={"units": "arb. (driver |U₀|²)"},
    )

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


def _clone_laser_for_beam(
    laser: Laser, beam: BeamArray, beam_index: int,
    *, override_laser: Laser | None = None,
) -> Laser:
    """Materialize a per-beam :class:`Laser` from the base + array geometry.

    The Phase C (homogeneous) default reuses the base laser and only
    rescales ``a0`` by the beam's ``a0_scale[i]``. The Extreme-Power
    overlay can pass ``override_laser`` to replace the wavelength /
    duration / spatial profile per beam — this single helper is the
    seam where heterogeneous-beam logic lives.
    """
    base = override_laser if override_laser is not None else laser
    return type(base)(
        a0=base.a0 * float(beam.a0_scale[beam_index]),
        wavelength_um=base.wavelength_um,
        duration_fs=base.duration_fs,
        cep=base.cep,
        polarization=base.polarization,
        envelope=base.envelope,
        angle_deg=base.angle_deg,
        spatial_profile=base.spatial_profile,
        spot_fwhm_um=base.spot_fwhm_um,
        super_gaussian_order=base.super_gaussian_order,
    )


def _maybe_apply_structured_mode(
    bp: _BeamProducts, beam: BeamArray, grid: SpatialGrid, laser: Laser,
) -> _BeamProducts:
    """Reapply a structured-light profile and rebuild ``far_amp``.

    The driver near-field is multiplied by the structured profile and
    the far-field amplitude is recomputed; everything else (dent map,
    spikes envelope, scale length) is unchanged because those depend on
    the intensity envelope, not the phase structure.
    """
    if beam.structured_mode is None:
        return bp
    fwhm_m = laser.spot_fwhm_um * 1e-6
    u0_new = apply_structured_mode(
        bp.u0, grid, beam.structured_mode, beam.structured_mode_params,
        fwhm_m=fwhm_m, wavelength_m=laser.wavelength_um * 1e-6,
    )
    far_amp_new, dx_far_new = stack_harmonics_far_amplitude(
        u0_new, bp.a0_map, bp.dmap, bp.diag, laser.angle_deg,
        grid.dx, laser.wavelength_um * 1e-6, DEFAULT_FOCUS_M,
    )
    far_stack_new = (far_amp_new.conj() * far_amp_new).real
    return bp._replace(
        u0=u0_new, far_amp=far_amp_new, far_stack=far_stack_new, dx_far=dx_far_new,
    )


def _resolve_focal_volume(numerics, wavelength_m: float) -> FocalVolume:
    n = getattr(numerics, "chf_focal_volume_n", None) or 16
    extent_um = getattr(numerics, "chf_focal_volume_extent_um", None) or 1.0
    mode = getattr(numerics, "chf_focal_volume_mode", "volume") or "volume"
    if mode == "point":
        n = 1
    return FocalVolume(n=int(n), extent_m=float(extent_um) * 1e-6)


def _phase_locking_sigma(beam: BeamArray, analytic_phase: np.ndarray) -> float:
    """Estimate σ as the std of the residual after analytic phase-locking."""
    residual = beam.relative_phase_rad - analytic_phase
    # Wrap to [-π, π] so the std isn't inflated by 2π jumps.
    wrapped = np.mod(residual + np.pi, 2 * np.pi) - np.pi
    return float(np.std(wrapped))


def _run_multi_beam(
    laser: Laser,
    target: Target,
    numerics,
    grid: SpatialGrid,
    laser_array: LaserArrayConfig,
) -> Result:
    """Phase C dispatcher: homogeneous coherent multi-beam run."""
    beam = build_beam_array(laser_array, focal_radius_m=DEFAULT_FOCUS_M)

    # 1. Run beam 0 to get the per-driver amplitude scale, fix delays/phases.
    laser_0 = _clone_laser_for_beam(laser, beam, 0)
    bp_0 = _run_single_beam(laser_0, target, numerics, grid)
    bp_0 = _maybe_apply_structured_mode(bp_0, beam, grid, laser_0)

    # 2. Resolve delays (geometric default) and phases (analytic default).
    if laser_array.relative_delay_fs is None:
        beam = with_delays(beam, geometric_delays(beam))
    if laser_array.relative_phase_rad is None:
        method = getattr(numerics, "phase_optimiser", "analytic") or "analytic"
        if method != "analytic":
            raise NotImplementedError(
                f"phase_optimiser={method!r} lands in Phase D"
            )
        wavelength_m = laser.wavelength_um * 1e-6
        h_centre = int(beam_array_centre_harmonic(bp_0))
        # |A_i| is identical for all i in the homogeneous case, so use
        # the beam-0 on-axis amplitude as the shared reference.
        N = grid.n
        h_idx = int(np.argmin(np.abs(bp_0.diag - h_centre)))
        A_i = complex(bp_0.far_amp[h_idx, N // 2, N // 2])
        amplitudes = np.full(beam.n_beams, A_i, dtype=np.complex128)
        beam = with_phases(
            beam,
            analytic_phase_lock(beam, amplitudes, h_centre, wavelength_m),
        )

    # 3. Build focal volume and accumulate beam 0 + remaining beams.
    volume = _resolve_focal_volume(numerics, laser.wavelength_um * 1e-6)
    far_amp_per_beam: list[np.ndarray] = [bp_0.far_amp]
    per_beam_far: list[np.ndarray] | None = (
        [bp_0.far_stack] if getattr(numerics, "store_per_beam_far_field", False) else None
    )
    spectrum_vals_acc = bp_0.spectrum_vals.copy()
    dent_peak = float(bp_0.dmap.max())

    for i in range(1, beam.n_beams):
        laser_i = _clone_laser_for_beam(laser, beam, i)
        bp_i = _run_single_beam(laser_i, target, numerics, grid)
        bp_i = _maybe_apply_structured_mode(bp_i, beam, grid, laser_i)
        far_amp_per_beam.append(bp_i.far_amp)
        if per_beam_far is not None:
            per_beam_far.append(bp_i.far_stack)
        spectrum_vals_acc = spectrum_vals_acc + bp_i.spectrum_vals
        dent_peak = max(dent_peak, float(bp_i.dmap.max()))

    acc = coherent_sum_homogeneous(
        beam=beam,
        far_amp_per_beam=far_amp_per_beam,
        diag_harmonics=bp_0.diag,
        wavelength_m=laser.wavelength_um * 1e-6,
        volume=volume,
    )

    # 4. Build the chf3d Result.
    return _build_multi_beam_result(
        laser=laser,
        bp_0=bp_0,
        beam=beam,
        grid=grid,
        spectrum_vals_summed=spectrum_vals_acc,
        accumulator=acc,
        per_beam_far=per_beam_far,
        dent_peak=dent_peak,
    )


def beam_array_centre_harmonic(bp: _BeamProducts) -> int:
    """Pick a diagnostic harmonic near the BGP plateau centre."""
    diag = np.asarray(bp.diag, dtype=int)
    if diag.size == 1:
        return int(diag[0])
    # Prefer a harmonic close to but past the first one (which is
    # typically the driver itself); pick the median.
    return int(np.median(diag))


def _build_multi_beam_result(
    *,
    laser: Laser,
    bp_0: _BeamProducts,
    beam: BeamArray,
    grid: SpatialGrid,
    spectrum_vals_summed: np.ndarray,
    accumulator,
    per_beam_far: list[np.ndarray] | None,
    dent_peak: float,
) -> Result:
    """Build a :class:`Result` from the chf3d Phase C accumulator."""
    grid_n = grid.n
    from harmonyemissions.units import keV_per_harmonic
    photon_energy_keV = bp_0.n * keV_per_harmonic(laser.wavelength_um)

    # Spatially averaged spectrum: each beam's plateau adds incoherently
    # (the coherent sum is voxel-level only). Normalise by N for the
    # "per-driver" representation that downstream tools expect.
    spec_avg = spectrum_vals_summed / max(beam.n_beams, 1)
    spec_da = xr.DataArray(
        spec_avg,
        coords={
            "harmonic": bp_0.n,
            "photon_energy_keV": ("harmonic", photon_energy_keV),
        },
        dims=["harmonic"],
        name="spectrum",
        attrs={
            "units": "arb. (per-driver-averaged relativistic-spikes envelope)",
            "description": "Spectrum averaged over drivers; coherent sum is voxel-level (chf_focal_volume).",
        },
    )

    # Pull beam-0's 2-D outputs for back-compat plotting.
    axis_near = (np.arange(grid_n) - grid_n // 2) * grid.dx
    dent_da = xr.DataArray(
        bp_0.dmap,
        coords={"x": axis_near, "y": axis_near},
        dims=["y", "x"],
        name="dent_map",
        attrs={"units": "λ", "note": "beam-0 dent map (multi-beam runs)"},
    )
    near_da = xr.DataArray(
        intensity(bp_0.u0),
        coords={"x": axis_near, "y": axis_near},
        dims=["y", "x"],
        name="beam_profile_near",
        attrs={"units": "arb.", "note": "beam-0 driver |U₀|² (multi-beam runs)"},
    )
    far_da = xr.DataArray(
        bp_0.far_stack,
        coords={
            "harmonic_diag": bp_0.diag,
            "xi": np.arange(grid_n),
            "yi": np.arange(grid_n),
        },
        dims=["harmonic_diag", "yi", "xi"],
        name="beam_profile_far",
        attrs={
            "description": "beam-0 |U(x,y,z,n)|² far-field (multi-beam runs)",
            "focus_distance_m": DEFAULT_FOCUS_M,
        },
    )

    # 3-D coherent focal-volume cube.
    volume = accumulator.volume
    diag = accumulator.diag_harmonics
    intensity_cube = accumulator.intensity()  # (H, n, n, n)
    voxel_axis = (np.arange(volume.n) - volume.n // 2) * volume.voxel_size_m
    focal_da = xr.DataArray(
        intensity_cube,
        coords={
            "harmonic_diag": diag,
            "x_focus_m": voxel_axis,
            "y_focus_m": voxel_axis,
            "z_focus_m": voxel_axis,
        },
        dims=["harmonic_diag", "x_focus_m", "y_focus_m", "z_focus_m"],
        name="chf_focal_volume",
        attrs={
            "description": "Coherent multi-beam I(r,n) from chf3d Phase C accumulator",
            "voxel_size_m": float(volume.voxel_size_m),
            "extent_m": float(volume.extent_m),
        },
    )

    # Per-beam far-field stack (optional).
    if per_beam_far is not None:
        per_beam_far_arr = np.stack(per_beam_far, axis=0)  # (N_beams, H, N, N)
        per_beam_da: xr.DataArray | None = xr.DataArray(
            per_beam_far_arr,
            coords={
                "beam_index": np.arange(per_beam_far_arr.shape[0]),
                "harmonic_diag": bp_0.diag,
                "xi": np.arange(grid_n),
                "yi": np.arange(grid_n),
            },
            dims=["beam_index", "harmonic_diag", "yi", "xi"],
            name="per_beam_far_field",
            attrs={"description": "Per-driver |U(x,y,z,n)|² in beam frame"},
        )
    else:
        per_beam_da = None

    # CHF gain — single-beam baseline is bp_0; coherent gain comes from
    # the volumetric peak.
    i_driver = float(intensity(bp_0.u0).max())
    i_atto = float(bp_0.far_stack[0].max())
    i_focus_2d = float(bp_0.far_stack.sum(axis=0).max())
    base = extrapolate_3d_gain(
        intensity_attosecond=max(i_atto, 1e-30),
        intensity_driver=max(i_driver, 1e-30),
        intensity_at_chf_focus_2d=max(i_focus_2d, 1e-30),
    )
    chf_gain = dict(base.to_dict())
    # Coherent multi-beam gain, normalised against the single-beam 2-D
    # CHF intensity.
    peak_per_h = accumulator.peak_intensity()
    gamma_3d_coherent = float(peak_per_h.max() / max(i_focus_2d, 1e-30))
    n_beams = int(beam.n_beams)
    # F_geom: ratio of measured to ideal N²·Γ_2D² scaling.
    ideal = n_beams * n_beams * base.gamma_2d ** 2
    f_geom = float(gamma_3d_coherent / ideal) if ideal > 0 else 0.0
    chf_gain.update(
        Gamma_3D_coherent=gamma_3d_coherent,
        Gamma_total_coherent=float(base.gamma_d * gamma_3d_coherent),
        F_geom=f_geom,
        N_beams=float(n_beams),
        phase_locking_sigma_rad=0.0,  # analytic phase-lock baseline
    )

    return Result(
        spectrum=spec_da,
        dent_map=dent_da,
        beam_profile_near=near_da,
        beam_profile_far=far_da,
        chf_focal_volume=focal_da,
        per_beam_far_field=per_beam_da,
        beam_array_geometry=to_record(beam),
        chf_gain=chf_gain,
        diagnostics={
            "a0_peak": float(laser.a0),
            "gamma_max": float(math.sqrt(1.0 + 0.5 * laser.a0 ** 2)),
            "n_cutoff_spikes": float(spikes_cutoff_harmonic(laser.a0)),
            "L_over_lambda": float(bp_0.L_over_lambda),
            "dent_peak_lambda": float(dent_peak),
            "slope_theory": -8.0 / 3.0,
            "n_beams": float(n_beams),
            "geometry": beam.geometry,
            "placement": beam.placement,
            "polarization_mode": beam.polarization_mode,
        },
        provenance={
            "model": "surface_pipeline",
            "mode": "chf3d",
            "reference": (
                "Timmis et al., Nature (2026); "
                "chf3d Phase C coherent multi-beam kernel (this run)"
            ),
        },
    )


@dataclass
class SurfacePipelineModel:
    name: str = "surface_pipeline"

    def run(
        self, laser: Laser, target: Target, numerics,
        *, laser_array: LaserArrayConfig | None = None,
        extreme_power=None,
    ) -> Result:
        grid_n = getattr(numerics, "pipeline_grid", DEFAULT_GRID_N) or DEFAULT_GRID_N
        dx_um = getattr(numerics, "pipeline_dx_um", DEFAULT_DX_UM) or DEFAULT_DX_UM
        grid = SpatialGrid(n=grid_n, dx=float(dx_um) * 1e-6)

        if laser_array is None and extreme_power is None:
            bp = _run_single_beam(laser, target, numerics, grid)
            return _build_single_beam_result(laser, bp, grid)

        if extreme_power is not None:
            from harmonyemissions.models._extreme_power import run_extreme_power
            return run_extreme_power(
                laser, target, numerics, grid, laser_array, extreme_power,
            )

        return _run_multi_beam(laser, target, numerics, grid, laser_array)
