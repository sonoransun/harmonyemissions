"""Extreme-Power overlay dispatcher (heterogeneous beams + RR + QED).

Sits on top of the chf3d Phase C kernel and adds three distinct things:

1. **Heterogeneous beams** — when ``ExtremePowerConfig.per_beam_lasers``
   is supplied each driver carries its own wavelength / duration /
   profile, and the coherent sum runs on a physical-frequency ω-grid
   instead of the homogeneous harmonic-order axis.
2. **Landau–Lifshitz radiation-reaction derate** — at a₀ ≳ 100 the
   classical γ³ cutoff is reduced by RR friction. Applied per-beam to
   the spikes envelope (not to the far-field amplitude — RR reshapes
   the cutoff, not the propagation).
3. **Perturbative QED diagnostics** — Schwinger ratio χ, vacuum
   birefringence Δφ, Breit–Wheeler pair rate at the post-CHF focal-volume
   peak. Off by default; gated on ``enable_qed``.

Each branch is independently togglable, so a config can opt into RR-only
or QED-only studies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from harmonyemissions.beam import SpatialGrid, intensity
from harmonyemissions.chf.gain import extrapolate_3d_gain
from harmonyemissions.chf.geometry import build_beam_array, to_record, with_delays
from harmonyemissions.chf.superposition import (
    FocalVolume,
    build_omega_grid,
    coherent_sum_heterogeneous_diag,
    coherent_sum_homogeneous,
    resample_to_omega,
)
from harmonyemissions.chf.timing import geometric_delays
from harmonyemissions.config import ExtremePowerConfig, LaserArrayConfig
from harmonyemissions.gamma.radiation_reaction import landau_lifshitz_cutoff_derate
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.qed import qed_diagnostics
from harmonyemissions.target import Target
from harmonyemissions.units import C, keV_per_harmonic

if TYPE_CHECKING:
    from harmonyemissions.models.surface_pipeline import _BeamProducts


def _materialize_per_beam_lasers(
    base_laser: Laser,
    extreme: ExtremePowerConfig,
    n_beams: int,
) -> list[Laser]:
    if extreme.per_beam_lasers is None:
        return [base_laser] * n_beams
    return [lc.build() for lc in extreme.per_beam_lasers]


def _heterogeneous(lasers: list[Laser]) -> bool:
    if not lasers:
        return False
    ref = lasers[0]
    for laser_i in lasers[1:]:
        if laser_i.wavelength_um != ref.wavelength_um:
            return True
    return False


def _apply_rr_derate(
    bp: _BeamProducts,
    laser_i: Laser,
    extreme: ExtremePowerConfig,
) -> tuple[_BeamProducts, dict[str, float]]:
    """Reshape the spikes envelope under Landau–Lifshitz friction."""
    derate, chi = landau_lifshitz_cutoff_derate(
        laser_i.a0,
        laser_i.wavelength_um,
        chi_clip=extreme.rr_clip_chi,
        a0_floor=extreme.rr_a0_floor,
    )
    rr_meta = {
        "a0": float(laser_i.a0),
        "wavelength_um": float(laser_i.wavelength_um),
        "chi_e": float(chi),
        "derate": float(derate),
        "clipped": bool(chi > extreme.rr_clip_chi),
    }
    if derate >= 1.0:
        return bp, rr_meta
    n = bp.n
    n_c_old = float(np.max(n) / 2.0)  # rough cutoff proxy
    n_c_new = derate * n_c_old
    # Smooth tanh roll-off centred on n_c_new with width 0.2 * n_c_old.
    width = max(0.2 * n_c_old, 1.0)
    derate_factor = 0.5 * (1.0 + np.tanh((n_c_new - n) / width))
    spectrum_new = bp.spectrum_vals * derate_factor
    return bp._replace(spectrum_vals=spectrum_new), rr_meta


def _on_axis_amp_per_n(bp: _BeamProducts) -> np.ndarray:
    """Return the centre-pixel complex amplitude for each diagnostic harmonic."""
    n_grid = bp.far_amp.shape[-1]
    return bp.far_amp[:, n_grid // 2, n_grid // 2]


def run_extreme_power(
    laser: Laser,
    target: Target,
    numerics,
    grid: SpatialGrid,
    laser_array: LaserArrayConfig,
    extreme: ExtremePowerConfig,
) -> Result:
    """Top-level dispatcher for the Extreme-Power overlay."""
    from harmonyemissions.models.surface_pipeline import (
        DEFAULT_FOCUS_M,
        _clone_laser_for_beam,
        _maybe_apply_structured_mode,
        _run_single_beam,
    )

    beam = build_beam_array(laser_array, focal_radius_m=DEFAULT_FOCUS_M)
    n_beams = beam.n_beams

    # Heterogeneous laser materialisation.
    per_beam_base_lasers = _materialize_per_beam_lasers(laser, extreme, n_beams)
    heterogeneous = _heterogeneous(per_beam_base_lasers)

    # Default delays (geometric) if user omitted them.
    if laser_array.relative_delay_fs is None:
        beam = with_delays(beam, geometric_delays(beam))

    # Run per-beam single-driver pipelines.
    bp_list: list[_BeamProducts] = []
    rr_meta_list: list[dict[str, float]] = []
    for i in range(n_beams):
        laser_i = _clone_laser_for_beam(
            laser, beam, i,
            override_laser=per_beam_base_lasers[i],
        )
        bp_i = _run_single_beam(laser_i, target, numerics, grid)
        bp_i = _maybe_apply_structured_mode(bp_i, beam, grid, laser_i)
        if extreme.enable_radiation_reaction:
            bp_i, rr_meta = _apply_rr_derate(bp_i, laser_i, extreme)
        else:
            rr_meta = {
                "a0": float(laser_i.a0),
                "wavelength_um": float(laser_i.wavelength_um),
                "derate": 1.0,
                "clipped": False,
            }
        rr_meta_list.append(rr_meta)
        bp_list.append(bp_i)

    bp_0 = bp_list[0]
    laser_0 = per_beam_base_lasers[0]
    spectrum_vals_summed = sum((bp.spectrum_vals for bp in bp_list), np.zeros_like(bp_0.spectrum_vals))

    # Heterogeneous → resample onto common ω-grid; homogeneous → reuse Phase C.
    if heterogeneous:
        spec_da, focal_da, _h_idx_unused, omega_centre = _heterogeneous_path(
            bp_list, per_beam_base_lasers, beam, numerics, extreme,
        )
        peak_intensity = float(focal_da.values.max())
        # Coordinates and units: spectrum is on omega_rad_s for heterogeneous.
        omega_eff = float(omega_centre)
        wavelength_eff = 2.0 * np.pi * C / max(omega_eff, 1e-30)
    else:
        spec_da, focal_da = _homogeneous_path(
            bp_list, per_beam_base_lasers, beam, numerics, grid, spectrum_vals_summed,
        )
        peak_intensity = float(focal_da.values.max())
        wavelength_eff = float(laser_0.wavelength_um) * 1e-6
        omega_eff = 2.0 * np.pi * C / wavelength_eff

    # Build base CHF gain (single-beam reference from beam 0) + coherent extras.
    i_driver = float(intensity(bp_0.u0).max())
    i_atto = float(bp_0.far_stack[0].max())
    i_focus_2d = float(bp_0.far_stack.sum(axis=0).max())
    base_gain = extrapolate_3d_gain(
        intensity_attosecond=max(i_atto, 1e-30),
        intensity_driver=max(i_driver, 1e-30),
        intensity_at_chf_focus_2d=max(i_focus_2d, 1e-30),
    )
    chf_gain = dict(base_gain.to_dict())
    gamma_3d_coherent = float(peak_intensity / max(i_focus_2d, 1e-30))
    ideal = n_beams * n_beams * base_gain.gamma_2d ** 2
    f_geom = float(gamma_3d_coherent / ideal) if ideal > 0 else 0.0
    chf_gain.update(
        Gamma_3D_coherent=gamma_3d_coherent,
        Gamma_total_coherent=float(base_gain.gamma_d * gamma_3d_coherent),
        F_geom=f_geom,
        N_beams=float(n_beams),
        phase_locking_sigma_rad=0.0,
    )

    # QED diagnostics from the focal-volume peak.
    qed_dict: dict[str, float | bool] = {}
    qed_warn: str | None = None
    if extreme.enable_qed:
        # Convert peak intensity from arbitrary internal units to W/m² is
        # not strictly meaningful (the surface-HHG analytical model is
        # parametric, not absolute). Estimate the absolute focal
        # intensity from the driver intensity scaled by gamma_3d_coherent.
        from harmonyemissions.units import a0_to_intensity
        i_driver_w_per_cm2 = a0_to_intensity(laser_0.a0, laser_0.wavelength_um)
        i_focal_w_per_m2 = i_driver_w_per_cm2 * 1e4 * max(gamma_3d_coherent, 0.0)
        rayleigh = np.pi * (laser_0.spot_fwhm_um * 1e-6 / 2.0) ** 2 / wavelength_eff
        qed_dict = qed_diagnostics(
            i_focal_w_per_m2, omega_eff, length_m=rayleigh,
            chi_warn=extreme.qed_chi_warn,
        )
        if qed_dict.get("validity_exceeded"):
            qed_warn = (
                f"χ = {qed_dict['schwinger_ratio']:.3e} exceeds qed_chi_warn = "
                f"{extreme.qed_chi_warn} — perturbative QED is no longer reliable."
            )

    rr_summary = {
        "enabled": bool(extreme.enable_radiation_reaction),
        "per_beam": rr_meta_list,
        "any_clipped": any(m.get("clipped", False) for m in rr_meta_list),
    }

    diagnostics = {
        "a0_peak": float(laser.a0),
        "n_beams": float(n_beams),
        "geometry": beam.geometry,
        "placement": beam.placement,
        "polarization_mode": beam.polarization_mode,
        "heterogeneous_beams": bool(heterogeneous),
        "rr_enabled": bool(extreme.enable_radiation_reaction),
        "qed_enabled": bool(extreme.enable_qed),
    }

    provenance = {
        "model": "surface_pipeline",
        "mode": "extreme_power",
        "reference": (
            "Timmis et al., Nature (2026); chf3d Phase C kernel + "
            "Landau–Lifshitz radiation friction (Bulanov 2011) + "
            "Heisenberg–Euler perturbative QED diagnostics"
        ),
    }
    if qed_warn:
        provenance["qed_validity_warning"] = qed_warn
    if rr_summary["any_clipped"]:
        provenance["rr_clipped"] = True

    return Result(
        spectrum=spec_da,
        chf_focal_volume=focal_da,
        beam_array_geometry=to_record(beam),
        chf_gain=chf_gain,
        qed_diagnostics=qed_dict,
        radiation_reaction=rr_summary,
        diagnostics=diagnostics,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Path selectors: homogeneous (reuse Phase C) vs. heterogeneous (ω-grid).
# ---------------------------------------------------------------------------


def _homogeneous_path(
    bp_list, per_beam_lasers, beam, numerics, grid, spectrum_vals_summed,
) -> tuple[xr.DataArray, xr.DataArray]:
    laser_0 = per_beam_lasers[0]
    far_amp_per_beam = [bp.far_amp for bp in bp_list]
    volume = FocalVolume(
        n=getattr(numerics, "chf_focal_volume_n", None) or 16,
        extent_m=(getattr(numerics, "chf_focal_volume_extent_um", None) or 1.0) * 1e-6,
    )
    if getattr(numerics, "chf_focal_volume_mode", "volume") == "point":
        volume = FocalVolume(n=1, extent_m=0.0)
    acc = coherent_sum_homogeneous(
        beam=beam,
        far_amp_per_beam=far_amp_per_beam,
        diag_harmonics=bp_list[0].diag,
        wavelength_m=laser_0.wavelength_um * 1e-6,
        volume=volume,
    )
    photon_energy_keV = bp_list[0].n * keV_per_harmonic(laser_0.wavelength_um)
    spec_da = xr.DataArray(
        spectrum_vals_summed / max(beam.n_beams, 1),
        coords={
            "harmonic": bp_list[0].n,
            "photon_energy_keV": ("harmonic", photon_energy_keV),
        },
        dims=["harmonic"],
        name="spectrum",
        attrs={"units": "arb. (driver-averaged spikes envelope)"},
    )
    cube = acc.intensity()
    voxel_axis = (np.arange(volume.n) - volume.n // 2) * volume.voxel_size_m
    focal_da = xr.DataArray(
        cube,
        coords={
            "harmonic_diag": bp_list[0].diag,
            "x_focus_m": voxel_axis,
            "y_focus_m": voxel_axis,
            "z_focus_m": voxel_axis,
        },
        dims=["harmonic_diag", "x_focus_m", "y_focus_m", "z_focus_m"],
        name="chf_focal_volume",
        attrs={
            "description": "Coherent multi-beam I(r,n) (homogeneous, extreme_power)",
            "voxel_size_m": float(volume.voxel_size_m),
            "extent_m": float(volume.extent_m),
        },
    )
    return spec_da, focal_da


def _heterogeneous_path(
    bp_list, per_beam_lasers, beam, numerics, extreme,
) -> tuple[xr.DataArray, xr.DataArray, int, float]:
    """Resample each beam's on-axis amplitude onto a common ω-grid and
    coherently sum across heterogeneous wavelengths."""
    # Build per-beam ω-arrays: ω_i(n) = n · 2π c / λ_i for n in bp_i.diag.
    # We coherently sum on-axis only at the diagnostic harmonics of each
    # beam (the volumetric cube is too expensive on 4096 ω-points).
    omega_min, omega_max = [], []
    on_axis_per_beam_diag: list[np.ndarray] = []
    omega_per_beam_diag: list[np.ndarray] = []
    for bp_i, laser_i in zip(bp_list, per_beam_lasers, strict=True):
        omega_i = bp_i.diag * (2.0 * np.pi * C) / (laser_i.wavelength_um * 1e-6)
        omega_min.append(float(omega_i.min()))
        omega_max.append(float(omega_i.max()))
        on_axis_per_beam_diag.append(_on_axis_amp_per_n(bp_i))
        omega_per_beam_diag.append(omega_i)

    omega_grid = build_omega_grid(
        omega_min, omega_max,
        n_points=extreme.omega_grid_points,
        pad=extreme.omega_grid_pad,
    )

    # Resample each beam's on-axis amplitude onto the common ω-grid.
    on_axis_per_beam_omega: list[np.ndarray] = []
    for A_diag, omega_i in zip(on_axis_per_beam_diag, omega_per_beam_diag, strict=True):
        on_axis_per_beam_omega.append(
            resample_to_omega(A_diag, omega_i, omega_grid)
        )

    # Pick a small set of diagnostic ω-indices for the focal-volume cube.
    # Default: 4 log-spaced points in the union range.
    n_diag = 4
    diag_indices = np.linspace(0, omega_grid.size - 1, n_diag, dtype=int)
    volume_n = getattr(numerics, "chf_focal_volume_n", None) or 8
    extent_m = (getattr(numerics, "chf_focal_volume_extent_um", None) or 1.0) * 1e-6
    if getattr(numerics, "chf_focal_volume_mode", "volume") == "point":
        volume = FocalVolume(n=1, extent_m=0.0)
    else:
        volume = FocalVolume(n=int(volume_n), extent_m=float(extent_m))

    full_centre, diag_cube = coherent_sum_heterogeneous_diag(
        beam=beam,
        on_axis_amp_per_omega=on_axis_per_beam_omega,
        omega_grid=omega_grid,
        diag_omega_indices=diag_indices,
        volume=volume,
    )

    spec_da = xr.DataArray(
        full_centre,
        coords={"omega_rad_s": omega_grid},
        dims=["omega_rad_s"],
        name="spectrum",
        attrs={
            "units": "arb. (coherent on-axis intensity, heterogeneous ω-axis)",
            "axis": "physical-frequency",
        },
    )
    voxel_axis = (np.arange(volume.n) - volume.n // 2) * volume.voxel_size_m
    focal_da = xr.DataArray(
        diag_cube,
        coords={
            "omega_diag_rad_s": omega_grid[diag_indices],
            "x_focus_m": voxel_axis,
            "y_focus_m": voxel_axis,
            "z_focus_m": voxel_axis,
        },
        dims=["omega_diag_rad_s", "x_focus_m", "y_focus_m", "z_focus_m"],
        name="chf_focal_volume",
        attrs={
            "description": "Coherent multi-beam I(r,ω) on diagnostic ω-subset",
            "voxel_size_m": float(volume.voxel_size_m),
            "extent_m": float(volume.extent_m),
        },
    )
    # Use the median ω as the "effective" frequency for QED diagnostics.
    omega_centre = float(np.median(omega_grid[diag_indices]))
    h_index_centre = int(diag_indices[len(diag_indices) // 2])
    return spec_da, focal_da, h_index_centre, omega_centre
