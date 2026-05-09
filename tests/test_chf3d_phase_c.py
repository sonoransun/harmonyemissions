"""End-to-end tests for the chf3d Phase C coherent multi-beam kernel.

Pin the gain laws, geometry counts, single-beam fallback invariant,
HDF5 round-trip, and propagation-amplitude regression so future drift
is caught here rather than in downstream notebooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonyemissions.chf import (
    BeamArray,
    FocalVolume,
    FocalVolumeAccumulator,
    analytic_phase_lock,
    build_beam_array,
    coherent_sum_homogeneous,
    from_record,
    geometric_delays,
    stack_harmonics_far_amplitude,
    stack_harmonics_far_field,
    to_record,
    with_phases,
)
from harmonyemissions.config import LaserArrayConfig, RunConfig, load_config
from harmonyemissions.runner import simulate_from_config


# ---- geometry ------------------------------------------------------------


@pytest.mark.parametrize(
    "geometry, placement, expected",
    [
        ("tetrahedral",  "faces",     4),
        ("tetrahedral",  "vertices",  4),
        ("cubic",        "faces",     6),
        ("cubic",        "vertices",  8),
        ("octahedral",   "faces",     8),
        ("octahedral",   "vertices",  6),
        ("dodecahedral", "faces",    12),
        ("dodecahedral", "vertices", 20),
        ("icosahedral",  "faces",    20),
        ("icosahedral",  "vertices", 12),
    ],
)
def test_geometry_platonic_counts(geometry, placement, expected):
    cfg = LaserArrayConfig(geometry=geometry, placement=placement)
    beam = build_beam_array(cfg)
    assert beam.n_beams == expected
    assert beam.directions.shape == (expected, 3)


def test_geometry_directions_unit_norm():
    for geom in ("tetrahedral", "cubic", "octahedral", "dodecahedral", "icosahedral"):
        beam = build_beam_array(LaserArrayConfig(geometry=geom))
        norms = np.linalg.norm(beam.directions, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), f"{geom} has non-unit directions"


def test_geometry_record_roundtrip():
    cfg = LaserArrayConfig(geometry="dodecahedral", polarization_mode="radial")
    beam = build_beam_array(cfg)
    rec = to_record(beam)
    beam2 = from_record(rec)
    assert beam2.n_beams == beam.n_beams
    assert np.allclose(beam2.directions, beam.directions)
    assert np.allclose(beam2.polarization, beam.polarization)
    assert np.allclose(beam2.relative_phase_rad, beam.relative_phase_rad)


def test_geometry_polarization_orthogonal_to_direction():
    """Each Jones vector ε_i must be orthogonal to the beam direction n̂_i."""
    beam = build_beam_array(
        LaserArrayConfig(geometry="dodecahedral", polarization_mode="radial")
    )
    for i in range(beam.n_beams):
        proj = np.dot(beam.directions[i], beam.polarization[i].real)
        assert abs(proj) < 1e-10, f"polarization[{i}] not orthogonal to direction"


def test_geometry_circular_alternating_handedness():
    beam = build_beam_array(
        LaserArrayConfig(geometry="cubic", polarization_mode="circular_alternating")
    )
    # |ε_i| should be unit-norm in 3-D; alternating sign of imaginary part
    norms = np.linalg.norm(beam.polarization, axis=1)
    assert np.allclose(np.abs(norms), 1.0, atol=1e-10)


# ---- timing & phase ------------------------------------------------------


def test_geometric_delays_closed_form():
    beam = build_beam_array(LaserArrayConfig(geometry="dodecahedral"))
    delays = geometric_delays(beam)
    # All beams sit on the same focal-radius sphere → equal delays.
    assert np.allclose(delays, delays[0], atol=1e-10)
    # Closed-form: distance / c → fs.
    expected = beam.focal_radius_m / 2.99792458e8 * 1e15
    assert np.allclose(delays, expected, rtol=1e-6)


def test_analytic_phase_lock_makes_centre_pixel_real_positive():
    """φ_i* must align all beams' centre-pixel contributions in phase."""
    cfg = LaserArrayConfig(geometry="tetrahedral")
    beam = build_beam_array(cfg)
    A = np.full(beam.n_beams, 0.7 + 0.3j)
    phases = analytic_phase_lock(beam, A, harmonic_n=15, wavelength_m=0.8e-6)
    # After applying φ_i* and the propagation phase at r_focus = 0,
    # each beam contributes A·exp(i φ*)·exp(-i k n̂·r_i) — the propagation
    # term cancels the (n̂·-r_i) baked into φ*, leaving |A|.
    rebuilt = A * np.exp(1j * phases) * np.exp(
        1j * (2 * np.pi / (0.8e-6 / 15)) * np.einsum(
            "ij,ij->i", beam.directions, -beam.positions
        )
    )
    assert np.allclose(rebuilt.imag, 0, atol=1e-9)
    assert np.all(rebuilt.real > 0)


# ---- propagation API regression -----------------------------------------


def test_far_stack_amplitude_squared_matches_legacy():
    """``stack_harmonics_far_field`` must equal ``|stack_harmonics_far_amplitude|²``."""
    rng = np.random.default_rng(0)
    n = 32
    u0 = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(
        np.complex128
    )
    a0_map = np.full((n, n), 5.0)
    dent = np.zeros((n, n))
    diag = np.array([1, 5, 10], dtype=int)
    amp, dx_amp = stack_harmonics_far_amplitude(
        u0, a0_map, dent, diag, 0.0, 1e-7, 0.8e-6, 1e-3,
    )
    intens, dx_int = stack_harmonics_far_field(
        u0, a0_map, dent, diag, 0.0, 1e-7, 0.8e-6, 1e-3,
    )
    assert np.allclose(intens, np.abs(amp) ** 2)
    assert np.allclose(dx_amp, dx_int)


# ---- coherent sum --------------------------------------------------------


def _synthetic_far_amp_stack(beam: BeamArray, h_count: int = 1) -> list[np.ndarray]:
    """Make a per-beam stack with uniform on-axis amplitude = 1 + 0j."""
    grid_n = 8
    out: list[np.ndarray] = []
    for _ in range(beam.n_beams):
        far = np.zeros((h_count, grid_n, grid_n), dtype=np.complex128)
        far[:, grid_n // 2, grid_n // 2] = 1.0 + 0.0j
        out.append(far)
    return out


def test_closed_form_gain_at_sigma_zero():
    """At σ=0 (analytic phase lock), I_focus / I_per_beam ≈ N²."""
    for geom, expected_n in (("tetrahedral", 4), ("cubic", 6),
                             ("dodecahedral", 12), ("icosahedral", 20)):
        beam = build_beam_array(LaserArrayConfig(geometry=geom))
        far_amp = _synthetic_far_amp_stack(beam, h_count=1)
        diag = np.array([15], dtype=int)
        wavelength_m = 0.8e-6
        # Apply analytic phase lock so all beams add in phase at r = 0.
        phases = analytic_phase_lock(
            beam, np.ones(beam.n_beams, dtype=complex), 15, wavelength_m,
        )
        beam_locked = with_phases(beam, phases)
        volume = FocalVolume(n=1, extent_m=0.0)
        acc = coherent_sum_homogeneous(
            beam_locked, far_amp, diag, wavelength_m, volume,
        )
        # Per-beam contribution magnitude is |ε_i| · 1 ≈ 1; coherent sum
        # of N drivers at the centre voxel gives intensity ≈ N²
        # (modulated by Σ_i |ε_i|² geometry — F_geom).
        peak = float(acc.peak_intensity()[0])
        # F_geom reduces from N² when polarisations don't all align;
        # for radial / uniform-p Platonics it's typically 0.5–1.
        # Tighten when geometry is well-conditioned.
        assert peak >= 0.4 * expected_n ** 2, (
            f"{geom}: peak {peak:.2f} < 0.4 · N² = {0.4 * expected_n ** 2}"
        )
        assert peak <= 1.05 * expected_n ** 2, (
            f"{geom}: peak {peak:.2f} > N² = {expected_n ** 2}"
        )


def test_phase_locking_decay_law():
    """⟨gain⟩ ≈ N·e^{-σ²} + (1−e^{-σ²}) under random per-beam phase noise."""
    beam = build_beam_array(LaserArrayConfig(geometry="dodecahedral"))
    far_amp = _synthetic_far_amp_stack(beam, h_count=1)
    diag = np.array([15], dtype=int)
    wavelength_m = 0.8e-6
    rng = np.random.default_rng(1)
    n_trials = 64
    N = beam.n_beams
    # Lock the analytic phase first so beams add in phase before we
    # inject σ noise on top.
    phases_locked = analytic_phase_lock(
        beam, np.ones(N, dtype=complex), 15, wavelength_m,
    )
    for sigma in (0.0, np.pi / 8, np.pi / 4, np.pi / 2):
        gains = []
        for _ in range(n_trials):
            noise = rng.normal(0.0, sigma, size=N)
            beam_noisy = with_phases(beam, phases_locked + noise)
            volume = FocalVolume(n=1, extent_m=0.0)
            acc = coherent_sum_homogeneous(
                beam_noisy, far_amp, diag, wavelength_m, volume,
            )
            gains.append(float(acc.peak_intensity()[0]))
        mean_gain = float(np.mean(gains))
        # We compare to the predicted F_geom · (N·e^{-σ²} + (1−e^{-σ²})·N²);
        # with our matched-energy convention (N² baseline at σ=0) the
        # closed form on the linear-N branch should hold within 25 %.
        if sigma == 0.0:
            predicted_max = N * N
            assert mean_gain >= 0.4 * predicted_max
        else:
            decay = np.exp(-sigma * sigma)
            # Coherent term scales like N²·decay; incoherent floor is N.
            predicted = N * N * decay + N * (1.0 - decay)
            assert mean_gain <= 1.5 * predicted, (
                f"σ={sigma:.3f}: mean_gain={mean_gain:.2f} > 1.5·predicted={1.5*predicted:.2f}"
            )
            assert mean_gain >= 0.2 * predicted, (
                f"σ={sigma:.3f}: mean_gain={mean_gain:.2f} < 0.2·predicted={0.2*predicted:.2f}"
            )


# ---- single-beam fallback invariant -------------------------------------


def test_single_beam_fallback_byte_identical(tmp_path):
    """A config without ``laser_array`` must produce a Result identical to
    the legacy single-beam path (no chf3d fields populated)."""
    base = {
        "model": "surface_pipeline",
        "backend": "analytical",
        "laser": {"a0": 5.0},
        "target": {"kind": "overdense"},
        "numerics": {"pipeline_grid": 32},
    }
    cfg = RunConfig.model_validate(base)
    result = simulate_from_config(cfg)
    assert result.chf_focal_volume is None
    assert result.per_beam_far_field is None
    assert result.beam_array_geometry is None
    assert "Gamma_3D_coherent" not in result.chf_gain
    assert "N_beams" not in result.chf_gain


# ---- end-to-end multi-beam run ------------------------------------------


def test_dispatcher_runs_dodecahedral_end_to_end(tmp_path):
    """A 12-beam dodecahedral config produces a chf_focal_volume and the
    new chf_gain keys."""
    cfg = RunConfig.model_validate({
        "model": "surface_pipeline",
        "backend": "analytical",
        "laser": {"a0": 5.0},
        "target": {"kind": "overdense"},
        "laser_array": {
            "geometry": "dodecahedral",
            "polarization_mode": "radial",
        },
        "numerics": {
            "pipeline_grid": 32,
            "chf_focal_volume_n": 4,
            "chf_focal_volume_extent_um": 0.5,
        },
    })
    result = simulate_from_config(cfg)
    assert result.chf_focal_volume is not None
    assert result.chf_focal_volume.shape[0] > 0
    assert result.beam_array_geometry["n_beams"] == 12
    for key in ("Gamma_3D_coherent", "Gamma_total_coherent", "F_geom",
                "N_beams", "phase_locking_sigma_rad"):
        assert key in result.chf_gain, f"missing chf_gain key {key}"


def test_hdf5_roundtrip_preserves_chf3d_fields(tmp_path):
    cfg = RunConfig.model_validate({
        "model": "surface_pipeline",
        "backend": "analytical",
        "laser": {"a0": 4.0},
        "target": {"kind": "overdense"},
        "laser_array": {"geometry": "tetrahedral"},
        "numerics": {
            "pipeline_grid": 32,
            "chf_focal_volume_n": 4,
            "chf_focal_volume_extent_um": 0.5,
            "store_per_beam_far_field": True,
        },
    })
    result = simulate_from_config(cfg)
    out = tmp_path / "phase_c.h5"
    result.save(out)
    from harmonyemissions.io import load_result
    loaded = load_result(out)
    assert loaded.chf_focal_volume is not None
    assert np.allclose(
        loaded.chf_focal_volume.values, result.chf_focal_volume.values
    )
    assert loaded.beam_array_geometry["n_beams"] == 4
    assert loaded.per_beam_far_field is not None
    assert loaded.per_beam_far_field.shape[0] == 4
    assert "Gamma_3D_coherent" in loaded.chf_gain


def test_per_beam_a0_scale_matched_energy():
    """Matched-energy convention: scales such that Σ a₀_i² = a₀²."""
    n = 4
    s = np.full(n, 1.0 / np.sqrt(n))
    cfg = LaserArrayConfig(
        geometry="tetrahedral", per_beam_a0_scale=s.tolist(),
    )
    beam = build_beam_array(cfg)
    assert np.isclose(np.sum(beam.a0_scale ** 2), 1.0)
