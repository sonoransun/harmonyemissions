"""Schema-only tests for the chf3d `laser_array` config block (Phase A).

Phase A only ships parsing, validation, and runner gating — the multi-beam
pipeline itself lands in Phase C. These tests pin the validator behaviour and
the round-trip / gating contract so later phases cannot drift the schema.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from harmonyemissions.config import (
    LaserArrayConfig,
    RunConfig,
    dump_config,
    load_config,
)


_BASE_OVERDENSE = {
    "model": "surface_pipeline",
    "backend": "analytical",
    "laser": {"a0": 5.0},
    "target": {"kind": "overdense"},
}


def _with_array(array: dict, **overrides) -> dict:
    cfg = {k: v for k, v in _BASE_OVERDENSE.items()}
    cfg.update(overrides)
    cfg["laser_array"] = array
    return cfg


# ---- effective_n_beams ----------------------------------------------------


@pytest.mark.parametrize(
    "geometry, placement, expected",
    [
        ("tetrahedral",  "faces",    4),
        ("tetrahedral",  "vertices", 4),
        ("cubic",        "faces",    6),
        ("cubic",        "vertices", 8),
        ("octahedral",   "faces",    8),
        ("octahedral",   "vertices", 6),
        ("dodecahedral", "faces",   12),
        ("dodecahedral", "vertices",20),
        ("icosahedral",  "faces",   20),
        ("icosahedral",  "vertices",12),
    ],
)
def test_platonic_face_vertex_counts(geometry, placement, expected):
    cfg = LaserArrayConfig(geometry=geometry, placement=placement)
    assert cfg.effective_n_beams() == expected


def test_explicit_geometry_uses_directions_length():
    cfg = LaserArrayConfig(
        geometry="explicit",
        directions=[(1, 0, 0), (-1, 0, 0), (0, 1, 0)],
    )
    assert cfg.effective_n_beams() == 3


def test_ring_requires_n_beams():
    with pytest.raises(ValidationError):
        LaserArrayConfig(geometry="ring")


def test_fibonacci_requires_n_beams():
    with pytest.raises(ValidationError):
        LaserArrayConfig(geometry="fibonacci_sphere")


# ---- validators -----------------------------------------------------------


def test_explicit_requires_directions():
    with pytest.raises(ValidationError):
        LaserArrayConfig(geometry="explicit")


def test_list_lengths_must_match_n_beams():
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="tetrahedral",
            relative_phase_rad=[0.0, 0.1],  # length 2, expected 4
        )


def test_relative_delay_length_must_match():
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="dodecahedral",  # 12 faces
            relative_delay_fs=[0.0] * 11,
        )


def test_per_beam_a0_scale_power_conservation():
    # 4 beams, each at 0.6 ⇒ sum-of-squares = 1.44 > 1 ⇒ must reject.
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="tetrahedral",
            per_beam_a0_scale=[0.6, 0.6, 0.6, 0.6],
        )


def test_per_beam_a0_scale_at_unity_ok():
    # 4 beams, each at 0.5 ⇒ sum-of-squares = 1.0, exactly at the limit.
    cfg = LaserArrayConfig(
        geometry="tetrahedral",
        per_beam_a0_scale=[0.5, 0.5, 0.5, 0.5],
    )
    assert cfg.effective_n_beams() == 4


def test_explicit_polarization_needs_vectors():
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="tetrahedral",
            polarization_mode="explicit",
        )


def test_lg_mode_requires_l_and_p():
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="explicit",
            directions=[(0, 0, 1)],
            structured_mode="lg",
            structured_mode_params={"l": 1},  # missing 'p'
        )


def test_bessel_mode_requires_kr():
    with pytest.raises(ValidationError):
        LaserArrayConfig(
            geometry="explicit",
            directions=[(0, 0, 1)],
            structured_mode="bessel",
            structured_mode_params={"order": 0},  # missing 'kr_per_k'
        )


def test_lg_mode_with_valid_params():
    cfg = LaserArrayConfig(
        geometry="explicit",
        directions=[(0, 0, 1)],
        structured_mode="lg",
        structured_mode_params={"l": 2, "p": 0},
    )
    assert cfg.structured_mode == "lg"


# ---- RunConfig integration ------------------------------------------------


def test_runconfig_accepts_laser_array_with_surface_pipeline():
    cfg = RunConfig.model_validate(
        _with_array({"geometry": "icosahedral"})
    )
    assert cfg.laser_array is not None
    assert cfg.laser_array.geometry == "icosahedral"
    assert cfg.laser_array.effective_n_beams() == 20


def test_runconfig_rejects_laser_array_with_lewenstein():
    payload = {
        "model": "lewenstein",
        "backend": "analytical",
        "laser": {"a0": 1.0},
        "target": {"kind": "gas", "gas_species": "Ar"},
        "laser_array": {"geometry": "tetrahedral"},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_runconfig_rejects_laser_array_with_betatron():
    payload = {
        "model": "betatron",
        "backend": "analytical",
        "laser": {"a0": 1.0},
        "target": {"kind": "underdense"},
        "laser_array": {"geometry": "tetrahedral"},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_runconfig_default_has_no_laser_array():
    cfg = RunConfig.model_validate(_BASE_OVERDENSE)
    assert cfg.laser_array is None


# ---- YAML round-trip ------------------------------------------------------


def test_yaml_roundtrip_preserves_laser_array(tmp_path):
    cfg = RunConfig.model_validate(
        _with_array({
            "geometry": "dodecahedral",
            "polarization_mode": "radial",
            "relative_phase_rad": [0.0] * 12,
        })
    )
    out = tmp_path / "round.yaml"
    dump_config(cfg, out)
    cfg2 = load_config(out)
    assert cfg2.laser_array is not None
    assert cfg2.laser_array.geometry == "dodecahedral"
    assert cfg2.laser_array.polarization_mode == "radial"
    assert cfg2.laser_array.relative_phase_rad == [0.0] * 12


def test_yaml_legacy_config_loads_with_laser_array_none():
    # Existing single-beam configs in configs/ should still load and have
    # laser_array == None (back-compat invariant).
    root = Path(__file__).resolve().parents[1] / "configs"
    for cfg_file in root.glob("*.yaml"):
        cfg = load_config(cfg_file)
        assert cfg.laser_array is None, f"{cfg_file.name} unexpectedly grew laser_array"


# ---- numerics chf3d knobs -------------------------------------------------


def test_numerics_chf3d_defaults():
    cfg = RunConfig.model_validate(_BASE_OVERDENSE)
    assert cfg.numerics.chf_focal_volume_n is None
    assert cfg.numerics.chf_focal_volume_extent_um is None
    assert cfg.numerics.chf_focal_volume_mode == "volume"
    assert cfg.numerics.store_per_beam_far_field is False
    assert cfg.numerics.phase_optimiser == "analytic"


def test_numerics_chf3d_yaml_roundtrip(tmp_path):
    payload = dict(_BASE_OVERDENSE)
    payload["numerics"] = {
        "chf_focal_volume_n": 32,
        "chf_focal_volume_extent_um": 0.5,
        "chf_focal_volume_mode": "point",
        "store_per_beam_far_field": True,
        "phase_optimiser": "scipy_lbfgs",
    }
    cfg = RunConfig.model_validate(payload)
    out = tmp_path / "n.yaml"
    dump_config(cfg, out)
    cfg2 = load_config(out)
    assert cfg2.numerics.chf_focal_volume_n == 32
    assert cfg2.numerics.chf_focal_volume_extent_um == 0.5
    assert cfg2.numerics.chf_focal_volume_mode == "point"
    assert cfg2.numerics.store_per_beam_far_field is True
    assert cfg2.numerics.phase_optimiser == "scipy_lbfgs"


# ---- runner gate ----------------------------------------------------------


def test_runner_raises_until_phase_c(tmp_path):
    from harmonyemissions.runner import simulate_from_config

    cfg_payload = _with_array({"geometry": "tetrahedral"})
    cfg_path = tmp_path / "phaseA.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_payload))
    cfg = load_config(cfg_path)

    with pytest.raises(NotImplementedError, match="Phase C"):
        simulate_from_config(cfg)


def test_runner_works_for_legacy_single_beam():
    """The legacy single-beam path must remain bit-for-bit unchanged."""
    from harmonyemissions.runner import simulate_from_config

    cfg = RunConfig.model_validate(_BASE_OVERDENSE)
    assert cfg.laser_array is None
    result = simulate_from_config(cfg)
    assert result.spectrum is not None
    assert result.chf_focal_volume is None
    assert result.per_beam_far_field is None
    assert result.beam_array_geometry is None
