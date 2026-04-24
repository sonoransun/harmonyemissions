from pathlib import Path

import pytest
import yaml

from harmonyemissions.config import LaserConfig, load_config
from harmonyemissions.presets import apply_preset, get_preset, list_presets

CONFIGS = Path(__file__).parent.parent / "configs"


def test_list_presets_non_empty():
    names = list_presets()
    assert set(names) >= {
        "hapls",
        "dipole",
        "bivoj",
        "polaris",
        "apollon",
        "yb_yag_generic",
    }


def test_get_preset_hapls_landmarks():
    p = get_preset("hapls")
    assert p["wavelength_um"] == pytest.approx(1.030)
    assert p["duration_fs"] == pytest.approx(30.0)
    assert p["polarization"] == "p"
    assert "HAPLS" in p["facility"]


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown laser preset"):
        get_preset("does-not-exist")


def test_apply_preset_fills_missing_fields():
    # User supplies only spot geometry → preset fills λ, τ, a₀, polarization, envelope.
    merged = apply_preset("hapls", {"spot_fwhm_um": 3.0})
    assert merged["wavelength_um"] == pytest.approx(1.030)
    assert merged["duration_fs"] == pytest.approx(30.0)
    assert merged["a0"] == pytest.approx(15.0)  # from a0_reference
    assert merged["polarization"] == "p"
    assert merged["spot_fwhm_um"] == pytest.approx(3.0)


def test_apply_preset_user_overrides_win():
    # User sets a smaller a0 and a different angle; they must beat the preset.
    merged = apply_preset("hapls", {"a0": 2.5, "angle_deg": 30.0})
    assert merged["a0"] == pytest.approx(2.5)
    assert merged["angle_deg"] == pytest.approx(30.0)
    # Non-overridden preset fields still present.
    assert merged["wavelength_um"] == pytest.approx(1.030)


def test_laser_config_preset_round_trip():
    # LaserConfig with only `preset` should validate (a0 filled from preset).
    lc = LaserConfig.model_validate({"preset": "hapls"})
    assert lc.preset == "hapls"
    assert lc.wavelength_um == pytest.approx(1.030)
    assert lc.a0 == pytest.approx(15.0)
    # build() produces a Laser without the preset field.
    laser = lc.build()
    assert laser.wavelength_um == pytest.approx(1.030)
    assert laser.a0 == pytest.approx(15.0)


def test_laser_config_preset_user_override():
    lc = LaserConfig.model_validate({"preset": "hapls", "a0": 2.5, "duration_fs": 25.0})
    assert lc.a0 == pytest.approx(2.5)
    assert lc.duration_fs == pytest.approx(25.0)
    # Wavelength still from preset.
    assert lc.wavelength_um == pytest.approx(1.030)


def test_laser_config_unknown_preset_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LaserConfig.model_validate({"preset": "bogus", "a0": 1.0})


def test_default_wavelength_unchanged_when_no_preset():
    # Pre-existing configs must keep 0.8 μm default (no accidental DPSSL shift).
    lc = LaserConfig.model_validate({"a0": 10.0})
    assert lc.preset is None
    assert lc.wavelength_um == pytest.approx(0.8)


@pytest.mark.parametrize(
    "config_name",
    [
        "hapls_surface_hhg.yaml",
        "dipole_surface_rom.yaml",
        "polaris_cwe.yaml",
        "apollon_chf.yaml",
        "hapls_betatron.yaml",
    ],
)
def test_new_dpssl_configs_validate(config_name):
    cfg = load_config(CONFIGS / config_name)
    assert cfg.laser.preset is not None
    # Every preset-backed config must end up with a real wavelength / a₀.
    assert cfg.laser.wavelength_um > 0.0
    assert cfg.laser.a0 > 0.0


def test_yaml_dump_of_preset_config_still_loadable(tmp_path):
    cfg = load_config(CONFIGS / "hapls_surface_hhg.yaml")
    # Round-trip through dump
    out = tmp_path / "roundtrip.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(), fh)
    cfg2 = load_config(out)
    assert cfg2.laser.wavelength_um == pytest.approx(cfg.laser.wavelength_um)
    assert cfg2.laser.a0 == pytest.approx(cfg.laser.a0)
