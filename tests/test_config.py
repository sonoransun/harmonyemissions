"""Tests for YAML config loading and regime validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from harmonyemissions.config import RunConfig, dump_config, load_config


def test_all_bundled_configs_load(tmp_path):
    root = Path(__file__).resolve().parents[1] / "configs"
    allowed = {
        "rom", "bgp", "cse", "lewenstein", "betatron", "surface_pipeline", "cwe",
        "bremsstrahlung", "kalpha", "ics",
    }
    for cfg_file in root.glob("*.yaml"):
        cfg = load_config(cfg_file)
        assert cfg.model in allowed, f"{cfg_file.name}: unexpected model {cfg.model!r}"


def test_regime_mismatch_raises():
    bad = {
        "model": "rom",
        "backend": "analytical",
        "laser": {"a0": 1.0},
        "target": {"kind": "gas", "gas_species": "Ar"},
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(bad)


def test_dump_then_load_roundtrip(tmp_path):
    src = Path(__file__).resolve().parents[1] / "configs" / "rom_default.yaml"
    cfg = load_config(src)
    out = tmp_path / "round.yaml"
    dump_config(cfg, out)
    cfg2 = load_config(out)
    assert cfg2.laser.a0 == cfg.laser.a0
    assert cfg2.model == cfg.model
