"""Tests for parameter-scan orchestration."""

import numpy as np
import pytest

from harmonyemissions.config import RunConfig
from harmonyemissions.scan import (
    _apply_override,
    _coerce,
    parse_param_spec,
    run_scan,
)


def test_parse_param_spec_basic():
    path, values = parse_param_spec("laser.a0=1,3,10")
    assert path == "laser.a0"
    assert values == [1, 3, 10]


def test_parse_param_spec_scientific_notation():
    path, values = parse_param_spec("target.prepulse_intensity_rel=1e-3,5e-4")
    assert path == "target.prepulse_intensity_rel"
    assert values == pytest.approx([1e-3, 5e-4])


def test_parse_param_spec_negative_and_mixed():
    path, values = parse_param_spec("laser.cep=-3.14,0,3.14")
    assert path == "laser.cep"
    assert values == pytest.approx([-3.14, 0.0, 3.14])


def test_parse_param_spec_missing_equals():
    with pytest.raises(ValueError):
        parse_param_spec("laser.a0,1,2")


def test_coerce_types():
    assert _coerce("42") == 42
    assert _coerce("3.14") == pytest.approx(3.14)
    assert _coerce("1e-3") == pytest.approx(1e-3)
    assert _coerce("gaussian") == "gaussian"


def test_apply_override_nested_path():
    d = {"a": {"b": {"c": 1}}}
    _apply_override(d, "a.b.c", 42)
    assert d["a"]["b"]["c"] == 42


def test_apply_override_replaces_nested_list():
    d = {"numerics": {"diag_harmonics": [1, 2, 3]}}
    _apply_override(d, "numerics.diag_harmonics", [10, 20, 30])
    assert d["numerics"]["diag_harmonics"] == [10, 20, 30]


def _make_base_cfg() -> RunConfig:
    from pathlib import Path

    from harmonyemissions.config import load_config
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "scan_example.yaml")


def test_run_scan_serial_vs_parallel_bit_identical(tmp_path):
    base = _make_base_cfg()
    grid = [{"laser.a0": a} for a in [1.0, 3.0, 5.0]]
    serial = run_scan(base, grid, output_dir=None, n_jobs=1)
    parallel = run_scan(base, grid, output_dir=None, n_jobs=2)
    assert len(serial) == len(parallel) == 3
    for s, p in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(s.result.spectrum.values, p.result.spectrum.values)


def test_run_scan_writes_files(tmp_path):
    base = _make_base_cfg()
    grid = [{"laser.a0": a} for a in [1.0, 5.0]]
    pts = run_scan(base, grid, output_dir=tmp_path, n_jobs=1)
    for pt in pts:
        assert pt.path is not None and pt.path.exists()


def test_run_scan_skips_file_write_when_output_dir_none():
    base = _make_base_cfg()
    grid = [{"laser.a0": 2.0}]
    pts = run_scan(base, grid, output_dir=None, n_jobs=1)
    assert pts[0].path is None
