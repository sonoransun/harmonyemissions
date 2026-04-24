"""CLI smoke tests using typer.testing.CliRunner."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from harmonyemissions.cli import app

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_validate_rom_config():
    result = runner.invoke(app, ["validate", str(CONFIGS / "rom_default.yaml")])
    assert result.exit_code == 0, result.output
    assert "rom" in result.output


def test_validate_all_bundled_configs():
    for cfg in CONFIGS.glob("*.yaml"):
        r = runner.invoke(app, ["validate", str(cfg)])
        assert r.exit_code == 0, f"{cfg.name}: {r.output}"


def test_run_writes_output(tmp_path):
    out = tmp_path / "run.h5"
    r = runner.invoke(app, ["run", str(CONFIGS / "rom_default.yaml"), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists() and out.stat().st_size > 1000


def test_scan_sweep(tmp_path):
    out_dir = tmp_path / "scan"
    r = runner.invoke(
        app,
        [
            "scan",
            str(CONFIGS / "scan_example.yaml"),
            "-p", "laser.a0=1,3,10",
            "-d", str(out_dir),
            "-j", "1",
        ],
    )
    assert r.exit_code == 0, r.output
    files = list(out_dir.glob("*.h5"))
    assert len(files) == 3


def test_plot_kind_invalid_errors(tmp_path):
    # First create a valid run file.
    out = tmp_path / "run.h5"
    runner.invoke(app, ["run", str(CONFIGS / "rom_default.yaml"), "-o", str(out)])
    r = runner.invoke(app, ["plot", str(out), "-k", "garbage"])
    assert r.exit_code != 0


# ----------------------------------------------------------------------
# Timmis 2026 surface_pipeline + detector CLI coverage
# ----------------------------------------------------------------------


@pytest.fixture
def pipeline_run(tmp_path):
    """Run a small surface_pipeline config and return the output path."""
    import yaml
    cfg_path = tmp_path / "pipeline.yaml"
    with open(CONFIGS / "chf_gemini.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["numerics"]["pipeline_grid"] = 32          # keep the test fast
    cfg["numerics"]["pipeline_dx_um"] = 0.2
    cfg["laser"]["a0"] = 10.0                      # smaller a0 → fewer harmonics
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    out = tmp_path / "pipeline.h5"
    r = runner.invoke(app, ["chf", str(cfg_path), "-o", str(out)])
    assert r.exit_code == 0, r.output
    return out


def test_harmony_chf_command(pipeline_run):
    assert pipeline_run.exists() and pipeline_run.stat().st_size > 1000


def test_harmony_detector_command(pipeline_run, tmp_path):
    out = tmp_path / "detector.h5"
    r = runner.invoke(app, ["detector", str(pipeline_run), "--al-um", "1.0", "-o", str(out)])
    assert r.exit_code == 0, r.output
    from harmonyemissions.io import load_result
    result = load_result(out)
    assert result.instrument_spectrum is not None


@pytest.mark.parametrize("kind", ["spectrum", "dent", "beam", "chf"])
def test_harmony_plot_each_kind(pipeline_run, tmp_path, kind):
    png = tmp_path / f"{kind}.png"
    r = runner.invoke(app, ["plot", str(pipeline_run), "-k", kind, "-o", str(png)])
    assert r.exit_code == 0, r.output
    assert png.exists() and png.stat().st_size > 1000


def test_harmony_plot_instrument(pipeline_run, tmp_path):
    out = tmp_path / "with_detector.h5"
    runner.invoke(app, ["detector", str(pipeline_run), "-o", str(out)])
    png = tmp_path / "instrument.png"
    r = runner.invoke(app, ["plot", str(out), "-k", "instrument", "-o", str(png)])
    assert r.exit_code == 0, r.output
    assert png.exists()
