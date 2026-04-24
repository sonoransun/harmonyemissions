"""Typer-based CLI: ``harmony run|scan|plot|validate``.

Entry point declared in ``pyproject.toml`` as ``harmony = harmonyemissions.cli:app``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import typer

from harmonyemissions.config import load_config
from harmonyemissions.detector.deconvolve import DetectorConfig, apply_instrument_response
from harmonyemissions.detector.filters import FilterSpec
from harmonyemissions.detector.gamma_response import GammaDetector
from harmonyemissions.detector.hard_xray import FilterStack
from harmonyemissions.detector.instrument import apply_detector, auto_band
from harmonyemissions.detector.scintillator import DetectorConfig as ScintConfig
from harmonyemissions.detector.soft_xray import SoftXrayConfig
from harmonyemissions.io import load_result
from harmonyemissions.presets import get_preset, list_presets, preset_metadata
from harmonyemissions.runner import simulate_from_config
from harmonyemissions.scan import parse_param_spec, run_scan
from harmonyemissions.viz import (
    plot_beam_profile,
    plot_chf_gain,
    plot_dent_map,
    plot_instrument_spectrum,
    plot_pulse,
    plot_scaling,
    plot_spectrum,
    save_figure,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Harmony of Emissions CLI")


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o", help="HDF5 output file."),
) -> None:
    """Run a single simulation described by a YAML config."""
    cfg = load_config(config)
    result = simulate_from_config(cfg)
    out_path = output or (Path(cfg.output) if cfg.output else Path("run.h5"))
    result.save(out_path)
    typer.echo(f"wrote {out_path}")
    for k, v in result.summary().items():
        typer.echo(f"  {k}: {v}")


@app.command()
def scan(
    config: Path = typer.Argument(..., exists=True, readable=True),
    param: list[str] = typer.Option(
        ..., "--param", "-p",
        help="Repeatable: 'dotted.path=v1,v2,...'. Cartesian product is swept.",
    ),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", "-d"),
    n_jobs: int = typer.Option(1, "--jobs", "-j"),
    mpi: bool = typer.Option(False, "--mpi", help="Distribute across MPI ranks (requires mpi4py + mpirun)."),
) -> None:
    """Run a Cartesian-product parameter sweep over one or more knobs."""
    cfg = load_config(config)
    specs = [parse_param_spec(s) for s in param]
    paths, value_lists = zip(*specs, strict=True) if specs else ([], [])
    grid = [
        dict(zip(paths, combo, strict=True))
        for combo in itertools.product(*value_lists)
    ]
    if mpi:
        from harmonyemissions.parallel.mpi import run_scan_mpi
        points = run_scan_mpi(cfg, grid, output_dir=output_dir, gather=True)
        if points is None:  # non-root rank
            return
    else:
        points = run_scan(cfg, grid, output_dir=output_dir, n_jobs=n_jobs)
    typer.echo(f"wrote {len(points)} runs to {output_dir}")
    for pt in points:
        typer.echo(f"  {pt.overrides}  →  {pt.path}")


@app.command()
def plot(
    run_file: Path = typer.Argument(..., exists=True, readable=True),
    kind: str = typer.Option(
        "spectrum", "--kind", "-k",
        help="spectrum | pulse | scaling | dent | beam | chf | instrument",
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Plot a single-run HDF5 file."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "spectrum":
        plot_spectrum(load_result(run_file), ax=ax)
    elif kind == "pulse":
        plot_pulse(load_result(run_file), ax=ax)
    elif kind == "scaling":
        paths = sorted(run_file.glob("*.h5")) if run_file.is_dir() else [run_file]
        plot_scaling(paths, param="a0", ax=ax)
    elif kind == "dent":
        plot_dent_map(load_result(run_file), ax=ax)
    elif kind == "beam":
        plot_beam_profile(load_result(run_file), which="near", ax=ax)
    elif kind == "chf":
        plot_chf_gain(load_result(run_file), ax=ax)
    elif kind == "instrument":
        plot_instrument_spectrum(load_result(run_file), ax=ax)
    else:
        raise typer.BadParameter(f"unknown --kind {kind!r}")
    fig.tight_layout()
    out = output or run_file.with_suffix(".png")
    save_figure(fig, out)
    typer.echo(f"wrote {out}")


@app.command()
def validate(
    config: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Schema-check a config file without running a simulation."""
    cfg = load_config(config)
    typer.echo(f"OK — {cfg.model} on {cfg.target.kind} via {cfg.backend}")


@app.command("list-presets")
def list_presets_cmd() -> None:
    """List the packaged laser-facility presets (HAPLS, DiPOLE, ...)."""
    for name in list_presets():
        p = get_preset(name)
        meta = preset_metadata(name)
        typer.echo(
            f"{name:16s} λ={p['wavelength_um']:5.3f} μm  "
            f"τ={p['duration_fs']:6.1f} fs  "
            f"a₀≈{p.get('a0_reference', '?'):>5}  "
            f"{meta.get('facility', '')}"
        )


@app.command()
def chf(
    config: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Run the full CHF pipeline (surface_pipeline model + CHF gain breakdown)."""
    cfg = load_config(config)
    # Force the model to surface_pipeline so users can drop a standard config in.
    cfg = cfg.model_copy(update={"model": "surface_pipeline"})
    result = simulate_from_config(cfg)
    out_path = output or Path("chf.h5")
    result.save(out_path)
    typer.echo(f"wrote {out_path}")
    typer.echo("CHF gain breakdown:")
    for k, v in result.chf_gain.items():
        typer.echo(f"  {k:12s}: {v:.3e}")
    typer.echo("Diagnostics:")
    for k, v in result.diagnostics.items():
        typer.echo(f"  {k}: {v}")


def _parse_filter_spec(s: str) -> FilterSpec:
    """Parse 'material-thicknessumORnm' — e.g. 'cu-50um', 'kapton-7um', 'be-100um'."""
    try:
        material, rest = s.split("-", 1)
    except ValueError as exc:
        raise typer.BadParameter(f"expected 'material-NNNum', got {s!r}") from exc
    rest = rest.lower()
    if rest.endswith("um"):
        thickness_um = float(rest[:-2])
    elif rest.endswith("nm"):
        thickness_um = float(rest[:-2]) * 1e-3
    elif rest.endswith("mm"):
        thickness_um = float(rest[:-2]) * 1e3
    else:
        raise typer.BadParameter(
            f"thickness must end in um/nm/mm, got {s!r}"
        )
    return FilterSpec(material=material.lower(), thickness_um=thickness_um)


@app.command("detector")
def detector_cmd(
    run_file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
    band: str = typer.Option(
        "auto", "--band",
        help="auto | xuv | xray-soft | xray-hard | gamma",
    ),
    filter_chain: list[str] = typer.Option(
        [], "--filter",
        help="Repeatable 'material-thicknessum' (e.g. cu-50um, kapton-7um).",
    ),
    detector: str | None = typer.Option(
        None, "--detector",
        help="Detector 'material-thicknessum' (e.g. si-500um, cdte-1mm).",
    ),
    driver_wavelength_um: float = typer.Option(0.8, "--lambda-um"),
    # Legacy XUV flags (honoured when band resolves to xuv):
    al_um: float = typer.Option(1.5, "--al-um"),
    include_second_order: bool = typer.Option(True, "--second-order/--no-second-order"),
    include_third_order: bool = typer.Option(False, "--third-order/--no-third-order"),
) -> None:
    """Apply the band-aware detector response to a saved simulation run."""
    result = load_result(run_file)
    resolved_band = band if band != "auto" else auto_band(result.spectrum, driver_wavelength_um)

    if resolved_band == "xuv":
        cfg = DetectorConfig(
            al_thickness_um=al_um,
            include_second_order=include_second_order,
            include_third_order=include_third_order,
        )
        sig = apply_instrument_response(result.spectrum, driver_wavelength_um, cfg)
    elif resolved_band == "xray-soft":
        soft_filters = tuple(_parse_filter_spec(f) for f in filter_chain) or (
            FilterSpec("kapton", 7.0),
        )
        sig = apply_detector(
            result.spectrum, driver_wavelength_um, band="xray-soft",
            soft_config=SoftXrayConfig(filters=soft_filters),
        )
    elif resolved_band in ("xray-hard", "gamma"):
        layers = tuple((f.material.capitalize() if f.material in ("al","cu","ta","w","pb","au") else f.material,
                        f.thickness_um) for f in (_parse_filter_spec(x) for x in filter_chain))
        filt = FilterStack(layers=layers) if layers else FilterStack(
            layers=(("Al", 500.0),) if resolved_band == "gamma" else (("Al", 100.0),)
        )
        if detector is not None:
            dspec = _parse_filter_spec(detector)
            # Canonicalise names to match ScintConfig keys (NaI / CsI / LYSO / CdTe / HPGe / Si / YAG / IP).
            canonical = {
                "nai": "NaI", "csi": "CsI", "lyso": "LYSO", "cdte": "CdTe",
                "hpge": "HPGe", "si": "Si", "yag": "YAG", "ip": "IP",
            }
            det_cfg = ScintConfig(
                name=canonical.get(dspec.material.lower(), "CsI"),
                thickness_mm=dspec.thickness_um * 1e-3,
            )
        else:
            det_cfg = ScintConfig(name="Si" if resolved_band == "xray-hard" else "CsI")
        sig = apply_detector(
            result.spectrum, driver_wavelength_um, band=resolved_band,
            gamma_detector=GammaDetector(filters=filt, detector=det_cfg),
        )
    else:
        raise typer.BadParameter(f"unknown --band {band!r}")

    result.instrument_spectrum = sig
    out_path = output or run_file.with_name(run_file.stem + "_detector.h5")
    result.save(out_path)
    typer.echo(f"wrote {out_path} (band={resolved_band})")


if __name__ == "__main__":
    app()
