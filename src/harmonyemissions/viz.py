"""Matplotlib plotting helpers that work on any :class:`Result`.

All plotting is regime-agnostic: spectrum plots consume ``result.spectrum``,
pulse plots consume ``result.attosecond_pulse`` or ``result.time_field``.
Callers that need fancier layouts can get the raw ``xarray.DataArray``s and
drive matplotlib themselves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from harmonyemissions.models.base import Result


def plot_spectrum(result: Result, ax=None, *, db: bool = True, show_fit: bool = True):
    """Plot the harmonic spectrum on a log scale.

    If ``show_fit`` is True, overlay the least-squares power-law fit.
    """
    ax = ax or plt.gca()
    n = result.spectrum.coords["harmonic"].values
    s = result.spectrum.values
    norm = s / s.max() if s.max() > 0 else s
    if db:
        y = 10.0 * np.log10(np.maximum(norm, 1e-30))
        ax.set_ylabel("spectral intensity [dB]")
    else:
        y = norm
        ax.set_yscale("log")
        ax.set_ylabel("spectral intensity (norm.)")
    ax.plot(n, y, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("harmonic order n")
    if show_fit:
        slope, intercept = result.fit_power_law()
        if np.isfinite(slope):
            mask = n >= 5
            y_fit = (np.exp(intercept) * n[mask] ** slope)
            y_fit_norm = y_fit / s.max() if s.max() > 0 else y_fit
            y_fit_plot = 10.0 * np.log10(np.maximum(y_fit_norm, 1e-30)) if db else y_fit_norm
            ax.plot(n[mask], y_fit_plot, "--", lw=0.8,
                    label=f"slope {slope:.2f}")
            ax.legend(loc="lower left")
    return ax


def plot_pulse(result: Result, ax=None):
    """Plot the attosecond / reflected-field pulse in the time domain."""
    ax = ax or plt.gca()
    arr = result.attosecond_pulse if result.attosecond_pulse is not None else result.time_field
    if arr is None:
        raise ValueError("Result has no time-domain field to plot.")
    ax.plot(arr.coords["t_over_T0"].values, arr.values, lw=1.0)
    ax.set_xlabel("t / T₀")
    ax.set_ylabel("field [arb.]")
    return ax


def plot_scaling(paths: list[Path], param: str, ax=None):
    """Plot cutoff harmonic vs a swept parameter (reads each run from disk)."""
    ax = ax or plt.gca()
    xs, ys = [], []
    for p in paths:
        r = Result.load(p)
        prov = r.provenance or {}
        value = prov.get("laser", {}).get(param.split(".")[-1])
        if value is None:
            continue
        xs.append(float(value))
        ys.append(r.cutoff_harmonic())
    order = np.argsort(xs)
    ax.loglog(np.array(xs)[order], np.array(ys)[order], "o-")
    ax.set_xlabel(param)
    ax.set_ylabel("cutoff harmonic n_c")
    return ax


def plot_dent_map(result: Result, ax=None):
    """Image-plot Δz(x', y') in units of λ (2-D dent map from surface_pipeline)."""
    ax = ax or plt.gca()
    if result.dent_map is None:
        raise ValueError("Result has no dent_map (non-pipeline model?).")
    dm = result.dent_map
    x = dm.coords["x"].values * 1e6  # m → μm
    y = dm.coords["y"].values * 1e6
    im = ax.imshow(
        dm.values,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="plasma",
    )
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("Plasma surface dent Δz / λ")
    plt.colorbar(im, ax=ax, label="Δz / λ")
    return ax


def plot_beam_profile(result: Result, which: str = "near", harmonic_idx: int = 0, ax=None):
    """Plot the driver near-field or a single harmonic's far-field intensity."""
    ax = ax or plt.gca()
    if which == "near":
        arr = result.beam_profile_near
        if arr is None:
            raise ValueError("Result has no beam_profile_near.")
        x = arr.coords["x"].values * 1e6
        y = arr.coords["y"].values * 1e6
        im = ax.imshow(arr.values, extent=(x.min(), x.max(), y.min(), y.max()),
                       origin="lower", cmap="inferno")
        ax.set_title("Driver near-field |U₀|²")
        ax.set_xlabel("x [μm]")
        ax.set_ylabel("y [μm]")
    elif which == "far":
        arr = result.beam_profile_far
        if arr is None:
            raise ValueError("Result has no beam_profile_far.")
        harm = int(arr.coords["harmonic_diag"].values[harmonic_idx])
        im = ax.imshow(arr.values[harmonic_idx], origin="lower", cmap="inferno")
        ax.set_title(f"Harmonic n={harm} far-field |U|²")
    else:
        raise ValueError(f"which must be 'near' or 'far', got {which!r}")
    plt.colorbar(im, ax=ax)
    return ax


def plot_chf_gain(result: Result, ax=None):
    """Bar chart of the CHF gain breakdown."""
    ax = ax or plt.gca()
    if not result.chf_gain:
        raise ValueError("Result has no chf_gain.")
    keys = list(result.chf_gain.keys())
    vals = [result.chf_gain[k] for k in keys]
    ax.bar(keys, vals)
    ax.set_yscale("log")
    ax.set_ylabel("gain factor")
    ax.set_title("CHF gain breakdown")
    return ax


def plot_instrument_spectrum(result: Result, ax=None):
    """Overlay simulated spectrum with the detector-corrected CCD signal."""
    ax = ax or plt.gca()
    if result.instrument_spectrum is None:
        raise ValueError("Result has no instrument_spectrum; run `harmony detector` first.")
    n_sim = result.spectrum.coords["harmonic"].values
    n_ccd = result.instrument_spectrum.coords["harmonic"].values
    ax.loglog(n_sim, result.spectrum.values / result.spectrum.values.max(),
              label="simulation (normalized)", lw=1.0)
    ax.loglog(n_ccd, result.instrument_spectrum.values / max(result.instrument_spectrum.values.max(), 1e-30),
              label="CCD-corrected (normalized)", lw=1.0)
    ax.set_xlabel("harmonic order n")
    ax.set_ylabel("normalized signal")
    ax.legend()
    return ax


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    p = Path(path)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    return p
