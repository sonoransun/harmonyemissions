"""Regenerate the PNGs embedded in the Harmony of Emissions documentation.

Usage::

    python docs/generate_images.py

Writes 12 images into ``docs/images/``.  Orchestrates calls into the
library (no new plotting code) so every figure stays in sync with the
physics as it evolves.

Keep wall-clock small: the CHF panel uses a 64² grid, all other runs are
O(ms).  Total: ~10 s on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.config import load_config
from harmonyemissions.contrast import scale_length_from_thdr
from harmonyemissions.detector.deconvolve import DetectorConfig, apply_instrument_response
from harmonyemissions.runner import simulate_from_config
from harmonyemissions.viz import (
    plot_beam_profile,
    plot_chf_gain,
    plot_dent_map,
    plot_instrument_spectrum,
    plot_spectrum,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
CONFIGS = ROOT / "configs"


def _save(fig, name: str) -> Path:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def hero_and_pipeline_panels() -> None:
    """Full Gemini surface_pipeline run at 64² — reused across hero/dent/beam/gain."""
    cfg = load_config(CONFIGS / "chf_gemini.yaml")
    cfg = cfg.model_copy(
        update={
            "numerics": cfg.numerics.model_copy(update={
                "pipeline_grid": 64, "pipeline_dx_um": 0.15,
            })
        }
    )
    result = simulate_from_config(cfg)

    # --- hero: 2x2 grid -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    plot_spectrum(result, ax=axes[0, 0])
    axes[0, 0].set_title("Harmonic spectrum (Gemini, a₀ = 24)")
    plot_dent_map(result, ax=axes[0, 1])
    plot_beam_profile(result, which="near", ax=axes[1, 0])
    plot_chf_gain(result, ax=axes[1, 1])
    fig.suptitle("Harmony of Emissions — CHF pipeline output", fontsize=13)
    fig.tight_layout()
    _save(fig, "hero_chf_gemini.png")

    # --- individual panels for docs/ ----------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_dent_map(result, ax=ax)
    fig.tight_layout()
    _save(fig, "dent_map_gemini.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_beam_profile(result, which="far", harmonic_idx=0, ax=ax)
    fig.tight_layout()
    _save(fig, "beam_profile_far.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_chf_gain(result, ax=ax)
    fig.tight_layout()
    _save(fig, "chf_gain_bar.png")


def bgp_slope_figure() -> None:
    r = simulate(Laser(a0=30.0), Target.overdense(100.0), model="bgp")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectrum(r, ax=ax)
    ax.set_title("BGP universal spectrum — slope matches −8/3")
    fig.tight_layout()
    _save(fig, "spectrum_bgp_slope.png")


def rom_vs_cse_figure() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, a0, label in [("bgp", 30.0, "ROM / BGP  (−8/3)"),
                              ("cse", 10.0, "CSE  (−4/3)")]:
        r = simulate(Laser(a0=a0), Target.overdense(200.0, 0.05), model=model)
        n = r.spectrum.coords["harmonic"].values
        s = r.spectrum.values
        ax.loglog(n, s / s.max(), label=label, lw=1.2)
    ax.set_xlabel("harmonic order n")
    ax.set_ylabel("I(n) (normalised)")
    ax.set_title("Surface-HHG regimes: ROM vs CSE")
    ax.legend()
    fig.tight_layout()
    _save(fig, "spectrum_rom_vs_cse.png")


def lewenstein_figure() -> None:
    cfg = load_config(CONFIGS / "gas_hhg_default.yaml")
    r = simulate_from_config(cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectrum(r, ax=ax, show_fit=False)
    ax.axvline(r.diagnostics["n_cutoff_corkum"], color="red", ls="--", lw=0.9,
               label="Corkum cutoff")
    ax.set_title("Gas-phase HHG (Ar) — Lewenstein three-step")
    ax.legend()
    fig.tight_layout()
    _save(fig, "spectrum_lewenstein_corkum.png")


def betatron_figure() -> None:
    cfg = load_config(CONFIGS / "betatron_default.yaml")
    r = simulate_from_config(cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectrum(r, ax=ax, show_fit=False)
    ax.axvline(r.diagnostics["omega_c_over_omega0"], color="red", ls="--", lw=0.9,
               label="ω_c (synchrotron)")
    ax.set_title("LWFA betatron — synchrotron envelope")
    ax.legend()
    fig.tight_layout()
    _save(fig, "spectrum_betatron.png")


def cwe_figure() -> None:
    cfg = load_config(CONFIGS / "cwe_default.yaml")
    r = simulate_from_config(cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectrum(r, ax=ax, show_fit=False)
    ax.axvline(r.diagnostics["n_plasma_cutoff"], color="red", ls="--", lw=0.9,
               label="√(n_e/n_c)")
    ax.set_title("Coherent wake emission — sub-relativistic SHHG")
    ax.legend()
    fig.tight_layout()
    _save(fig, "spectrum_cwe.png")


def scaling_figure() -> None:
    """n_c ∝ a₀³ from the BGP model across five a₀ values."""
    a0s = np.array([3.0, 6.0, 10.0, 20.0, 40.0])
    n_c = np.array([
        simulate(Laser(a0=float(a)), Target.overdense(100.0), model="bgp")
        .diagnostics["n_cutoff"]
        for a in a0s
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(a0s, n_c, "o-", label="n_c (BGP)", lw=1.4)
    ref = n_c[0] * (a0s / a0s[0]) ** 3
    ax.loglog(a0s, ref, "--", label="γ³ ≈ a₀³ reference", lw=0.9)
    ax.set_xlabel("a₀")
    ax.set_ylabel("cutoff harmonic n_c")
    ax.set_title("BGP scaling law: n_c ∝ a₀³")
    ax.legend()
    fig.tight_layout()
    _save(fig, "scaling_nc_vs_a0.png")


def contrast_figure() -> None:
    t_hdr = np.linspace(100.0, 1500.0, 200)
    L = np.array([scale_length_from_thdr(t) for t in t_hdr])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_hdr, L, lw=1.4)
    for t_mark, tag in [(351.0, "Timmis 2026 optimum"), (711.0, "over-expanded")]:
        L_mark = scale_length_from_thdr(t_mark)
        ax.plot(t_mark, L_mark, "o")
        ax.annotate(f"{tag}\n  t_HDR = {t_mark:.0f} fs\n  L/λ ≈ {L_mark:.2f}",
                    (t_mark, L_mark), xytext=(8, 8), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("t_HDR [fs]")
    ax.set_ylabel("plasma scale length  L / λ")
    ax.set_title("DPM contrast model  L(t_HDR)")
    fig.tight_layout()
    _save(fig, "contrast_L_vs_thdr.png")


def instrument_figure() -> None:
    # Clean Gemini-like spectrum as the input.
    r = simulate(Laser(a0=24.0), Target.overdense(200.0, 0.05), model="bgp")
    cfg = DetectorConfig(al_thickness_um=1.5, include_second_order=True)
    ccd = apply_instrument_response(r.spectrum, 0.8, cfg)
    r.instrument_spectrum = ccd
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_instrument_spectrum(r, ax=ax)
    ax.set_title("Instrument response S(λ) applied to a simulated spectrum")
    fig.tight_layout()
    _save(fig, "instrument_response.png")


# ---------------------------------------------------------------------------
# Frequency-domain comparison panels (docs/comparison.md)
# ---------------------------------------------------------------------------


def _spectrum_energy_keV(result) -> tuple[np.ndarray, np.ndarray]:
    """Return (energy_keV, intensity) with the energy coord extracted either
    from a native ``photon_energy_keV`` coord or from the harmonic-order axis
    times hc/λ — works for every model."""
    from harmonyemissions.units import keV_per_harmonic
    spec = result.spectrum
    if "photon_energy_keV" in spec.coords:
        E = np.asarray(spec.coords["photon_energy_keV"].values, dtype=float)
    else:
        n = np.asarray(spec.coords["harmonic"].values, dtype=float)
        # Driver wavelength from the simulation provenance, default 800 nm.
        wl = float(result.provenance.get("laser", {}).get("wavelength_um", 0.8))
        E = n * keV_per_harmonic(wl)
    return E, np.asarray(spec.values, dtype=float)


def compare_sources_energy_axis() -> None:
    """Cross-source overlay on a shared photon-energy axis (0.1 eV – 10 MeV)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    runs = [
        ("BGP surface HHG",    simulate(Laser(a0=10.0), Target.overdense(200.0, 0.05), model="bgp")),
        ("ROM surface HHG",    simulate(Laser(a0=10.0), Target.overdense(200.0, 0.05), model="rom")),
        ("CSE nanobunching",   simulate(Laser(a0=15.0), Target.overdense(200.0, 0.01), model="cse")),
        ("CWE (sub-relativistic)", simulate(Laser(a0=0.3), Target.overdense(400.0, 0.05), model="cwe")),
        ("Gas HHG (Ar)",       simulate_from_config(load_config(CONFIGS / "gas_hhg_default.yaml"))),
        ("LWFA betatron",      simulate(Laser(a0=2.0),
                                         Target.underdense(0.001, electron_energy_mev=500.0),
                                         model="betatron")),
        ("Inverse Compton",    simulate(Laser(a0=0.3, spot_fwhm_um=5.0, duration_fs=30.0),
                                         Target.electron_beam(beam_energy_mev=500.0),
                                         model="ics")),
        ("Bremsstrahlung (hot-e)", simulate(Laser(a0=5.0), Target.overdense(100.0), model="bremsstrahlung")),
        ("Kα (Cu target)",     simulate(Laser(a0=5.0), Target.overdense(100.0, material="Cu"), model="kalpha")),
    ]
    for label, r in runs:
        E, S = _spectrum_energy_keV(r)
        mask = (E > 1e-4) & (S > 0)
        if not np.any(mask):
            continue
        S_norm = S[mask] / S[mask].max()
        ax.loglog(E[mask] * 1e3, S_norm, lw=1.1, label=label)  # keV → eV for axis
    ax.set_xlabel("photon energy [eV]")
    ax.set_ylabel("I / I_peak (per source)")
    ax.set_xlim(0.1, 1.0e7)
    ax.set_ylim(1e-6, 2.0)
    ax.set_title("Frequency-domain comparison: every emission source at matched driver conditions")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "compare_sources_energy_axis.png")


def compare_surface_regimes_harmonic() -> None:
    """Surface-HHG family on the harmonic-order axis at matched a₀ = 10."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, model, target in [
        ("ROM / BGP",          "rom", Target.overdense(200.0, 0.05)),
        ("BGP universal",      "bgp", Target.overdense(200.0, 0.05)),
        ("CSE nanobunching",   "cse", Target.overdense(200.0, 0.01)),
        ("CWE",                "cwe", Target.overdense(400.0, 0.05)),
    ]:
        r = simulate(Laser(a0=10.0), target, model=model)
        n = r.spectrum.coords["harmonic"].values
        s = r.spectrum.values / r.spectrum.values.max()
        ax.loglog(n, s, lw=1.2, label=label)
    # Slope rulers for the two universal predictions.
    n_ref = np.geomspace(3, 300, 50)
    for slope, name in [(-8.0 / 3.0, "n⁻⁸/³ (BGP)"), (-4.0 / 3.0, "n⁻⁴/³ (CSE)")]:
        ax.loglog(n_ref, (n_ref / n_ref[0]) ** slope, "k:", lw=0.7, alpha=0.5)
        ax.text(n_ref[-1] * 1.1, (n_ref[-1] / n_ref[0]) ** slope, name, fontsize=8)
    ax.set_xlabel("harmonic order n")
    ax.set_ylabel("I(n) / I_peak")
    ax.set_title("Surface-HHG regimes — same a₀ = 10, normalised")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "compare_surface_regimes_harmonic.png")


def compare_gamma_sources_keV() -> None:
    """Hard-X-ray → γ family on a shared keV axis with cutoffs annotated."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    runs = [
        ("Betatron (500 MeV)",  simulate(Laser(a0=2.0),
                                          Target.underdense(0.001, electron_energy_mev=500.0),
                                          model="betatron"),  "photon_energy_keV_critical"),
        ("ICS (500 MeV beam)",  simulate(Laser(a0=0.3, spot_fwhm_um=5.0, duration_fs=30.0),
                                          Target.electron_beam(beam_energy_mev=500.0),
                                          model="ics"),       "photon_energy_keV_cutoff"),
        ("Bremsstrahlung (Wilks)", simulate(Laser(a0=5.0), Target.overdense(100.0),
                                              model="bremsstrahlung"), "photon_energy_keV_efolding"),
        ("Kα (Ag target)",      simulate(Laser(a0=5.0), Target.overdense(100.0, material="Ag"),
                                          model="kalpha"),    "K_alpha1_keV"),
    ]
    colours = ["C0", "C1", "C2", "C3"]
    for (label, r, tag), col in zip(runs, colours, strict=True):
        E, S = _spectrum_energy_keV(r)
        mask = (E > 0) & (S > 0)
        ax.loglog(E[mask], S[mask] / S[mask].max(), lw=1.2, color=col, label=label)
        if tag in r.diagnostics:
            ax.axvline(r.diagnostics[tag], color=col, ls="--", lw=0.7, alpha=0.6)
    ax.set_xlabel("photon energy [keV]")
    ax.set_ylabel("I / I_peak (per source)")
    ax.set_xlim(0.5, 1.0e4)
    ax.set_title("Hard-X-ray → γ-band sources (dashed = diagnostic cutoff)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "compare_gamma_sources_keV.png")


def sweep_a0_surface() -> None:
    """a₀ sweep for ROM with the −8/3 reference slope."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), sharex=True, sharey=True)
    n_ref = np.geomspace(3, 500, 50)
    slope_line = (n_ref / n_ref[0]) ** (-8.0 / 3.0)
    for ax, a0 in zip(axes.ravel(), [3.0, 10.0, 24.0, 60.0], strict=True):
        r = simulate(Laser(a0=a0), Target.overdense(200.0, 0.05), model="rom")
        n = r.spectrum.coords["harmonic"].values
        s = r.spectrum.values / r.spectrum.values.max()
        ax.loglog(n, s, lw=1.2)
        ax.loglog(n_ref, slope_line / slope_line[0] * s[int(np.searchsorted(n, 3))],
                  "k:", lw=0.8, label="n⁻⁸/³")
        nc = r.diagnostics.get("n_cutoff", None)
        if nc is not None:
            ax.axvline(nc, color="r", ls="--", lw=0.7)
            ax.text(nc * 1.1, 1e-4, f"n_c ≈ {nc:.0f}", fontsize=8, color="r")
        ax.set_title(f"a₀ = {a0:g}")
        ax.grid(True, which="both", alpha=0.25)
    axes[1, 0].set_xlabel("harmonic order n")
    axes[1, 1].set_xlabel("harmonic order n")
    axes[0, 0].set_ylabel("I / I_peak")
    axes[1, 0].set_ylabel("I / I_peak")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("ROM spectrum vs driver a₀ — cutoff n_c ∝ γ³ ≈ a₀³", fontsize=11)
    fig.tight_layout()
    _save(fig, "sweep_a0_surface.png")


def sweep_gamma_betatron_ics() -> None:
    """Betatron and ICS γ-sweeps with χ_e annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Betatron
    ax = axes[0]
    for E_mev in [500.0, 1000.0, 2000.0]:
        r = simulate(Laser(a0=2.0),
                     Target.underdense(0.001, electron_energy_mev=E_mev),
                     model="betatron")
        E, S = _spectrum_energy_keV(r)
        mask = (E > 0) & (S > 0)
        chi = r.diagnostics["chi_e"]
        ax.loglog(E[mask], S[mask] / S[mask].max(), lw=1.2,
                  label=f"γE = {E_mev:.0f} MeV  (χ_e = {chi:.2e})")
    ax.set_xlabel("photon energy [keV]")
    ax.set_ylabel("I / I_peak")
    ax.set_title("Betatron vs electron energy")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    # ICS
    ax = axes[1]
    for E_mev in [100.0, 500.0, 1000.0]:
        r = simulate(Laser(a0=0.3, spot_fwhm_um=5.0, duration_fs=30.0),
                     Target.electron_beam(beam_energy_mev=E_mev),
                     model="ics")
        E, S = _spectrum_energy_keV(r)
        mask = (E > 0) & (S > 0)
        chi = r.diagnostics["chi_e_nominal"]
        ax.loglog(E[mask], S[mask] / S[mask].max(), lw=1.2,
                  label=f"beam = {E_mev:.0f} MeV  (χ_e = {chi:.2e})")
    ax.set_xlabel("photon energy [keV]")
    ax.set_ylabel("I / I_peak")
    ax.set_title("Inverse Compton vs beam energy")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("γ-band parametric sweeps — higher γ → harder photons (quantum χ suppression at χ ≳ 0.1)",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "sweep_gamma_betatron_ics.png")


def sweep_kalpha_materials() -> None:
    """Kα + Kβ lines for five materials on one keV axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for material in ["Al", "Ti", "Fe", "Cu", "Mo", "Ag"]:
        r = simulate(Laser(a0=5.0), Target.overdense(100.0, material=material),
                     model="kalpha")
        E, S = _spectrum_energy_keV(r)
        mask = (E > 0) & (S > 0)
        ax.semilogy(E[mask], S[mask] / S[mask].max(), lw=1.0, label=material)
        E_ka = r.diagnostics.get("K_alpha1_keV") or r.diagnostics.get("K_alpha_keV")
        if E_ka:
            ax.axvline(E_ka, color="k", ls=":", lw=0.4, alpha=0.3)
    ax.set_xlim(1.0, 90.0)
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("photon energy [keV]")
    ax.set_ylabel("line intensity (norm.)")
    ax.set_title("Kα / Kβ fluorescence — target material tunes the line position")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "sweep_kalpha_materials.png")


FIGURES = [
    hero_and_pipeline_panels,
    bgp_slope_figure,
    rom_vs_cse_figure,
    lewenstein_figure,
    betatron_figure,
    cwe_figure,
    scaling_figure,
    contrast_figure,
    instrument_figure,
    # Frequency-domain comparisons
    compare_sources_energy_axis,
    compare_surface_regimes_harmonic,
    compare_gamma_sources_keV,
    sweep_a0_surface,
    sweep_gamma_betatron_ics,
    sweep_kalpha_materials,
]


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn()
    print(f"\n{len(list(IMAGES.glob('*.png')))} PNG(s) in {IMAGES}")


if __name__ == "__main__":
    main()
