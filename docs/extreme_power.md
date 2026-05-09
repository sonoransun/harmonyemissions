# Extreme-Power mode

The **Extreme-Power overlay** is the second of two new modes introduced
alongside chf3d Phase C. Where Phase C combines `N` *homogeneous*
phase-locked drivers on the existing `LaserArrayConfig` schema, the
Extreme-Power overlay extends the same coherent kernel with three
additional capabilities:

1. **Heterogeneous beams** — each driver carries its own
   `LaserConfig` (different λ, τ, a₀, profile). Models combining e.g.
   Apollon + ELI-NP + SEL beamlines, multi-color pump-probe arrangements,
   or arrays with deliberately mis-tuned drivers.
2. **Landau–Lifshitz radiation-reaction derate** — at extreme intensities
   the classical γ³ harmonic cutoff is reduced by radiation friction. The
   overlay reshapes the spikes envelope per-beam (Bulanov, Phys. Rev. E
   84 056605, 2011; Di Piazza, RMP 84 1177, 2012 §IV.A).
3. **Perturbative QED diagnostics** — Schwinger ratio χ = E/E_S,
   vacuum-birefringence phase shift Δφ (Heisenberg & Euler, *Z. Phys.*
   1936), and Breit–Wheeler pair-production rate density at the post-CHF
   focal-volume peak. Diagnostic-only (not self-consistent QED).

The overlay is opt-in: a multi-beam config without an `extreme_power:`
block runs Phase C unchanged.

## When to use Extreme-Power vs Phase C

| Question | Use |
|---|---|
| Coherent N-beam combining of homogeneous drivers | **Phase C** (`laser_array:` only) |
| Need to combine drivers from facilities with different λ / τ / a₀ | **Extreme-Power** (heterogeneous beams) |
| Post-CHF intensity is approaching ≳ 10²⁴ W/cm² and you need RR | **Extreme-Power** (`enable_radiation_reaction: true`) |
| Want χ, vacuum birefringence, BW pair rate as diagnostics | **Extreme-Power** (`enable_qed: true`) |

If your run is single-beam, neither mode applies — the legacy
`surface_pipeline` remains the right path.

## Schema

```yaml
extreme_power:
  per_beam_lasers:                # Optional — heterogeneous beams
    - {a0: 130.0, wavelength_um: 0.80, duration_fs: 15.0, polarization: p}
    - {a0:  80.0, wavelength_um: 0.82, duration_fs: 25.0, polarization: p}
    # … one entry per driver, length must equal laser_array.n_beams
  enable_radiation_reaction: false   # Landau–Lifshitz derate of harmonic cutoff
  enable_qed: false                  # Schwinger / birefringence / BW diagnostics
  rr_a0_floor: 50.0                  # below this a₀, derate is forced to 1.0
  rr_clip_chi: 2.0                   # χ above this clips + flags rr_clipped
  qed_chi_warn: 0.5                  # χ above this populates provenance warning
  omega_grid_points: 4096            # ω-axis resolution for heterogeneous sum
  omega_grid_pad: 1.05               # multiplicative padding around per-beam ranges
```

`extreme_power:` requires `laser_array:` to be set (Phase C must run
underneath) and `model: surface_pipeline` (only overdense-target chf3d
runs are supported).

## Heterogeneous frequency axis

Phase C's coherent sum uses harmonic order `n` as a shared axis because
all beams share λ. With heterogeneous λ, harmonic `n_i` of beam `i`
sits at physical angular frequency `ω_i = n_i · 2πc / λ_i`, so the
shared axis must be the *physical* ω. The overlay:

1. Builds a log-spaced ω-grid from the union of per-beam ranges
   (`build_omega_grid`).
2. Resamples each beam's on-axis far-field complex amplitude onto the
   common grid (`resample_to_omega`).
3. Performs the coherent sum on ω, with phase factors
   `exp(i k_ω n̂_i · r)` using `k_ω = ω/c`.

When all drivers share λ the homogeneous (Phase C) path is taken
automatically — the dispatcher checks `per_beam_lasers` for wavelength
heterogeneity and falls back to the cheaper harmonic-axis sum when it
isn't required.

## Landau–Lifshitz radiation-reaction derate

The classical BGP cutoff `n_c ∝ γ³` ignores radiation friction. The
Landau–Lifshitz correction caps γ at a saturation value set by the
radiation-reaction parameter `R_χ = (a₀/a_RR)³ · γ₀` where
`a_RR ≈ 3300 · (λ/μm)` (Di Piazza RMP 2012 §IV.A). The implementation
returns

```
derate_γ = 1 / (1 + R_χ)^(1/3)
derate_n_c = derate_γ³
```

Below `rr_a0_floor` the function returns 1.0 (RR is sub-percent and not
worth applying). The spectrum reshape is a smooth tanh roll-off centred
on `derate · n_c`, which preserves the BGP plateau and only modifies
the cutoff tail.

`Result.radiation_reaction` carries per-beam metadata:
`a0`, `wavelength_um`, `chi_e`, `derate`, `clipped`. The provenance
gains `rr_clipped: true` if any beam saturates at `rr_clip_chi`.

## QED diagnostics

Three numeric diagnostics are computed at the focal-volume peak intensity
`I_focus`:

- **Schwinger ratio** `χ = E_focus / E_S` with `E_focus = √(2 I_focus / ε₀ c)`.
- **Vacuum birefringence** `Δφ = (α/15π)·χ²·(ωL/c)` for a probe of
  angular frequency ω over length `L = π·(FWHM/2)²/λ_eff`
  (Rayleigh-length proxy).
- **Breit–Wheeler pair rate** `R ∝ E_S²·χ²·exp(-π/χ)` per unit
  volume per second; clipped to zero when `exp(-π/χ)` underflows.

`Result.qed_diagnostics` returns a numeric-only dict including
`schwinger_ratio`, `vacuum_birefringence_phase_shift_rad`,
`breit_wheeler_pair_rate_per_m3_per_s`, `field_V_per_m`, plus a
`validity_exceeded: bool` flag. When χ > `qed_chi_warn` the provenance
gains `qed_validity_warning: "<description>"` so downstream consumers
see the message without it polluting the numeric attrs.

## What this mode does NOT do

- **No pair-cascade backreaction.** The Breit–Wheeler rate is reported as
  a diagnostic; produced pairs are not propagated and do not feed back
  into the plasma model.
- **No vacuum-birefringence beyond perturbative leading order.** The
  Heisenberg–Euler expansion is truncated at first non-trivial order
  in α (E/E_S)², appropriate for χ ≲ 1.
- **No plasma-density backreaction from radiation-reaction.** RR derates
  the harmonic cutoff via γ-saturation; it does not modify the dent
  map, scale length, or attosecond-pulse synthesis.
- **No PIC fidelity.** This remains an analytical / parametric model.
  For PIC-grounded RR + QED, route through SMILEI's QED module
  (still WIP — `docs/backends.md`).

## Worked example — heterogeneous SEL + ELI-NP

`configs/extreme_power_combined.yaml` sets up an 8-beam cubic geometry
combining 4 SEL drivers (λ = 800 nm, a₀ = 130) on `±x/±y` faces with
4 ELI-NP drivers (λ = 820 nm, a₀ = 80) on `±z`. Both RR and QED are on.
Run + plot:

```bash
harmony validate configs/extreme_power_combined.yaml
harmony run      configs/extreme_power_combined.yaml -o /tmp/ep.h5
harmony plot     /tmp/ep.h5 -k chf
```

The notebook `examples/14_extreme_power_combined.ipynb` walks through
the Schwinger-ratio progression: single-beam SEL → 12-beam SEL chf3d →
8-beam heterogeneous SEL+ELI-NP, including the heterogeneous spectrum
on the ω-axis.

## Cross-references

- [`docs/chf3d.md`](chf3d.md) — Phase C kernel that this overlay sits on.
- [`docs/combined_power.md`](combined_power.md) — coherent-within-regime
  vs incoherent-across-regime decision matrix.
- [`docs/theory.md`](theory.md) — radiation-reaction / QED context for
  the analytical models.
