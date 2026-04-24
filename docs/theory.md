# Theory: high-frequency emission from relativistic laser-plasma interactions

Harmony of Emissions bundles four complementary techniques for turning a
strong-field laser pulse into coherent high-frequency light, plus a
spatiotemporal compression stage (Coherent Harmonic Focus, CHF) that
unifies them. Each regime has its own physical context, scaling laws,
and tuning knobs. This document collects the derivations in one place so
that the code is easier to read and the diagnostics easier to interpret.

The CHF pipeline and most of its constituents are direct implementations
of Timmis et al., *Nature* (2026), DOI
[10.1038/s41586-026-10400-2](https://doi.org/10.1038/s41586-026-10400-2)
— see [chf.md](chf.md) for the high-level walkthrough.

```mermaid
flowchart TD
    A[Laser-plasma interaction]
    A --> S1[Surface HHG — overdense]
    A --> G1[Gas HHG]
    A --> U1[LWFA — underdense]

    S1 --> ROM[ROM<br/>I ∝ n⁻⁸/³, n_c ∝ γ³]
    S1 --> BGP[BGP universal<br/>analytical envelope]
    S1 --> CSE[CSE nanobunching<br/>I ∝ ω⁻⁴/³]
    S1 --> CWE[CWE<br/>cutoff at √(n_e/n_c)]
    S1 --> SP[surface_pipeline<br/>+ CHF focusing]

    G1 --> LEW[Lewenstein 3-step<br/>ħω_max = I_p + 3.17 U_p]
    U1 --> BET[Betatron<br/>ω_c = 3/2·γ³·ω_β²·r_β/c]
```

## Normalized units

We work in the standard normalized laser-plasma unit system:

- time in units of the laser period T₀ = λ/c,
- length in units of the wavelength λ,
- electromagnetic field in units of the normalized vector potential
  a = eE/(mₑωc),
- density in units of the critical density n_c(ω) = ε₀ mₑ ω² / e².

The dimensionless peak amplitude `a₀` is the single most important
parameter: the interaction is *relativistic* once a₀ ≳ 1. For a linearly
polarized wave, a single electron reaches a peak Lorentz factor

    γ_e = √(1 + a₀²/2).

At 800 nm, a₀ = 1 corresponds to a peak intensity of about
2.14 × 10¹⁸ W/cm².

Relevant unit helpers live in
[`harmonyemissions.units`](../src/harmonyemissions/units.py).

## Regime 1 — Surface HHG on overdense plasma (ROM + BGP + CSE)

When a high-intensity laser hits a sharp, overdense plasma surface
(density n ≫ n_c, gradient length L ≪ λ), the surface acts as a
relativistically oscillating mirror (ROM). The reflected field is
Doppler-compressed because the surface oscillates along the laser
direction at v_s(t), producing harmonics far above the driving frequency.

### The reduced-mirror model

In a fully self-consistent picture the plasma surface dynamics solve the
coupled Vlasov–Maxwell system. Bulanov et al. (1994) showed that a single
ODE captures the essential physics:

    m γ³ d²x_s/dt² = F_drive(t, x_s) − F_restore(x_s)

where F_drive is the ponderomotive push and F_restore is a linear
restoring force proportional to the local plasma stiffness. Gonoskov et
al. (2011) refined this reduction and showed when it is quantitatively
reliable.

`harmonyemissions.models.rom.ROMModel` uses a parametric variant of the
reduced mirror: the surface velocity is taken to follow the laser drive
with a tanh-compressed, relativistically-saturated envelope

    β_s(t) = β_peak · env²(t) · tanh(K sin(2π t + φ_CEP))

with β_peak set such that γ_peak ≈ a₀ · (gradient factor) · (density factor)
and K chosen to make the turnaround sharper at higher a₀. This avoids the
numerical pathologies of the raw ODE (surface velocity saturating to c)
while preserving the Doppler-compression physics that matters for
attosecond-pulse synthesis. The reflected field is then computed via the
retarded-time map

    E_r(t) = E_i(t_r),    t_r = t − 2 x_s(t_r).

### The BGP universal spectrum

Baeva, Gordienko & Pukhov (2006) showed that in the ultra-relativistic
limit the spectrum of a ROM source has a *universal* form:

    I(n) ∝ n^(−8/3)   up to   n_c ≃ 4√(2α) γ_max³,

with α the fine-structure constant. The ROM model in this library uses
this BGP envelope as its spectrum (anchored by the mirror γ_peak), and
reserves the time-domain reflected field for attosecond-pulse synthesis.
That decouples spectral-analysis accuracy from FFT sample-rate limits.

`harmonyemissions.models.bgp.BGPModel` exposes the closed form directly
for fast scans and as a regression target for the ROM spectrum.

### The CSE regime

An der Brügge & Pukhov (2010) pointed out that for some configurations
the plasma surface ejects a thin, dense electron nanobunch once per cycle;
this bunch radiates synchrotron-like spikes with a different scaling:

    I(ω) ∝ ω^(−4/3),   cutoff n_c ∝ γ³.

`harmonyemissions.models.cse.CSEModel` produces the nanobunching envelope;
experiment suggests CSE dominates over BGP when the gradient is very
short (L/λ ≲ 0.02) and the amplitude is very high (a₀ ≳ 10).

![ROM vs CSE](images/spectrum_rom_vs_cse.png)

*Side-by-side envelopes: ROM/BGP at a₀=30 (universal −8/3 plateau) vs
CSE at a₀=10 with L/λ=0.05 (shallower −4/3 slope, sharper spikes).*

## Regime 2 — Gas-phase HHG (Lewenstein / Corkum)

For atomic-gas targets driven at moderate intensity (10¹³–10¹⁵ W/cm²,
a₀ ≪ 1) the relevant physics is the *three-step model* of Corkum (1993):

1. **Ionization.** The bound electron tunnels through the suppressed
   atomic potential, with ADK rate
   w(t) ∝ exp(−2(2 I_p)^{3/2} / (3|E|)).
2. **Propagation.** The free electron is accelerated classically in the
   laser field, gains kinetic energy, and is accelerated back toward the
   parent ion.
3. **Recombination.** On return, it recombines and emits a photon of
   energy ħω = I_p + E_kin, where E_kin is the kinetic energy at
   recollision.

The classical cutoff follows from the maximum recollision energy
E_kin^max = 3.17 U_p:

    ħω_max = I_p + 3.17 U_p,

with U_p the ponderomotive energy U_p [eV] ≈ 9.33 × 10⁻¹⁴ · I[W/cm²] · (λ[μm])².

Lewenstein, Balcou, Ivanov, L'Huillier & Corkum (1994) turned the three
steps into a dipole integral that can be evaluated in the strong-field
approximation. `harmonyemissions.models.lewenstein.LewensteinModel` does
this classically — summing short-trajectory returns weighted by ADK rate
and phase — which reproduces the Corkum cutoff and an approximate
plateau. The library's implementation is simplified relative to the full
complex-trajectory Lewenstein integral; the tradeoff is clarity and
reproducibility rather than photometric accuracy.

![Argon HHG spectrum](images/spectrum_lewenstein_corkum.png)

*Argon HHG at a₀ = 0.08 — the dashed red line marks the Corkum cutoff
ħω_max = I_p + 3.17 U_p.*

## Regime 3 — Laser-wakefield betatron

A different route to high-frequency light is the betatron source in a
laser-wakefield accelerator. Self-injected electrons accelerate to GeV
energies inside the ion bubble of an underdense plasma (n_e ≪ n_c) and,
thanks to the linear transverse focusing, oscillate at the betatron
frequency

    ω_β = ω_p / √(2γ),

where ω_p is the background plasma frequency and γ the instantaneous
Lorentz factor. Kostyukov, Pukhov & Kiselev (2004) computed the
synchrotron-like spectrum of this radiation in closed form; in the
"wiggler regime" K = γ k_β r_β ≫ 1 it reduces to the usual synchrotron
envelope with critical frequency

    ω_c = (3/2) γ³ ω_β² r_β / c = (3/4) γ² K ω_β.

For an e-beam at 500 MeV with a 1 μm betatron amplitude in a plasma of
10⁻³ n_c, this puts the photon cutoff in the 10 keV range.

`harmonyemissions.models.betatron.BetatronModel` returns the synchrotron
envelope I(ω) ∝ ξ K_{2/3}(ξ/2)² (with ξ = ω/ω_c) — correct at both
asymptotic limits — expressed in harmonic-of-ω₀ units for consistency
with the surface-HHG regimes.

![Betatron synchrotron envelope](images/spectrum_betatron.png)

*Betatron spectrum for a 500 MeV electron in 10⁻³ n_c plasma with a 1 μm
oscillation amplitude — the envelope peaks below ω_c and drops
exponentially above.*

## Regime 4 — Coherent Harmonic Focus (CHF) on overdense plasma

Reference: Timmis et al., *Nature* (2026); Gordienko et al., *Phys. Rev.
Lett.* **94**, 103903 (2005); Baeva et al., *Phys. Rev. E* **74**, 046404
(2006); Vincenti et al., *Nat. Commun.* **5**, 3403 (2014).

CHF turns surface-HHG from a spectroscopy demonstration into a
**field-intensity engineering** tool. The reflected harmonic beam is
spatiotemporally compressed into a focus whose peak intensity exceeds
the driver's by 10²–10⁴×, approaching the Schwinger limit (2.3 × 10²⁹
W/cm²) for 50 PW-class laser systems.

### Three ingredients

1. Diffraction-limited focusing: λ_n = λ/n → spot area ~ (λ/n)².
2. Attosecond phase locking between harmonics.
3. Slow efficiency decay η_n ∝ n^(−8/3) (BGP universal spectrum), so
   combined with the n² area reduction the field strength falls only as
   n^(−1/3) — adjacent harmonics contribute comparably to the CHF.

### Analytical pipeline (Timmis 2026, Methods §"Analytical model of XUV
beam profiles")

Relativistic-spikes spatial filter (ref. 10 / Gordienko 2005):

    S(n, a₀(x',y'))         # sharp roll-off at n_c ∝ a₀³

Fraunhofer near-to-far propagation:

    U(x, y, z) ∝ F{U₀(x', y')}|_{f_x = x/λz, f_y = y/λz}       (eq. 7)

Per-harmonic driver near-field:

    U₀(x', y', n) ∝ √S(n, a₀(x', y')) · U₀(x', y')              (eq. 8)

Ion-surface dent from ponderomotive pressure on an exponential
preplasma of scale length L:

    Δz_i(x', y') = 2 L · ln(1 + (Π₀ / 2 L cos θ) · ∫_{-∞}^{t_p} a_L dt')  (eq. 10)

with Π₀ = √(R · Z · m_e · cos θ / (2 A M_p)).

Phase imprint:

    φ_n = 2 k_n Δz cos θ                                        (eq. 12 phase)

Full far-field intensity of the nth harmonic:

    I(x, y, z, n) ∝ |F{√S U₀ exp(−2 i k_n Δz cos θ)}|²          (eq. 12)

### 3D gain extrapolation

The paper extrapolates 2D PIC to 3D under the axisymmetric approximation:

    Γ_D    = I₀ / I_L                      # temporal compression
    Γ_2D   = I_f / I₀                      # 2D spatial compression
    Γ_3D   = Γ_2D²                         # 3D extrapolation
    Γ      = Γ_D · Γ_3D                    # total

and the empirical scaling I_CHF / I ∝ a₀³, anchored at the Gemini
calibration (a₀ ≈ 24, Γ ≳ 80).

The relativistic-spikes cutoff — the single most important scaling in
the CHF pipeline — is a short dependency chain:

```mermaid
flowchart LR
    a0[a₀\neE/(mₑωc)] --> g[γ_e = √1+a₀²/2]
    g --> nc[n_c ≈ 4√2α · a₀³]
    nc --> S["S(n, a₀(x', y'))<br/>logistic / exp / sharp"]
    S --> Ichf[I_CHF / I ∝ a₀³]
```

![a₀³ scaling](images/scaling_nc_vs_a0.png)

*BGP cutoff harmonic vs driver a₀ at 800 nm — the scan (circles) tracks
the γ³ ≈ a₀³ reference (dashed) across 1.5 decades.*

### Where it lives in the code

- `harmonyemissions.beam` — U₀ builder + Fraunhofer propagation.
- `harmonyemissions.emission.spikes` — S(n, a₀) filter.
- `harmonyemissions.contrast` — t_HDR + prepulse → L/λ.
- `harmonyemissions.surface.denting` — Δz(x', y') from eqs. 10–11.
- `harmonyemissions.chf.propagation` — per-harmonic near/far field.
- `harmonyemissions.chf.gain` — Γ breakdown, a₀³ scaling.
- `harmonyemissions.models.surface_pipeline` — the orchestrator.

## When to use which model

| Situation                                               | Start with           |
|---------------------------------------------------------|----------------------|
| Sharp solid-density target, PW-class laser              | `rom`                |
| Quick analytical scan over laser intensity              | `bgp`                |
| Ultra-short pre-plasma gradient, nanobunching           | `cse`                |
| Gas jet, high repetition rate                           | `lewenstein`         |
| Underdense plasma, electrons self-inject, X-rays        | `betatron`           |
| Sub-relativistic surface HHG (a₀ ≲ 1)                   | `cwe`                |
| CHF, denting, beam-profile effects — Timmis 2026 study  | `surface_pipeline`   |
| PIC-level fidelity on surface HHG                       | `surface_pipeline` + SMILEI backend |

For side-by-side overlays and an interactive decision matrix that spans
every model listed above, see [`comparison.md`](comparison.md) and the
interactive notebook `examples/11_source_comparison.ipynb`.

## References

- T. Baeva, S. Gordienko, A. Pukhov, *Phys. Rev. E* **74**, 046404 (2006).
- D. an der Brügge, A. Pukhov, *Phys. Plasmas* **17**, 033110 (2010).
- S. V. Bulanov et al., *Phys. Plasmas* **1**, 745 (1994).
- P. Corkum, *Phys. Rev. Lett.* **71**, 1994 (1993).
- A. Gonoskov et al., *Phys. Rev. E* **84**, 046403 (2011).
- I. Kostyukov, A. Pukhov, S. Kiselev, *Phys. Plasmas* **11**, 5256 (2004).
- M. Lewenstein et al., *Phys. Rev. A* **49**, 2117 (1994).
- S. Corde et al., *Rev. Mod. Phys.* **85**, 1 (2013).
