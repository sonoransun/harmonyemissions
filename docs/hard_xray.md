# Hard X-ray / γ-ray pipeline

Emission models up to `v0.3` topped out in the XUV (~100 eV). The
keV–MeV extension adds four source models, a shared filter library,
and three new detector-response bands.

## Source models

| Model             | Output band         | Regime          | Physics                                   |
|-------------------|---------------------|-----------------|-------------------------------------------|
| `bremsstrahlung`  | 10 keV – 10 MeV     | `overdense`     | Kramers + Wilks T_hot → exp1(E/T_hot)     |
| `kalpha`          | 1 – 30 keV          | `overdense`     | Casnati σ_K · ω_K, Kα₁/Kα₂/Kβ Lorentzians |
| `ics`             | 10 keV – tens MeV   | `underdense` / `electron_beam` | Esarey head-on + Klein-Nishina recoil |
| `betatron` (updated) | 100 eV – few MeV   | `underdense`    | Synchrotron envelope × χ_e quantum suppression |

All four carry the new non-dim coord `photon_energy_keV` on their
spectrum. You get it automatically:

```python
from harmonyemissions import Laser, Target, simulate
r = simulate(Laser(a0=2.0, wavelength_um=1.03),
             Target.underdense(0.001, electron_energy_mev=1000),
             model="betatron")
r.spectrum.coords["photon_energy_keV"].values    # goes up to ~MeV
```

## Hot-electron temperature

Bremsstrahlung and Kα derive their hot-electron temperature from
Wilks ponderomotive scaling by default:

```
T_hot [keV] = m_ec² · (√(1 + a₀²/2) − 1)
```

Override via `Target(hot_electron_temp_keV=250.0, ...)` if you have a
PIC-calibrated value. Alternative scaling via
`units.hot_electron_temperature_keV(a0, λ, scaling="beg")`.

## Detector bands

Output can be folded through four band-specific detector stacks,
dispatched automatically by the energy range of the source spectrum:

| Band         | Energy (eV)     | Stack                                                        |
|--------------|-----------------|--------------------------------------------------------------|
| `xuv`        | 10 – 100        | Al L-edge filter + Andor DV436 grating + soft-XUV CCD (legacy) |
| `xray-soft`  | 100 – 1000      | Mylar/Kapton/Al + Ni-Au grating + BI-CCD QE                  |
| `xray-hard`  | 1 000 – 100 000 | K-edge filters (Cu/Ni/Mo/Sn/Ag) + Si/CdTe absorber           |
| `gamma`      | 10⁵ – 10⁷       | LYSO/BGO/CsI scintillator + Compton / pair correction       |

```bash
# End-to-end LWFA betatron on HAPLS → γ-ray detector.
harmony run      configs/hapls_betatron.yaml -o betatron.h5
harmony detector betatron.h5 --band gamma --detector csi-20mm
harmony plot     betatron_detector.h5 -k instrument
```

Use `--filter cu-50um` (repeatable) to insert filters before the
detector; use `--ross-pair cu-50um,ni-55um` (TODO) for differential
K-edge spectroscopy.

## Adding a new filter or detector

- New filter material: drop a JSON table under
  `src/harmonyemissions/detector/data/attenuation/` following the
  schema in `detector/filters.py`. Re-run
  `scripts/build_attenuation_tables.py` to regenerate from anchors.
- New scintillator: add an entry in `detector/scintillator.py`
  (`_DETECTOR_DENSITY_G_CM3` + `_DETECTOR_PROXY_ELEMENT`).

See also
--------

- [`docs/presets.md`](presets.md) — DPSSL preset catalogue
- [`docs/theory.md`](theory.md) — physics derivations
- [`docs/instrument.md`](instrument.md) — XUV (legacy) detector pipeline
