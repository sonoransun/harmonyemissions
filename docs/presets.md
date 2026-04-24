# Laser facility presets

Harmony of Emissions ships a registry of high-power laser-driver
presets covering both Ti:Sapphire flagships and the new family of
diode-pumped solid-state lasers (DPSSL) built around Yb:YAG at 1030 nm.

## Using a preset

Set `laser.preset: <name>` in a YAML config. The preset fills any
fields the user omits; user-specified values always win.

```yaml
laser:
  preset: hapls            # λ=1.030 μm, τ=30 fs, a₀_ref=15
  spot_fwhm_um: 2.0        # experiment-specific, not in preset
  a0: 12.0                 # user overrides preset's a0_reference
```

Or list the catalogue from the CLI:

```bash
harmony list-presets
```

## Catalogue

| Name             | Facility                               | Gain medium         | λ (μm) | τ (fs)  | a₀ ref. | Rep rate |
|------------------|----------------------------------------|---------------------|--------|---------|---------|----------|
| `hapls`          | HAPLS / ELI-Beamlines L3-HAPLS         | Yb:YAG DPSSL        | 1.030  | 30      | 15      | 10 Hz    |
| `dipole`         | DiPOLE-100 / CLF-RAL                   | Yb:YAG multi-slab   | 1.030  | 500     | 5       | 10 Hz    |
| `bivoj`          | BIVOJ / ELI-Beamlines L2-BIVOJ         | Yb:YAG cryogenic    | 1.030  | 500     | 5       | 10 Hz    |
| `polaris`        | POLARIS / HI-Jena                      | Yb:glass DPSSL      | 1.030  | 98      | 10      | 0.1 Hz   |
| `apollon`        | Apollon / LULI-CNRS                    | Ti:Sapphire CPA     | 0.800  | 24      | 30      | 1 Hz     |
| `yb_yag_generic` | Generic Yb:YAG DPSSL                   | Yb:YAG              | 1.030  | 100     | 5       | 10 Hz    |

The `a0_reference` column is the facility's best-shot a₀ at typical
focusing; treat it as a ballpark. Override `laser.a0:` for specific
experimental configurations.

## Adding a new facility

1. Append an entry to `src/harmonyemissions/data/laser_presets.yaml`:
   ```yaml
   my_facility:
     facility: "MyLab 10 PW"
     gain_medium: "Ti:Sapphire"
     wavelength_um: 0.800
     duration_fs: 25.0
     a0_reference: 20.0
     polarization: p
     envelope: gaussian
     rep_rate_hz: 1.0
     citation: "MyLab team, Journal X (2026)"
   ```
2. That's it — the preset is now visible to `harmony list-presets`
   and usable as `laser.preset: my_facility` in any config.
