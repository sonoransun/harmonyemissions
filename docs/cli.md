# CLI reference

The `harmony` command is installed as a console script by
[`pyproject.toml`](../pyproject.toml). It exposes four subcommands.

## `harmony validate`

```bash
harmony validate <config.yaml>
```

Loads the config, runs pydantic validation, and prints a one-line summary.
Non-zero exit code on any validation failure. Use it as a cheap dry-run
before launching an expensive PIC sweep.

## `harmony run`

```bash
harmony run <config.yaml> [-o|--output <path.h5>]
```

Executes a single simulation described by `<config.yaml>`, writes the
result to `<path.h5>` (defaulting to `run.h5` or the `output:` field of
the config), and prints the flat diagnostics dictionary.

## `harmony scan`

```bash
harmony scan <config.yaml> \
    -p|--param <dotted.path>=v1,v2,... \
    [-p ...] \
    [-d|--output-dir <runs/>] \
    [-j|--jobs N]
```

Cartesian-product parameter sweep over one or more knobs.

- `--param` is repeatable. Each instance takes one dotted path (e.g.
  `laser.a0`, `target.gradient_L_over_lambda`, `numerics.samples_per_period`)
  and a comma-separated list of values. Values are coerced to `int`,
  `float`, or `str` in that order.
- `--output-dir` is created if missing. Each run produces one HDF5 file
  named after its override tuple, e.g.
  `run_laser-a0=10_target-gradient_L_over_lambda=0.05.h5`.
- `--jobs` passes straight to `joblib.Parallel(n_jobs=...)`. Each run
  reloads state, so parallelism is safe up to your CPU budget.

Example — sweep a₀ over five values at two gradient scales, in parallel:

```bash
harmony scan configs/scan_example.yaml \
    -p laser.a0=1,2,5,10,20 \
    -p target.gradient_L_over_lambda=0.02,0.1 \
    -d runs/ -j 4
```

## `harmony plot`

```bash
harmony plot <path> [-k|--kind spectrum|pulse|scaling] [-o|--output <png>]
```

Plot a previously-saved run file.

- `spectrum` (default) — harmonic spectrum on a log–log axis with a
  least-squares power-law fit annotated.
- `pulse` — attosecond or reflected-field pulse in the time domain. Only
  meaningful for surface-HHG / gas-HHG runs that have a `time_field`.
- `scaling` — point `<path>` at a *directory* of scan outputs; the
  command plots cutoff harmonic vs the first swept parameter.

If `--output` is not given, the plot is saved next to the input with a
`.png` suffix.

## Exit codes

- `0` — success.
- `1` — generic error (validation, runtime, I/O).
- `2` — invalid CLI arguments (Typer default).

## See also

[`docs/gallery.md`](gallery.md) — a visual index showing every plot kind
(`spectrum`, `pulse`, `scaling`, `dent`, `beam`, `chf`, `instrument`)
with the exact CLI command that reproduces each figure.
