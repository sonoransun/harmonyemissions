"""Parameter sweeps.

A scan takes a base :class:`RunConfig` plus one or more parameters to vary,
and produces a list of (param_value, Result) pairs — optionally written to
disk as individual HDF5 files. joblib gives trivial CPU-level parallelism
without pulling in dask.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

from harmonyemissions.config import RunConfig
from harmonyemissions.models.base import Result
from harmonyemissions.runner import simulate_from_config


@dataclass
class ScanPoint:
    """A single point in a parameter scan."""

    overrides: dict[str, Any]  # dotted-path → value
    result: Result
    path: Path | None = None


def _apply_override(cfg_dict: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg_dict
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _run_one(
    base_cfg: RunConfig,
    overrides: dict[str, Any],
    output_dir: Path | None,
) -> ScanPoint:
    cfg_dict = base_cfg.model_dump()
    for k, v in overrides.items():
        _apply_override(cfg_dict, k, v)
    cfg = RunConfig.model_validate(cfg_dict)
    result = simulate_from_config(cfg)
    path = None
    if output_dir is not None:
        tag = "_".join(f"{k.replace('.', '-')}={v}" for k, v in overrides.items())
        path = output_dir / f"run_{tag}.h5"
        result.save(path)
    return ScanPoint(overrides=overrides, result=result, path=path)


def run_scan(
    base: RunConfig,
    grid: Iterable[dict[str, Any]],
    output_dir: str | Path | None = None,
    n_jobs: int = 1,
) -> list[ScanPoint]:
    """Run a scan across a grid of override dicts.

    Example::

        grid = [{"laser.a0": a0} for a0 in [1, 2, 5, 10, 20]]
        points = run_scan(base_cfg, grid, output_dir="runs/", n_jobs=4)
    """
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    grid_list = list(grid)
    return Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(copy.deepcopy(base), o, out_dir) for o in grid_list
    )


def parse_param_spec(spec: str) -> tuple[str, list[float]]:
    """Parse a ``'path=v1,v2,v3'`` CLI spec into (path, [values])."""
    if "=" not in spec:
        raise ValueError(f"Expected 'path=v1,v2,...' got {spec!r}")
    path, values = spec.split("=", 1)
    return path.strip(), [_coerce(v.strip()) for v in values.split(",") if v.strip()]


def _coerce(v: str) -> Any:
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            continue
    return v
