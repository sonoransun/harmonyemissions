"""I/O helpers for persisting and re-loading Result files.

Most of the heavy lifting lives on :class:`harmonyemissions.models.base.Result`
directly (``.save`` / ``.load``). This module adds convenience wrappers used
by the scan orchestrator and the CLI.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from harmonyemissions.models.base import Result


def save_result(result: Result, path: str | Path) -> Path:
    """Save a Result to HDF5; returns the resolved path."""
    return result.save(path)


def load_result(path: str | Path) -> Result:
    """Load a Result from HDF5."""
    return Result.load(path)


def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a run file as a raw xarray.Dataset without reconstructing a Result."""
    return xr.open_dataset(path, engine="h5netcdf")
