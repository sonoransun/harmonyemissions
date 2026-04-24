"""MPI-based parameter-sweep dispatcher.

Turn a ``run_scan`` grid into a multi-node job: each MPI rank takes the
slice ``grid[rank::size]`` of the Cartesian product, runs it
independently, and writes its HDF5 outputs to the shared ``output_dir``.
The calling rank (rank 0) optionally gathers the per-rank metadata back.

Runtime dependency on ``mpi4py`` is optional — absence raises
:class:`MpiNotAvailable`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from harmonyemissions.config import RunConfig
from harmonyemissions.scan import ScanPoint, _run_one


class MpiNotAvailable(RuntimeError):
    """Raised when mpi4py is unavailable or not running under mpirun."""


def _get_mpi():
    try:  # pragma: no cover - requires mpi4py at test time
        from mpi4py import MPI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MpiNotAvailable(
            "mpi4py is not installed. Install it (pip install mpi4py) and "
            "launch the scan under mpirun to use the MPI backend."
        ) from exc
    return MPI


def run_scan_mpi(
    base: RunConfig,
    grid: Iterable[dict[str, Any]],
    output_dir: str | Path,
    gather: bool = True,
) -> Sequence[ScanPoint] | None:
    """Distribute a scan across MPI ranks.

    Parameters
    ----------
    base : RunConfig
        Base config each point overrides.
    grid : iterable of dict
        Per-point override dicts (dotted-path → value).
    output_dir : path
        Shared directory; per-rank output files are written here.
    gather : bool
        When True (default), rank 0 gathers all :class:`ScanPoint` results
        and returns them; other ranks return ``None``. When False, every
        rank returns only its own slice (useful if callers don't want to
        pay the gather cost).
    """
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid_list = list(grid)
    my_slice = grid_list[rank::size]
    out_dir = Path(output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    local: list[ScanPoint] = [
        _run_one(copy.deepcopy(base), o, out_dir) for o in my_slice
    ]

    if not gather:
        return local

    collected = comm.gather(local, root=0)
    if rank != 0:
        return None
    out: list[ScanPoint] = []
    for chunk in collected or []:
        out.extend(chunk)
    return out
