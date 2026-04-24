"""Tests for the MPI parallel dispatcher (gated behind HARMONY_TEST_MPI)."""

import os

import pytest

from harmonyemissions.parallel import MpiNotAvailable, run_scan_mpi


def test_mpinotavailable_raised_without_mpi4py(monkeypatch):
    """Simulate missing mpi4py: calling run_scan_mpi must raise our wrapped error."""

    def _fail_import(*_args, **_kwargs):
        raise ImportError("mocked")

    # Patch the private helper that performs the import.
    from harmonyemissions.parallel import mpi as mpi_mod
    monkeypatch.setattr(mpi_mod, "_get_mpi", lambda: (_ for _ in ()).throw(
        MpiNotAvailable("mpi4py not installed (mocked)")
    ))
    with pytest.raises(MpiNotAvailable):
        run_scan_mpi(base=None, grid=[], output_dir="/tmp/_none")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not os.environ.get("HARMONY_TEST_MPI"),
    reason="MPI scan requires HARMONY_TEST_MPI=1 and mpirun -n",
)
def test_mpi_scan_matches_serial(tmp_path):  # pragma: no cover - env-gated
    from pathlib import Path

    from harmonyemissions.config import load_config
    from harmonyemissions.scan import run_scan

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "scan_example.yaml")
    grid = [{"laser.a0": a} for a in [1.0, 3.0, 5.0, 10.0]]
    out_serial = tmp_path / "serial"
    out_mpi = tmp_path / "mpi"
    serial = run_scan(base, grid, output_dir=out_serial, n_jobs=1)
    mpi = run_scan_mpi(base, grid, output_dir=out_mpi, gather=True)
    assert mpi is not None
    assert len(mpi) == len(serial)
