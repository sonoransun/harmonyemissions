"""EPOCH PIC backend (adapter stub).

Like :mod:`harmonyemissions.backends.smilei` but targeting the EPOCH PIC
code. EPOCH uses a Fortran-flavored ``input.deck`` format and writes SDF
output files, so the render/parse paths differ. Both backends expose the
same :meth:`simulate` interface.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target


class EpochNotAvailable(RuntimeError):
    """Raised when the EPOCH executable cannot be located on PATH."""


@dataclass
class EpochBackend:
    name: str = "epoch"
    executable: str = "epoch1d"

    def simulate(self, laser: Laser, target: Target, model: str, numerics) -> Result:
        if shutil.which(self.executable) is None:
            raise EpochNotAvailable(
                f"Could not find {self.executable!r} on PATH. "
                "Install EPOCH (https://cfsa-pmw.warwick.ac.uk/EPOCH) and retry."
            )
        if model not in {"rom", "cse"}:
            raise NotImplementedError(
                f"EPOCH backend only wired for surface HHG (rom/cse); got {model!r}."
            )
        with tempfile.TemporaryDirectory(prefix="harmony-epoch-") as td:
            td_path = Path(td)
            (td_path / "input.deck").write_text(self._render_deck(laser, target))
            (td_path / "deck.file").write_text(str(td_path) + "\n")
            r = subprocess.run([self.executable], input=str(td_path).encode(),
                               cwd=td_path, capture_output=True, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"EPOCH exited {r.returncode}: {r.stderr.decode()}")
            return self._parse(td_path, laser)

    def _render_deck(self, laser: Laser, target: Target) -> str:
        return dedent(f"""
            begin:control
              nx = {256 * 40}
              t_end = {laser.duration_fs * 5e-15}
              x_min = 0.0
              x_max = {40 * laser.wavelength_um * 1e-6}
            end:control

            begin:boundaries
              bc_x_min = simple_laser
              bc_x_max = simple_outflow
            end:boundaries

            begin:laser
              boundary = x_min
              amp = {laser.a0}
              lambda = {laser.wavelength_um * 1e-6}
              profile = gauss(time - 3*{laser.duration_fs}e-15, {laser.duration_fs}e-15)
            end:laser

            begin:species
              name = electron
              charge = -1.0
              mass = 1.0
              density = {target.n_over_nc} * critical(omega)
              density_min = 0.0
            end:species
        """)

    def _parse(self, run_dir: Path, laser: Laser) -> Result:
        raise NotImplementedError(
            "EPOCH SDF parsing is not yet implemented. "
            f"See {run_dir}. Track status in docs/backends.md."
        )
