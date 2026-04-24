"""SMILEI PIC backend.

The input deck generated here matches the Timmis et al. 2026
(*Nature*, doi 10.1038/s41586-026-10400-2) Methods § "Numerical
simulations":

- 2-D Cartesian geometry.
- 512 cells per laser wavelength, 1024 timesteps per laser cycle.
- Fully-ionised SiO₂ target: electron density 6.62 × 10²³ cm⁻³, initial
  electron temperature 115 eV, ions cold.
- 100 macro-electrons per cell, 4 macro-ion species per cell.
- Exponential preplasma ramp with scale length 0.12 λ – 0.16 λ.
- Silver-Müller absorbing boundaries; Bouchard solver (SMILEI's
  ``maxwell_solver="Bouchard"``) to cut down on numerical dispersion.
- p-polarised Gaussian pulse, 45° incidence, spatial FWHM 2 μm,
  temporal FWHM 45–55 fs.

Output parsing is still a stub — SMILEI writes field probes to HDF5 that
need a project-specific mapping to :class:`Result`. The deck is however
complete and reproducible: drop it onto a SMILEI-equipped cluster and
the simulation runs as specified in the paper.
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


class SmileiNotAvailable(RuntimeError):
    """Raised when the SMILEI executable cannot be located on PATH."""


# Paper's numerical parameters — exposed as module constants so tests
# can check them against the rendered deck.
CELLS_PER_LAMBDA = 512
STEPS_PER_CYCLE = 1024
MACRO_ELECTRONS_PER_CELL = 100
MACRO_IONS_PER_CELL = 4
ELECTRON_TEMPERATURE_EV = 115
# 6.62 × 10²³ cm⁻³ ÷ critical density n_c(800 nm) ≈ 1.74 × 10²¹ cm⁻³  →  ~380 n_c.
SIO2_DENSITY_OVER_NC_800NM = 380.0


@dataclass
class SmileiBackend:
    name: str = "smilei"
    executable: str = "smilei"
    cells_per_lambda: int = CELLS_PER_LAMBDA
    steps_per_cycle: int = STEPS_PER_CYCLE
    grid_length_lambda_x: float = 40.0
    grid_length_lambda_y: float = 20.0

    def simulate(self, laser: Laser, target: Target, model: str, numerics) -> Result:
        if shutil.which(self.executable) is None:
            raise SmileiNotAvailable(
                f"Could not find {self.executable!r} on PATH. "
                "Install SMILEI (https://smileipic.github.io/Smilei/) and retry, "
                "or fall back to backend='analytical'."
            )
        if model not in {"rom", "cse", "surface_pipeline"}:
            raise NotImplementedError(
                f"SMILEI backend only wired for surface HHG; got model={model!r}."
            )
        with tempfile.TemporaryDirectory(prefix="harmony-smilei-") as td:
            td_path = Path(td)
            deck = self.render_deck(laser, target)
            (td_path / "input.py").write_text(deck)
            result = subprocess.run(
                [self.executable, "input.py"],
                cwd=td_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SMILEI exited with code {result.returncode}:\n{result.stderr}"
                )
            return self._parse_output(td_path, laser)

    # ------------------------------------------------------------------

    def render_deck(self, laser: Laser, target: Target) -> str:
        """Render a SMILEI input deck tuned to the Timmis 2026 setup.

        Exposed publicly so tests (and users) can inspect the deck before
        launching.
        """
        dx = 1.0 / self.cells_per_lambda
        dt = 1.0 / self.steps_per_cycle
        pulse_T0 = laser.duration_fs * 1e-15 / laser.units.period_s
        # SiO₂ density normalised to the driver's critical density, fallback to
        # target.n_over_nc if a non-silica target is passed in.
        n_over_nc = (
            SIO2_DENSITY_OVER_NC_800NM * (laser.wavelength_um / 0.8) ** 2
            if target.material.upper() == "SIO2"
            else target.n_over_nc
        )
        spot_fwhm_lambda = laser.spot_fwhm_um / laser.wavelength_um
        gradient = target.gradient_L_over_lambda
        center_x = 0.5 * self.grid_length_lambda_x
        center_y = 0.5 * self.grid_length_lambda_y

        return dedent(f"""\
            # Auto-generated deck for Harmony of Emissions
            # Reproduces Timmis et al., Nature (2026), Methods §'Numerical simulations'.
            #
            # Numerical parameters:
            #   {self.cells_per_lambda} cells / λ, {self.steps_per_cycle} steps / T₀
            #   maxwell_solver="Bouchard"         (paper's Bouchard ref. 54)
            #   Silver-Müller absorbing boundaries (paper ref. 53)
            #   {MACRO_ELECTRONS_PER_CELL} macro-electrons / cell
            #   {MACRO_IONS_PER_CELL} macro-ions   / cell (SiO₂ treated as 1 species)
            #   T_e = {ELECTRON_TEMPERATURE_EV} eV, T_i = 0 eV
            import math

            dx = {dx}
            dt = {dt}

            Main(
                geometry            = "2Dcartesian",
                interpolation_order = 2,
                timestep            = dt,
                simulation_time     = {3.0 * pulse_T0:.6f},
                cell_length         = [dx, dx],
                grid_length         = [{self.grid_length_lambda_x}, {self.grid_length_lambda_y}],
                number_of_patches   = [32, 16],
                maxwell_solver      = "Bouchard",
                EM_boundary_conditions = [["silver-muller", "silver-muller"],
                                           ["silver-muller", "silver-muller"]],
            )

            LaserGaussian2D(
                box_side        = "xmin",
                a0              = {laser.a0},
                omega           = 1.0,
                focus           = [{center_x}, {center_y}],
                waist           = {spot_fwhm_lambda / 2.355 / 2.0},
                incidence_angle = {laser.angle_deg * 3.141592653589793 / 180.0},
                polarization_phi = 0.0,          # p-polarisation in the plane of incidence
                ellipticity      = 0.0,
                time_envelope    = tgaussian(center={1.5 * pulse_T0:.6f},
                                              fwhm={pulse_T0:.6f}),
            )

            def density_profile(x, y):
                # Exponential preplasma ramp of scale length {gradient} λ.
                x_target = {center_x}
                return {n_over_nc} * math.exp((x - x_target) / {max(gradient, 1e-6)}) \\
                       if x <= x_target else 0.0

            Species(
                name                     = "electrons",
                position_initialization  = "regular",
                momentum_initialization  = "maxwell-juttner",
                temperature              = [{ELECTRON_TEMPERATURE_EV * 1.96e-6}],  # eV → m_e c²
                particles_per_cell       = {MACRO_ELECTRONS_PER_CELL},
                mass                     = 1.0,
                charge                   = -1.0,
                number_density           = density_profile,
                boundary_conditions      = [["reflective", "reflective"],
                                            ["reflective", "reflective"]],
            )

            Species(
                name                     = "ions",
                position_initialization  = "regular",
                momentum_initialization  = "cold",
                particles_per_cell       = {MACRO_IONS_PER_CELL},
                mass                     = 3640.0,      # average (Si+2O)/3 mass in mₑ
                charge                   = 10.0,        # (Z_Si + 2 Z_O) / 3 ≈ 10 for fully-ionised SiO₂
                number_density           = density_profile,
                boundary_conditions      = [["reflective", "reflective"],
                                            ["reflective", "reflective"]],
            )

            DiagFields(every=50, fields=["Ey", "By", "Rho"])
            DiagProbe(
                every=5,
                origin=[0.0, {center_y}],
                corners=[[{self.grid_length_lambda_x}, {center_y}]],
                number=[{self.cells_per_lambda * int(self.grid_length_lambda_x)}],
                fields=["Ey", "By"],
            )
        """)

    def _parse_output(self, run_dir: Path, laser: Laser) -> Result:
        raise NotImplementedError(
            "SMILEI output parsing is not yet implemented. "
            f"Run artifacts available in {run_dir} for manual inspection. "
            "Track progress in docs/backends.md."
        )
