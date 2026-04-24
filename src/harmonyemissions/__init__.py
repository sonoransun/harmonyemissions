"""Harmony of Emissions — configurable laser-plasma high-frequency emission workflows.

Public API:

    from harmonyemissions import Laser, Target, simulate
    result = simulate(laser, target, model="rom")
    result.spectrum  # xarray.DataArray: harmonic spectrum dI/dω
    result.attosecond_pulse  # time-domain filtered pulse
    result.save("run.h5")
"""

from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.runner import load_result, simulate
from harmonyemissions.target import Target

__all__ = ["Laser", "Target", "Result", "simulate", "load_result"]
__version__ = "0.3.0"
