"""Execution backends: analytical (default), SMILEI, EPOCH, CuPy."""

from harmonyemissions.backends.analytical import AnalyticalBackend
from harmonyemissions.backends.cupy import CupyBackend
from harmonyemissions.backends.epoch import EpochBackend
from harmonyemissions.backends.smilei import SmileiBackend

BACKEND_REGISTRY = {
    "analytical": AnalyticalBackend,
    "smilei": SmileiBackend,
    "epoch": EpochBackend,
    "cupy": CupyBackend,
}

__all__ = [
    "AnalyticalBackend",
    "SmileiBackend",
    "EpochBackend",
    "CupyBackend",
    "BACKEND_REGISTRY",
]
