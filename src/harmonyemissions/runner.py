"""Top-level simulation entry point (``harmonyemissions.simulate``).

This is the single place where ``(laser, target, model, backend)`` gets
resolved into a backend instance and dispatched. Keeping it thin lets the
CLI, notebooks, and parameter-sweep code all share one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harmonyemissions.backends import BACKEND_REGISTRY
from harmonyemissions.config import NumericsConfig, RunConfig
from harmonyemissions.laser import Laser
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target


def simulate(
    laser: Laser,
    target: Target,
    model: str = "rom",
    backend: str = "analytical",
    numerics: NumericsConfig | None = None,
    **backend_kwargs: Any,
) -> Result:
    """Run a single simulation and return a :class:`Result`.

    Parameters
    ----------
    laser : Laser
        Incident laser pulse.
    target : Target
        Plasma / gas target (kind must match the chosen model's regime).
    model : str
        One of ``"rom"``, ``"bgp"``, ``"cse"``, ``"lewenstein"``, ``"betatron"``.
    backend : str
        One of ``"analytical"`` (default), ``"smilei"``, ``"epoch"``.
    numerics : NumericsConfig or None
        Grid and pulse-synthesis controls. Defaults to ``NumericsConfig()``.
    **backend_kwargs
        Forwarded to the backend constructor (e.g. ``executable=...``).
    """
    numerics = numerics or NumericsConfig()
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend {backend!r}; known: {sorted(BACKEND_REGISTRY)}"
        )
    impl = BACKEND_REGISTRY[backend](**backend_kwargs)
    result = impl.simulate(laser, target, model, numerics)
    result.provenance.setdefault("laser", laser.__dict__.copy())
    result.provenance.setdefault("target", target.__dict__.copy())
    result.provenance.setdefault("model", model)
    return result


def simulate_from_config(config: RunConfig) -> Result:
    """Convenience: run a simulation described by a :class:`RunConfig`."""
    if config.laser_array is not None:
        raise NotImplementedError(
            "laser_array (3-D multi-beam coherent harmonic focus) is parsed and "
            "validated, but the runtime pipeline lands in Phase C. Run the "
            "single-beam pipeline by removing the 'laser_array' block from your "
            "config, or wait for the chf3d release."
        )
    return simulate(
        laser=config.laser.build(),
        target=config.target.build(),
        model=config.model,
        backend=config.backend,
        numerics=config.numerics,
    )


def load_result(path: str | Path) -> Result:
    """Load a previously-saved Result from HDF5."""
    return Result.load(path)
