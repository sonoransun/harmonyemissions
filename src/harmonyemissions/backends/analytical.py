"""Default backend: dispatch to a semi-analytical model in-process."""

from __future__ import annotations

from dataclasses import dataclass

from harmonyemissions.laser import Laser
from harmonyemissions.models import MODEL_REGISTRY
from harmonyemissions.models.base import Result
from harmonyemissions.target import Target


@dataclass
class AnalyticalBackend:
    name: str = "analytical"

    def simulate(self, laser: Laser, target: Target, model: str, numerics) -> Result:
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model!r}; known: {sorted(MODEL_REGISTRY)}"
            )
        impl = MODEL_REGISTRY[model]()
        result = impl.run(laser, target, numerics)
        result.provenance.setdefault("backend", "analytical")
        return result
