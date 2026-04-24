"""Physics models for high-frequency emission from laser-plasma interactions."""

from harmonyemissions.emission.cwe import CWEModel
from harmonyemissions.models.base import EmissionModel, Result
from harmonyemissions.models.betatron import BetatronModel
from harmonyemissions.models.bgp import BGPModel
from harmonyemissions.models.bremsstrahlung import BremsstrahlungModel
from harmonyemissions.models.cse import CSEModel
from harmonyemissions.models.ics import ICSModel
from harmonyemissions.models.kalpha import KalphaModel
from harmonyemissions.models.lewenstein import LewensteinModel
from harmonyemissions.models.rom import ROMModel
from harmonyemissions.models.surface_pipeline import SurfacePipelineModel

MODEL_REGISTRY: dict[str, type[EmissionModel]] = {
    "rom": ROMModel,
    "bgp": BGPModel,
    "cse": CSEModel,
    "lewenstein": LewensteinModel,
    "betatron": BetatronModel,
    # Timmis 2026 additions:
    "surface_pipeline": SurfacePipelineModel,
    "cwe": CWEModel,
    # Hard-X-ray family:
    "bremsstrahlung": BremsstrahlungModel,
    "kalpha": KalphaModel,
    "ics": ICSModel,
}

__all__ = [
    "EmissionModel",
    "Result",
    "ROMModel",
    "BGPModel",
    "BremsstrahlungModel",
    "KalphaModel",
    "ICSModel",
    "CSEModel",
    "LewensteinModel",
    "BetatronModel",
    "SurfacePipelineModel",
    "CWEModel",
    "MODEL_REGISTRY",
]
