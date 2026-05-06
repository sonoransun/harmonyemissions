"""YAML-backed run configuration.

A config file fully specifies a run: laser, target, model, backend, and
numerical controls. pydantic handles validation and provides reasonable
error messages when fields are missing or out of range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from harmonyemissions.laser import Envelope, Laser, Polarization, SpatialProfile
from harmonyemissions.presets import apply_preset, list_presets
from harmonyemissions.target import Kind as TargetKind
from harmonyemissions.target import Target


class LaserConfig(BaseModel):
    preset: str | None = None
    a0: float = Field(gt=0.0)
    wavelength_um: float = Field(default=0.8, gt=0.0)
    duration_fs: float = Field(default=5.0, gt=0.0)
    cep: float = 0.0
    polarization: Polarization = "p"
    envelope: Envelope = "gaussian"
    angle_deg: float = 0.0
    spatial_profile: SpatialProfile = "super_gaussian"
    spot_fwhm_um: float = 2.0
    super_gaussian_order: int = 8

    @model_validator(mode="before")
    @classmethod
    def _merge_preset(cls, data: Any) -> Any:
        # Only dict inputs go through preset merging; pydantic may pass other shapes.
        if not isinstance(data, dict):
            return data
        name = data.get("preset")
        if name is None:
            return data
        if name not in list_presets():
            raise ValueError(
                f"Unknown laser preset {name!r}. Known: {sorted(list_presets())}"
            )
        merged = apply_preset(name, {k: v for k, v in data.items() if k != "preset"})
        merged["preset"] = name
        return merged

    def build(self) -> Laser:
        fields = self.model_dump()
        fields.pop("preset", None)
        return Laser(**fields)


class TargetConfig(BaseModel):
    kind: TargetKind
    n_over_nc: float = 100.0
    gradient_L_over_lambda: float = 0.1
    material: str = "SiO2"
    reflectivity: float = 0.6
    t_HDR_fs: float = 351.0
    prepulse_intensity_rel: float = 0.0
    prepulse_delay_fs: float = 0.0
    hot_electron_temp_keV: float | None = None
    gas_species: str = "Ar"
    pressure_mbar: float = 20.0
    ionization_potential_eV: float | None = None
    bubble_radius_um: float = 10.0
    electron_energy_mev: float = 200.0
    betatron_amplitude_um: float = 1.0
    beam_energy_mev: float = 100.0
    beam_charge_pc: float = 50.0
    beam_divergence_mrad: float = 1.0
    beam_bunch_length_fs: float = 50.0

    def build(self) -> Target:
        return Target(**self.model_dump())


class NumericsConfig(BaseModel):
    n_periods: float = 20.0
    samples_per_period: int = 512
    harmonic_window: tuple[float, float] | None = None  # (n_low, n_high) for pulse synthesis
    # surface_pipeline extras (Timmis 2026 path):
    pipeline_grid: int | None = None              # transverse grid size N×N
    pipeline_dx_um: float | None = None           # pixel size, μm
    diag_harmonics: tuple[int, ...] | None = None  # 2-D far-field slices to store
    # 3-D coherent-focus (chf3d) extras — consumed by laser_array runs:
    chf_focal_volume_n: int | None = None
    chf_focal_volume_extent_um: float | None = None
    chf_focal_volume_mode: Literal["point", "volume"] = "volume"
    store_per_beam_far_field: bool = False
    phase_optimiser: Literal["analytic", "scipy_lbfgs", "gerchberg_saxton"] = "analytic"


# Geometries supported by chf/geometry.py (lands in Phase C). Phase A only
# parses these names and validates structural consistency.
GeometryName = Literal[
    "tetrahedral", "cubic", "octahedral", "dodecahedral", "icosahedral",
    "explicit", "ring", "fibonacci_sphere",
]
PolarizationMode = Literal[
    "uniform_p", "uniform_s", "radial", "azimuthal",
    "circular_alternating", "explicit",
]
StructuredMode = Literal["lg", "bessel", "radial", "azimuthal"]

_PLATONIC_FACE_COUNT: dict[str, int] = {
    "tetrahedral": 4, "cubic": 6, "octahedral": 8,
    "dodecahedral": 12, "icosahedral": 20,
}
_PLATONIC_VERTEX_COUNT: dict[str, int] = {
    "tetrahedral": 4, "cubic": 8, "octahedral": 6,
    "dodecahedral": 20, "icosahedral": 12,
}


class LaserArrayConfig(BaseModel):
    """Multi-driver beam-array geometry for 3-D coherent harmonic focus.

    Only the parsing and structural validation lands in Phase A; geometry
    materialisation (`build()` → BeamArray) and the multi-beam pipeline
    arrive in Phase C. A config with `laser_array` set therefore parses
    cleanly and round-trips through YAML/HDF5 today, but raises
    NotImplementedError at the runner until the physics kernel ships.
    """

    geometry: GeometryName
    placement: Literal["faces", "vertices"] = "faces"
    n_beams: int | None = None
    directions: list[tuple[float, float, float]] | None = None
    relative_phase_rad: list[float] | None = None
    relative_delay_fs: list[float] | None = None
    polarization_mode: PolarizationMode = "uniform_p"
    polarization_vectors: list[tuple[float, float, float]] | None = None
    per_beam_a0_scale: list[float] | None = None
    structured_mode: StructuredMode | None = None
    structured_mode_params: dict[str, Any] | None = None

    def effective_n_beams(self) -> int:
        if self.directions is not None:
            return len(self.directions)
        if self.geometry in _PLATONIC_FACE_COUNT:
            counts = (_PLATONIC_FACE_COUNT if self.placement == "faces"
                      else _PLATONIC_VERTEX_COUNT)
            return counts[self.geometry]
        if self.n_beams is None:
            raise ValueError(
                f"could not infer n_beams for geometry={self.geometry!r}; "
                f"set n_beams or directions explicitly"
            )
        return self.n_beams

    @model_validator(mode="after")
    def _validate(self) -> LaserArrayConfig:
        if self.geometry == "explicit":
            if not self.directions:
                raise ValueError("geometry='explicit' requires non-empty directions")
        elif self.geometry in {"ring", "fibonacci_sphere"}:
            if self.n_beams is None and self.directions is None:
                raise ValueError(
                    f"geometry={self.geometry!r} requires n_beams (or explicit directions)"
                )
        n = self.effective_n_beams()
        for name, lst in (
            ("directions", self.directions),
            ("relative_phase_rad", self.relative_phase_rad),
            ("relative_delay_fs", self.relative_delay_fs),
            ("per_beam_a0_scale", self.per_beam_a0_scale),
        ):
            if lst is not None and len(lst) != n:
                raise ValueError(
                    f"{name} length {len(lst)} != effective n_beams {n}"
                )
        if self.polarization_mode == "explicit":
            if not self.polarization_vectors or len(self.polarization_vectors) != n:
                raise ValueError(
                    f"polarization_mode='explicit' needs {n} polarization_vectors"
                )
        if self.per_beam_a0_scale is not None:
            ss = sum(float(s) ** 2 for s in self.per_beam_a0_scale)
            if ss > 1.0 + 1e-9:
                raise ValueError(
                    f"per_beam_a0_scale sums-of-squares = {ss:.3f} exceeds 1 "
                    f"(violates total-power conservation; pass scales such that "
                    f"sum(s_i^2) ≤ 1)"
                )
        if self.structured_mode == "lg":
            params = self.structured_mode_params or {}
            if "l" not in params or "p" not in params:
                raise ValueError(
                    "structured_mode='lg' requires structured_mode_params={'l': int, 'p': int}"
                )
        elif self.structured_mode == "bessel":
            params = self.structured_mode_params or {}
            if "kr_per_k" not in params:
                raise ValueError(
                    "structured_mode='bessel' requires structured_mode_params={'kr_per_k': float, ...}"
                )
        return self


class RunConfig(BaseModel):
    model: Literal[
        "rom", "bgp", "cse", "lewenstein", "betatron", "surface_pipeline", "cwe",
        "ics", "bremsstrahlung", "kalpha",
    ]
    backend: Literal["analytical", "smilei", "epoch"] = "analytical"
    laser: LaserConfig
    target: TargetConfig
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    laser_array: LaserArrayConfig | None = None
    output: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_regime_match(self) -> RunConfig:
        surface = {
            "rom", "bgp", "cse", "surface_pipeline", "cwe",
            "bremsstrahlung", "kalpha",
        }
        if self.model in surface and self.target.kind != "overdense":
            raise ValueError(
                f"Model {self.model!r} requires kind='overdense', "
                f"got kind={self.target.kind!r}"
            )
        if self.model == "lewenstein" and self.target.kind != "gas":
            raise ValueError("Lewenstein model requires kind='gas'")
        if self.model == "betatron" and self.target.kind != "underdense":
            raise ValueError("Betatron model requires kind='underdense'")
        if self.model == "ics" and self.target.kind not in ("underdense", "electron_beam"):
            raise ValueError(
                "ICS model requires kind='underdense' (LWFA-ICS) or 'electron_beam'"
            )
        if self.laser_array is not None and self.model != "surface_pipeline":
            raise ValueError(
                f"laser_array is currently only supported with model='surface_pipeline'; "
                f"got model={self.model!r}"
            )
        return self


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML config file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return RunConfig.model_validate(data)


def dump_config(config: RunConfig, path: str | Path) -> None:
    """Write a config back to YAML (useful for record-keeping of scan runs)."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, sort_keys=False)
