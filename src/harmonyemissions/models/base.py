"""Shared types for all emission models.

Every model returns the same :class:`Result` object regardless of physics.
That invariant is what lets the CLI, plotting, and parameter-sweep code
stay regime-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from harmonyemissions.laser import Laser
    from harmonyemissions.target import Target


@runtime_checkable
class EmissionModel(Protocol):
    """Protocol every physics model implements."""

    name: str

    def run(self, laser: Laser, target: Target, numerics: Any) -> Result:
        """Compute emission for the given laser/target configuration."""


@dataclass
class Result:
    """Regime-agnostic output of a simulation run.

    ``spectrum`` is mandatory; the rest are optional and may be ``None`` when
    a model cannot meaningfully produce them (e.g. BGP has no time-domain
    field, only a spectrum; non-pipeline models have no 2-D dent or beam
    profile; non-CHF runs have an empty ``chf_gain`` dict).
    """

    spectrum: xr.DataArray  # dI/dω vs harmonic order n
    time_field: xr.DataArray | None = None  # E(t) in normalized units, indexed by t/T₀
    attosecond_pulse: xr.DataArray | None = None  # filtered time-domain pulse
    # Timmis 2026 pipeline outputs:
    dent_map: xr.DataArray | None = None          # Δz(x', y') / λ
    beam_profile_near: xr.DataArray | None = None  # |U₀(x', y')|² driver
    beam_profile_far: xr.DataArray | None = None   # |U(x, y, z, n)|² per harmonic
    chf_gain: dict[str, float] = field(default_factory=dict)
    # Post-processed, detector-corrected spectrum (if the instrument pipeline ran).
    instrument_spectrum: xr.DataArray | None = None
    # 3-D coherent-focus (chf3d) outputs — populated only when laser_array is set.
    chf_focal_volume: xr.DataArray | None = None      # (harmonic_diag, x, y, z) intensity
    per_beam_far_field: xr.DataArray | None = None    # (beam_index, harmonic_diag, yi, xi)
    beam_array_geometry: dict[str, Any] | None = None  # JSON-round-trippable geometry record
    diagnostics: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---- summaries -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a flat dict suitable for printing or tabulating."""
        out: dict[str, Any] = {"n_points": int(self.spectrum.size)}
        out.update(self.diagnostics)
        out.update({f"provenance.{k}": v for k, v in self.provenance.items()})
        return out

    def cutoff_harmonic(self, threshold_db: float = 30.0) -> float:
        """Empirical cutoff: highest harmonic still within ``threshold_db`` of the peak."""
        s = self.spectrum.values
        n = self.spectrum.coords["harmonic"].values
        if s.max() <= 0:
            return float("nan")
        ratio_db = 10.0 * np.log10(np.maximum(s / s.max(), 1e-30))
        above = np.where(ratio_db > -threshold_db)[0]
        return float(n[above[-1]]) if above.size else float("nan")

    def fit_power_law(self, n_min: float = 5.0, n_max: float | None = None) -> tuple[float, float]:
        """Least-squares fit log I(n) = a + b log n over [n_min, n_max].

        Returns (slope, intercept). For surface HHG the BGP prediction is
        slope ≈ −8/3 in the plateau; CSE predicts ≈ −4/3.
        """
        n = self.spectrum.coords["harmonic"].values
        s = self.spectrum.values
        mask = (n >= n_min) & (s > 0)
        if n_max is not None:
            mask &= n <= n_max
        if mask.sum() < 3:
            return (float("nan"), float("nan"))
        log_n = np.log(n[mask])
        log_s = np.log(s[mask])
        slope, intercept = np.polyfit(log_n, log_s, 1)
        return float(slope), float(intercept)

    # ---- persistence ---------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        """Pack everything into a single xarray.Dataset for persistence."""
        data: dict[str, xr.DataArray] = {"spectrum": self.spectrum}
        for key in (
            "time_field",
            "attosecond_pulse",
            "dent_map",
            "beam_profile_near",
            "beam_profile_far",
            "instrument_spectrum",
            "chf_focal_volume",
            "per_beam_far_field",
        ):
            val = getattr(self, key)
            if val is not None:
                data[key] = val
        ds = xr.Dataset(data)
        ds.attrs["diagnostics_json"] = _dumps(self.diagnostics)
        ds.attrs["provenance_json"] = _dumps(self.provenance)
        ds.attrs["chf_gain_json"] = _dumps(self.chf_gain)
        if self.beam_array_geometry is not None:
            ds.attrs["beam_array_geometry_json"] = _dumps(self.beam_array_geometry)
        return ds

    def save(self, path: str | Path) -> Path:
        """Persist this Result to an HDF5 (netCDF4) file via h5netcdf."""
        p = Path(path)
        self.to_dataset().to_netcdf(p, engine="h5netcdf")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Result:
        ds = xr.open_dataset(path, engine="h5netcdf")

        def take(name: str) -> xr.DataArray | None:
            return ds[name] if name in ds else None

        geom_json = ds.attrs.get("beam_array_geometry_json")
        return cls(
            spectrum=ds["spectrum"],
            time_field=take("time_field"),
            attosecond_pulse=take("attosecond_pulse"),
            dent_map=take("dent_map"),
            beam_profile_near=take("beam_profile_near"),
            beam_profile_far=take("beam_profile_far"),
            instrument_spectrum=take("instrument_spectrum"),
            chf_focal_volume=take("chf_focal_volume"),
            per_beam_far_field=take("per_beam_far_field"),
            beam_array_geometry=_loads(geom_json) if geom_json else None,
            chf_gain=_loads(ds.attrs.get("chf_gain_json", "{}")),
            diagnostics=_loads(ds.attrs.get("diagnostics_json", "{}")),
            provenance=_loads(ds.attrs.get("provenance_json", "{}")),
        )


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, default=_safe_default)


def _loads(s: str) -> dict[str, Any]:
    import json
    return json.loads(s)


def _safe_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return str(obj)
