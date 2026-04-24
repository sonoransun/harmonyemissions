"""Laser facility presets for high-power plasma drivers.

Presets populate :class:`LaserConfig` fields when a YAML config sets
``laser.preset: <name>``. User-supplied keys win on merge so presets
act as defaults, not constraints.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PRESET_FILE = Path(__file__).with_name("data") / "laser_presets.yaml"

# Fields the preset is allowed to fill on LaserConfig. Anything else in the
# YAML (facility, gain_medium, rep_rate_hz, citation) is metadata only.
_LASER_FIELDS: set[str] = {
    "a0",
    "wavelength_um",
    "duration_fs",
    "cep",
    "polarization",
    "envelope",
    "angle_deg",
    "spatial_profile",
    "spot_fwhm_um",
    "super_gaussian_order",
}


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, dict[str, Any]]:
    with open(_PRESET_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_presets() -> list[str]:
    """Return preset names in the order they appear in the YAML."""
    return list(_load_raw().keys())


def get_preset(name: str) -> dict[str, Any]:
    """Return the raw preset dict (including metadata) for ``name``."""
    raw = _load_raw()
    if name not in raw:
        raise ValueError(
            f"Unknown laser preset {name!r}. Known: {sorted(raw)}"
        )
    return dict(raw[name])


def apply_preset(name: str, user_laser_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge a preset into a user LaserConfig dict.

    User keys win; the preset's ``a0_reference`` is renamed to ``a0`` only
    when the user did not set ``a0`` explicitly.
    """
    preset = get_preset(name)
    merged: dict[str, Any] = {}
    # Start from preset fields that map to LaserConfig.
    for key in _LASER_FIELDS:
        if key in preset:
            merged[key] = preset[key]
    if "a0" not in merged and "a0_reference" in preset:
        merged["a0"] = preset["a0_reference"]
    # User keys override.
    for key, val in user_laser_cfg.items():
        if val is None and key == "preset":
            # Ignore the preset marker itself.
            continue
        merged[key] = val
    return merged


def preset_metadata(name: str) -> dict[str, Any]:
    """Facility / citation metadata (non-physics fields) for documentation."""
    preset = get_preset(name)
    return {
        k: preset[k]
        for k in ("facility", "gain_medium", "rep_rate_hz", "citation")
        if k in preset
    }
