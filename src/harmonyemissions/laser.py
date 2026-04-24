"""Laser pulse representation.

A :class:`Laser` bundles everything that defines the drive field:
wavelength, peak normalized amplitude a₀, pulse duration and envelope,
carrier-envelope phase, and polarization.
The time-domain electric field is evaluated in normalized units
E(t)/(mₑωc/e) = a(t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from harmonyemissions.units import LaserUnits, intensity_to_a0

Polarization = Literal["p", "s", "linear", "circular"]
Envelope = Literal["gaussian", "sin2", "flat-top"]
SpatialProfile = Literal["gaussian", "super_gaussian", "top_hat", "jinc"]


@dataclass(frozen=True)
class Laser:
    """Incident laser pulse in normalized units.

    Parameters
    ----------
    a0 : float
        Peak normalized vector potential (dimensionless).
    wavelength_um : float
        Central wavelength in μm.
    duration_fs : float
        FWHM (for gaussian / sin²) or flat-top duration of the intensity envelope.
    cep : float
        Carrier-envelope phase in radians.
    polarization : {"p", "s", "linear", "circular"}
        Polarization state. For surface-HHG simulations, "p" is the canonical choice.
    envelope : {"gaussian", "sin2", "flat-top"}
        Temporal envelope shape.
    angle_deg : float
        Angle of incidence on the target (0 = normal). Only used by models that care.
    spatial_profile : {"gaussian", "super_gaussian", "top_hat", "jinc"}
        Transverse amplitude profile of the focal spot. Super-Gaussian is the
        default because it matches saturated high-power amplifier chains and
        produces the wings that enable Coherent Harmonic Focusing.
    spot_fwhm_um : float
        Intensity FWHM of the focal spot in μm.
    super_gaussian_order : int
        Order ``p`` of the super-Gaussian profile (≥ 2); larger = flatter top.
    """

    a0: float
    wavelength_um: float = 0.8
    duration_fs: float = 5.0
    cep: float = 0.0
    polarization: Polarization = "p"
    envelope: Envelope = "gaussian"
    angle_deg: float = 0.0
    spatial_profile: SpatialProfile = "super_gaussian"
    spot_fwhm_um: float = 2.0
    super_gaussian_order: int = 8

    @classmethod
    def from_intensity(
        cls,
        intensity_w_per_cm2: float,
        wavelength_um: float = 0.8,
        **kwargs,
    ) -> Laser:
        """Build a Laser from peak intensity in W/cm²."""
        return cls(a0=intensity_to_a0(intensity_w_per_cm2, wavelength_um),
                   wavelength_um=wavelength_um, **kwargs)

    @property
    def units(self) -> LaserUnits:
        return LaserUnits.from_wavelength_um(self.wavelength_um)

    @property
    def omega0(self) -> float:
        """Carrier angular frequency in SI [rad/s]."""
        return self.units.omega

    def envelope_value(self, t_over_T0: np.ndarray) -> np.ndarray:
        """Return the pulse envelope a(t)/a₀ evaluated at times in units of T₀."""
        duration_T0 = self.duration_fs * 1e-15 / self.units.period_s
        t = np.asarray(t_over_T0, dtype=float)
        center = 0.5 * (t.min() + t.max()) if t.size else 0.0
        u = t - center
        if self.envelope == "gaussian":
            # duration_fs is the INTENSITY FWHM (standard laser-physics convention).
            # For amplitude a(t) = exp(-t²/(2σ²)), intensity a²(t) has
            # FWHM = 2σ√(ln 2) → σ = duration / (2 √(ln 2)).
            sigma = duration_T0 / (2.0 * math.sqrt(math.log(2.0)))
            return np.exp(-0.5 * (u / sigma) ** 2)
        if self.envelope == "sin2":
            half = duration_T0
            env = np.zeros_like(t)
            mask = np.abs(u) <= half
            env[mask] = np.cos(math.pi * u[mask] / (2 * half)) ** 2
            return env
        if self.envelope == "flat-top":
            env = np.zeros_like(t)
            env[np.abs(u) <= 0.5 * duration_T0] = 1.0
            return env
        raise ValueError(f"unknown envelope {self.envelope!r}")

    def field(self, t_over_T0: np.ndarray) -> np.ndarray:
        """Drive field a(t) = a₀ · envelope(t) · cos(2π t + φ_CEP)."""
        t = np.asarray(t_over_T0, dtype=float)
        return self.a0 * self.envelope_value(t) * np.cos(2 * math.pi * t + self.cep)

    def time_grid(self, n_periods: float = 20.0, samples_per_period: int = 512) -> np.ndarray:
        """Uniform time grid in units of T₀ spanning ± n_periods / 2 about pulse peak."""
        n = int(n_periods * samples_per_period)
        return np.linspace(-0.5 * n_periods, 0.5 * n_periods, n, endpoint=False)
