"""CHF gain — 2-D → 3-D extrapolation and the I_CHF/I ∝ a₀³ scaling.

Definitions (Timmis 2026, Methods § "CHF 3D gain convergence"):

    Γ_D   = I₀ / I_L              # temporal compression (attosecond pulse vs driver)
    Γ_2D  = I_f / I₀               # spatial compression at the CHF focus in 2D
    Γ_3D  = Γ_2D²                  # 2→3D extrapolation (axisymmetric approximation)
    Γ     = Γ_D · Γ_3D             # total intensity boost

    I_CHF / I  ∝  a₀³             # paper's Conclusions scaling law

This module provides a small dataclass to carry the breakdown and
convenience helpers to combine the four gains and to predict CHF for
different laser systems from their a₀.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChfGainBreakdown:
    """Container for the four CHF gain factors."""

    gamma_d: float      # temporal compression
    gamma_2d: float     # 2-D spatial compression
    gamma_3d: float     # 3-D extrapolation = Γ_2D²
    gamma_total: float  # Γ_D · Γ_3D

    def to_dict(self) -> dict[str, float]:
        return {
            "Gamma_D": self.gamma_d,
            "Gamma_2D": self.gamma_2d,
            "Gamma_3D": self.gamma_3d,
            "Gamma_total": self.gamma_total,
        }


def extrapolate_3d_gain(
    intensity_attosecond: float,
    intensity_driver: float,
    intensity_at_chf_focus_2d: float,
) -> ChfGainBreakdown:
    """Build the CHF gain breakdown from three measured intensities.

    Parameters
    ----------
    intensity_attosecond : I₀
        Intensity of the bright attosecond pulse that seeds the CHF.
    intensity_driver : I_L
        Intensity of the incident driver laser pulse.
    intensity_at_chf_focus_2d : I_f
        Intensity at the CHF focus extracted from a 2-D PIC simulation.
    """
    if intensity_driver <= 0 or intensity_attosecond <= 0:
        raise ValueError("intensities must be positive")
    gamma_d = intensity_attosecond / intensity_driver
    gamma_2d = intensity_at_chf_focus_2d / intensity_attosecond
    gamma_3d = gamma_2d ** 2
    gamma_total = gamma_d * gamma_3d
    return ChfGainBreakdown(gamma_d, gamma_2d, gamma_3d, gamma_total)


def scaling_I_chf_over_I(a0: float, reference_a0: float = 24.0, reference_ratio: float = 80.0) -> float:
    """Predict I_CHF/I using the paper's a₀³ scaling law.

    The paper's Gemini point (a₀ ≈ 24, Γ ≳ 80) is taken as the anchor by
    default; pass a different (``reference_a0``, ``reference_ratio``) pair
    if you want to rescale against a different calibration shot.
    """
    if a0 <= 0 or reference_a0 <= 0:
        raise ValueError("a₀ must be positive")
    return reference_ratio * (a0 / reference_a0) ** 3


def predict_chf_intensity(
    driver_intensity_w_per_cm2: float,
    a0: float,
    reference_a0: float = 24.0,
    reference_ratio: float = 80.0,
) -> float:
    """Predicted absolute CHF intensity [W/cm²] using the a₀³ scaling."""
    return driver_intensity_w_per_cm2 * scaling_I_chf_over_I(
        a0, reference_a0, reference_ratio
    )
