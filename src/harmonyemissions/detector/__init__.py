"""Instrument modelling.

Two regimes share one import surface:

- **XUV** — flat-field grating, Al-foil filter, back-thinned CCD
  (paper's ``S(λ) = Al · QE · G · Al₂O₃ · CH``).  See
  :mod:`harmonyemissions.detector.deconvolve`.
- **Hard X-ray / γ** — passive attenuation stack (Al / Cu / Ta / Pb / Au)
  followed by a scintillator or semiconductor active layer (NaI, CsI,
  LYSO, CdTe, HPGe, Si, YAG, IP).  See
  :mod:`harmonyemissions.detector.gamma_response`.

Both paths apply a multiplicative response ``S(E)`` to a simulation
``Result.spectrum`` and populate ``Result.instrument_spectrum``.
"""

from harmonyemissions.detector.al_filter import al_filter_transmission
from harmonyemissions.detector.deconvolve import apply_instrument_response
from harmonyemissions.detector.gamma_response import (
    GammaDetector,
    apply_gamma_response,
)
from harmonyemissions.detector.gamma_response import (
    spectral_response as gamma_spectral_response,
)
from harmonyemissions.detector.grating import grating_order_ratio
from harmonyemissions.detector.hard_xray import (
    FilterStack,
    filter_transmission,
    mass_attenuation_cm2_per_g,
)
from harmonyemissions.detector.scintillator import (
    DetectorConfig as GammaActiveLayer,
)
from harmonyemissions.detector.scintillator import (
    detector_absorption,
    detector_response,
)

__all__ = [
    # XUV
    "al_filter_transmission",
    "apply_instrument_response",
    "grating_order_ratio",
    # hard-X-ray / γ
    "FilterStack",
    "filter_transmission",
    "mass_attenuation_cm2_per_g",
    "GammaActiveLayer",
    "detector_absorption",
    "detector_response",
    "GammaDetector",
    "gamma_spectral_response",
    "apply_gamma_response",
]
