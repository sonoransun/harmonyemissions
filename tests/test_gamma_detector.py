"""Tests for the hard-X-ray / γ-band instrument response.

Covers the three modules I added for extreme-photon-energy detection:
``detector.hard_xray`` (filter attenuation), ``detector.scintillator``
(active-layer QE), and ``detector.gamma_response`` (integrated S(E)).
"""

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.detector import (
    FilterStack,
    GammaActiveLayer,
    GammaDetector,
    apply_gamma_response,
    detector_absorption,
    filter_transmission,
    gamma_spectral_response,
    mass_attenuation_cm2_per_g,
)

# ---------------------------------------------------------------------------
# hard_xray.py
# ---------------------------------------------------------------------------


def test_mass_attenuation_unknown_material_raises():
    with pytest.raises(ValueError, match="Unknown material"):
        mass_attenuation_cm2_per_g(100.0, "Tantalum-Alloy-7")


def test_mass_attenuation_monotone_in_range():
    """Past the photoelectric regime, μ/ρ decreases roughly monotonically."""
    E = np.geomspace(30.0, 1000.0, 10)
    mu = mass_attenuation_cm2_per_g(E, "Cu")
    # No strict monotonicity across L/K edges, but over 30–1000 keV the
    # drop is ~40×.
    assert mu[0] / mu[-1] > 20.0


def test_filter_transmission_endpoints():
    # 1 mm Al at 1 keV → essentially 0; at 1 MeV → close to 1.
    T_soft = filter_transmission(1.0, "Al", 1000.0)
    T_hard = filter_transmission(1000.0, "Al", 1000.0)
    assert T_soft < 1e-10
    assert 0.6 < T_hard < 1.0


def test_filter_stack_is_multiplicative():
    """A two-layer stack equals the product of two single layers."""
    stack = FilterStack(layers=(("Al", 500.0), ("Cu", 50.0)))
    E = np.array([50.0, 200.0])
    T_stack = stack.transmission(E)
    T_product = (
        filter_transmission(E, "Al", 500.0) * filter_transmission(E, "Cu", 50.0)
    )
    np.testing.assert_allclose(T_stack, T_product, rtol=1e-12)


# ---------------------------------------------------------------------------
# scintillator.py
# ---------------------------------------------------------------------------


def test_detector_absorption_bounded_in_unit_interval():
    E = np.geomspace(10.0, 10000.0, 32)
    F = detector_absorption(E, GammaActiveLayer(name="CsI", thickness_mm=5.0))
    assert np.all(F >= 0.0) and np.all(F <= 1.0)


def test_thick_detector_absorbs_more():
    E = np.array([100.0])
    thin = detector_absorption(E, GammaActiveLayer(name="HPGe", thickness_mm=1.0))
    thick = detector_absorption(E, GammaActiveLayer(name="HPGe", thickness_mm=20.0))
    assert thick[0] > thin[0]


def test_semiconductor_menu_covers_common_detectors():
    for name in ["NaI", "CsI", "LYSO", "CdTe", "HPGe", "Si", "YAG", "IP"]:
        GammaActiveLayer(name=name, thickness_mm=3.0)  # construction shouldn't raise


# ---------------------------------------------------------------------------
# gamma_response.py
# ---------------------------------------------------------------------------


def test_spectral_response_shape():
    E = np.geomspace(1.0, 10000.0, 100)
    S = gamma_spectral_response(E)
    assert S.shape == E.shape
    assert np.all(S >= 0.0)
    # Below the Al L-edge (~1.5 keV) transmission is essentially zero
    # through a 500 μm Al filter.
    assert S[0] < 1e-10


def test_apply_gamma_response_to_betatron_run():
    """End-to-end: betatron spectrum → gamma S(E) → instrument_spectrum."""
    r = simulate(
        Laser(a0=2.0), Target.underdense(0.001, electron_energy_mev=500.0),
        model="betatron",
    )
    det = GammaDetector(
        filters=FilterStack(layers=(("Al", 200.0), ("Cu", 25.0))),
        detector=GammaActiveLayer(name="CsI", thickness_mm=5.0),
    )
    sig = apply_gamma_response(r.spectrum, det)
    # Same coord, same shape.
    assert sig.shape == r.spectrum.shape
    assert sig.attrs["detector_name"] == "CsI"
    # Must be non-negative and not all zero.
    assert np.all(sig.values >= 0.0)
    assert np.any(sig.values > 0.0)


def test_apply_gamma_response_missing_coord_raises():
    """Result without photon_energy_keV coord should raise cleanly."""
    from harmonyemissions import simulate
    r = simulate(Laser(a0=10.0), Target.overdense(100.0), model="bgp")
    # BGP result has only `harmonic` coord — no absolute energy grid.
    # strip any photon_energy_keV that might have been added upstream:
    spec = r.spectrum.drop_vars("photon_energy_keV", errors="ignore")
    with pytest.raises(ValueError, match="photon_energy_keV"):
        apply_gamma_response(spec)
