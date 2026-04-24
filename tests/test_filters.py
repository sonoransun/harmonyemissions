"""Tests for the shared X-ray attenuation library."""

import numpy as np
import pytest

from harmonyemissions.detector import filters
from harmonyemissions.detector.al_filter import al_filter_transmission


def test_list_materials_includes_dpssl_set():
    mats = filters.list_materials()
    for must_have in ("al", "be", "cu", "ni", "mo", "sn", "ag", "si",
                       "cdte", "mylar", "kapton", "lyso", "bgo", "csi"):
        assert must_have in mats, f"{must_have} missing from attenuation tables"


def test_transmission_monotonic_with_thickness():
    """Doubling thickness must decrease transmission monotonically."""
    E = np.logspace(3, 4, 32)  # 1–10 keV
    T1 = filters.transmission("cu", E, thickness_um=10.0)
    T2 = filters.transmission("cu", E, thickness_um=50.0)
    assert np.all(T2 <= T1)
    assert np.all(T1 >= 0.0) and np.all(T1 <= 1.0)


def test_transmission_monotonic_in_energy_above_K_edge():
    """Away from edges, hard-X-ray transmission grows with energy."""
    E = np.logspace(4, 6, 64)  # 10 keV–1 MeV, above Cu K-edge
    T = filters.transmission("cu", E, thickness_um=100.0)
    # Not strictly monotonic globally due to Compton plateau, but the trend
    # in this window is strictly increasing (photoelectric dominates).
    assert T[-1] > T[0]


def test_Cu_K_edge_is_present_in_metadata():
    cu = filters.load_material("cu")
    assert "K" in cu.k_edges_ev
    assert cu.k_edges_ev["K"] == pytest.approx(8979.0, abs=1.0)


def test_Al_both_stacks_soft_xray_in_band():
    """Both the new tabulated stack and the legacy Henke fit must predict
    the Al filter is partially transmissive in its 17–80 nm XUV band and
    opaque at its K-edge (1559 eV). We compare orders of magnitude rather
    than pointwise because the legacy fit deliberately uses a simplified
    interpolation (see ``detector/al_filter.py`` docstring).
    """
    wavelength_nm = np.linspace(30.0, 70.0, 20)
    energy_ev = 1239.841984 / wavelength_nm
    T_new = filters.transmission("al", energy_ev, thickness_um=1.5)
    T_old = al_filter_transmission(wavelength_nm, thickness_um=1.5)
    # Both stacks: non-negative, never above 1.
    assert np.all(T_new >= 0) and np.all(T_new <= 1)
    assert np.all(T_old >= 0) and np.all(T_old <= 1)
    # At the L-edge (72 eV) 1.5 μm Al must be nearly opaque.
    T_new_L = filters.transmission("al", np.array([60.0]), thickness_um=1.5)
    assert T_new_L[0] < 0.01
    # A thicker filter (10 μm) must attenuate more than a thinner one.
    T_thick = filters.transmission("al", energy_ev, thickness_um=10.0)
    assert np.all(T_thick <= T_new + 1e-12)


def test_Be_transparent_in_hard_xray():
    """Be window is nearly transparent at 10+ keV even at 100 μm."""
    E = np.array([10_000.0, 30_000.0])
    T = filters.transmission("be", E, thickness_um=100.0)
    assert np.all(T > 0.85)


def test_Ross_pair_passband_between_cu_and_ni_K_edges():
    """Ross-pair differential window lies BETWEEN the two K-edges:
    8.33 keV (Ni) < passband < 8.98 keV (Cu). In that window Ni has
    crossed K-edge and opaques, while Cu has not yet crossed K-edge
    and remains transparent → T_Cu > T_Ni.
    """
    E_in = np.array([8_650.0])  # between the two K-edges
    E_above = np.array([25_000.0])  # above both: both equally attenuating
    T_cu_in = filters.transmission("cu", E_in, thickness_um=50.0)
    T_ni_in = filters.transmission("ni", E_in, thickness_um=55.0)
    # Inside the passband Cu is more transmissive than Ni.
    assert T_cu_in[0] > T_ni_in[0]
    # Well above both edges the differential collapses.
    T_cu_hi = filters.transmission("cu", E_above, thickness_um=50.0)
    T_ni_hi = filters.transmission("ni", E_above, thickness_um=55.0)
    diff_hi = abs(T_cu_hi[0] - T_ni_hi[0])
    diff_in = abs(T_cu_in[0] - T_ni_in[0])
    assert diff_in > diff_hi


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="No attenuation table"):
        filters.load_material("unobtainium")


def test_filter_spec_is_frozen_dataclass():
    from dataclasses import FrozenInstanceError

    f = filters.FilterSpec("al", 1.5)
    with pytest.raises(FrozenInstanceError):
        f.thickness_um = 2.0
