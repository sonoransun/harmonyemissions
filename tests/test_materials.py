import pytest

from harmonyemissions.materials import MATERIAL_REGISTRY, MaterialData, lookup


def test_registry_has_expected_materials():
    for key in ("Al", "Si", "SiO2", "Ti", "Fe", "Cu", "Mo", "Ag"):
        assert key in MATERIAL_REGISTRY


def test_kalpha_below_k_edge_for_every_entry():
    # Physical consistency: Kα emission fills the K-shell vacancy,
    # its photon energy must lie below the K-edge it came from.
    for name, m in MATERIAL_REGISTRY.items():
        assert m.K_alpha1_keV < m.K_edge_keV, f"{name}: Kα1 must be < K-edge"
        assert m.K_alpha2_keV < m.K_edge_keV, f"{name}: Kα2 must be < K-edge"
        assert m.K_beta_keV < m.K_edge_keV, f"{name}: Kβ must be < K-edge"


def test_kalpha2_below_kalpha1():
    for name, m in MATERIAL_REGISTRY.items():
        assert m.K_alpha2_keV <= m.K_alpha1_keV, f"{name}: Kα2 must not exceed Kα1"


def test_fluorescence_yield_monotone_with_Z():
    # ω_K rises with Z (Bambynek). Spot-check ordering across our set.
    entries = sorted(MATERIAL_REGISTRY.values(), key=lambda m: m.Z)
    ys = [m.fluorescence_yield_K for m in entries]
    # Allow ties for Si/SiO2 (same Z).
    for a, b in zip(ys, ys[1:], strict=False):
        assert b + 1e-12 >= a


def test_lookup_case_insensitive():
    assert lookup("cu").name == "Cu"
    assert lookup("Cu") is MATERIAL_REGISTRY["Cu"]


def test_lookup_unknown_raises():
    with pytest.raises(ValueError, match="No material data"):
        lookup("unobtainium")


def test_cu_kalpha1_landmark():
    # Cu Kα1 is the X-ray crystallography reference line at 8.048 keV.
    m: MaterialData = lookup("Cu")
    assert m.K_alpha1_keV == pytest.approx(8.048, abs=0.005)
    assert m.K_edge_keV == pytest.approx(8.979, abs=0.005)
