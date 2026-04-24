"""Tests for Target factory methods and ionization_potential lookup."""

import pytest

from harmonyemissions.target import (
    IONIZATION_POTENTIALS_EV,
    Target,
    ionization_potential,
)


def test_overdense_factory_sets_kind():
    t = Target.overdense(n_over_nc=100.0, gradient_L_over_lambda=0.1)
    assert t.kind == "overdense"
    assert t.n_over_nc == 100.0
    assert t.gradient_L_over_lambda == 0.1
    assert t.material == "SiO2"


def test_sio2_factory_defaults():
    t = Target.sio2()
    assert t.kind == "overdense"
    assert t.material == "SiO2"
    assert t.n_over_nc == 200.0
    assert t.gradient_L_over_lambda == 0.14
    assert t.t_HDR_fs == 351.0


def test_sio2_factory_accepts_contrast_knobs():
    t = Target.sio2(t_HDR_fs=711.0, prepulse_intensity_rel=1e-4, prepulse_delay_fs=50.0)
    assert t.t_HDR_fs == 711.0
    assert t.prepulse_intensity_rel == 1e-4
    assert t.prepulse_delay_fs == 50.0


def test_gas_factory_pressure_and_species():
    t = Target.gas("Ar", pressure_mbar=30.0)
    assert t.kind == "gas"
    assert t.gas_species == "Ar"
    assert t.pressure_mbar == 30.0


def test_underdense_factory():
    t = Target.underdense(n_over_nc=0.001, electron_energy_mev=500.0, betatron_amplitude_um=2.0)
    assert t.kind == "underdense"
    assert t.electron_energy_mev == 500.0
    assert t.betatron_amplitude_um == 2.0


def test_ionization_potentials_for_noble_gases():
    for species, ip_eV in [("H", 13.6), ("He", 24.59), ("Ar", 15.76), ("Xe", 12.13)]:
        t = Target.gas(species)
        assert ionization_potential(t) == pytest.approx(ip_eV, rel=1e-3)


def test_ionization_potential_override():
    t = Target.gas("He", ionization_potential_eV=20.0)
    assert ionization_potential(t) == 20.0


def test_unknown_species_raises():
    t = Target.gas("Fr")  # francium not in table
    with pytest.raises(ValueError, match="No default ionization potential"):
        ionization_potential(t)


def test_target_is_frozen():
    t = Target.overdense(100.0)
    with pytest.raises(AttributeError):
        t.n_over_nc = 200.0  # type: ignore[misc]


def test_all_noble_gases_have_ip_entry():
    expected = {"H", "He", "Ne", "Ar", "Kr", "Xe"}
    assert expected.issubset(IONIZATION_POTENTIALS_EV.keys())
