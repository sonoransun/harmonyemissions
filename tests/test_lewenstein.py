"""Regression tests for the Lewenstein gas-HHG model."""

import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.units import a0_to_intensity, ponderomotive_energy_ev


def test_corkum_cutoff_formula():
    """Diagnostic n_cutoff_corkum must equal (I_p + 3.17 U_p) / ħω₀."""
    laser = Laser(a0=0.08, wavelength_um=0.8, duration_fs=10.0)
    target = Target.gas("Ar")
    r = simulate(laser, target, model="lewenstein")
    ip_ev = 15.76  # Argon
    intensity = a0_to_intensity(0.08, 0.8)
    up_ev = ponderomotive_energy_ev(intensity, 0.8)
    photon_ev = 1239.84198 / 0.8
    expected = (ip_ev + 3.17 * up_ev) / photon_ev
    assert r.diagnostics["n_cutoff_corkum"] == pytest.approx(expected, rel=1e-6)


def test_cutoff_rises_with_intensity():
    r_low = simulate(Laser(a0=0.05), Target.gas("Ar"), model="lewenstein")
    r_high = simulate(Laser(a0=0.12), Target.gas("Ar"), model="lewenstein")
    assert r_high.diagnostics["n_cutoff_corkum"] > r_low.diagnostics["n_cutoff_corkum"]


def test_ionization_potential_lookup_for_noble_gases():
    for species, expected in [("He", 24.59), ("Ne", 21.56), ("Ar", 15.76), ("Xe", 12.13)]:
        r = simulate(
            Laser(a0=0.08), Target.gas(species), model="lewenstein",
        )
        assert r.diagnostics["ionization_potential_eV"] == pytest.approx(expected, rel=1e-3)


def test_unknown_species_requires_explicit_ip():
    from harmonyemissions.target import Target, ionization_potential
    t = Target.gas("Fr")  # not in table
    try:
        ionization_potential(t)
    except ValueError as e:
        assert "No default ionization potential" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown species")
