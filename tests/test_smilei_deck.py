"""Tests for the SMILEI input-deck generator (Timmis 2026 setup)."""

import pytest

from harmonyemissions import Laser, Target
from harmonyemissions.backends.smilei import (
    CELLS_PER_LAMBDA,
    MACRO_ELECTRONS_PER_CELL,
    MACRO_IONS_PER_CELL,
    STEPS_PER_CYCLE,
    SmileiBackend,
    SmileiNotAvailable,
)


@pytest.fixture
def deck() -> str:
    laser = Laser(
        a0=24.0, wavelength_um=0.8, duration_fs=50.0, angle_deg=45.0,
        spatial_profile="super_gaussian", spot_fwhm_um=2.0, super_gaussian_order=8,
    )
    target = Target.sio2(t_HDR_fs=351.0)
    return SmileiBackend().render_deck(laser, target)


def test_deck_uses_paper_cell_count(deck: str):
    assert f"dx = {1.0 / CELLS_PER_LAMBDA}" in deck


def test_deck_uses_paper_step_count(deck: str):
    assert f"dt = {1.0 / STEPS_PER_CYCLE}" in deck


def test_deck_uses_bouchard_solver(deck: str):
    assert 'maxwell_solver      = "Bouchard"' in deck


def test_deck_uses_silver_muller_bc(deck: str):
    assert "silver-muller" in deck.lower()


def test_deck_sets_macroparticle_counts(deck: str):
    assert f"particles_per_cell       = {MACRO_ELECTRONS_PER_CELL}" in deck
    assert f"particles_per_cell       = {MACRO_IONS_PER_CELL}" in deck


def test_deck_sio2_average_ion_charge(deck: str):
    # (Z_Si + 2 Z_O) / 3 = (14 + 16)/3 = 10.
    assert "charge                   = 10.0" in deck


def test_deck_p_polarisation_default(deck: str):
    assert 'polarization_phi = 0.0' in deck


def test_missing_smilei_raises_not_available():
    backend = SmileiBackend(executable="definitely-not-smilei-bin")
    laser = Laser(a0=1.0)
    target = Target.sio2()
    with pytest.raises(SmileiNotAvailable):
        backend.simulate(laser, target, model="rom", numerics=None)


def test_surface_pipeline_accepted_as_model(deck: str):
    # The deck itself is identical for ROM / CSE / surface_pipeline — the model
    # name only routes inside the backend. But the guard in simulate() must
    # accept surface_pipeline.
    backend = SmileiBackend(executable="/bin/false")  # valid path, will error later
    with pytest.raises(RuntimeError):
        backend.simulate(
            Laser(a0=1.0), Target.sio2(), model="surface_pipeline", numerics=None,
        )
