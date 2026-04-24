"""Tests for runner.simulate and simulate_from_config dispatch + validation."""

import numpy as np
import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.config import (
    LaserConfig,
    NumericsConfig,
    RunConfig,
    TargetConfig,
)
from harmonyemissions.runner import simulate_from_config


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        simulate(Laser(a0=1.0), Target.overdense(100.0), model="rom", backend="zzz")


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        simulate(Laser(a0=1.0), Target.overdense(100.0), model="nope")


def test_direct_vs_config_equivalence():
    laser = Laser(a0=5.0, wavelength_um=0.8, duration_fs=8.0)
    target = Target.overdense(100.0, 0.1)
    r_direct = simulate(laser, target, model="bgp")

    cfg = RunConfig(
        model="bgp",
        laser=LaserConfig(a0=5.0, wavelength_um=0.8, duration_fs=8.0),
        target=TargetConfig(kind="overdense", n_over_nc=100.0, gradient_L_over_lambda=0.1),
        numerics=NumericsConfig(),
    )
    r_cfg = simulate_from_config(cfg)
    np.testing.assert_allclose(r_direct.spectrum.values, r_cfg.spectrum.values)


def test_simulate_provenance_captures_inputs():
    laser = Laser(a0=3.0)
    target = Target.overdense(100.0)
    r = simulate(laser, target, model="bgp")
    assert r.provenance["model"] == "bgp"
    assert r.provenance["backend"] == "analytical"
    assert r.provenance["laser"]["a0"] == 3.0
    assert r.provenance["target"]["n_over_nc"] == 100.0


def test_regime_mismatch_rejected_by_config():
    # surface model with a gas target must be flagged by the RunConfig validator.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunConfig(
            model="rom",
            laser=LaserConfig(a0=1.0),
            target=TargetConfig(kind="gas", gas_species="Ar"),
        )


def test_simulate_forwards_numerics():
    """NumericsConfig.n_periods should control the time grid used by ROM."""
    laser = Laser(a0=2.0, duration_fs=5.0)
    target = Target.overdense(100.0, 0.05)
    r_short = simulate(laser, target, model="rom",
                       numerics=NumericsConfig(n_periods=5.0, samples_per_period=128))
    r_long = simulate(laser, target, model="rom",
                      numerics=NumericsConfig(n_periods=40.0, samples_per_period=128))
    # Longer grid → more samples in the time-domain field.
    assert r_long.time_field.size > r_short.time_field.size
