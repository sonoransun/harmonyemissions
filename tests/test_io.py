"""Save/load roundtrip tests."""

import numpy as np

from harmonyemissions import Laser, Target, load_result, simulate


def test_save_load_roundtrip(tmp_path):
    laser = Laser(a0=10.0)
    target = Target.overdense(100.0, 0.05)
    r = simulate(laser, target, model="rom")
    path = tmp_path / "run.h5"
    r.save(path)
    r2 = load_result(path)
    np.testing.assert_allclose(r.spectrum.values, r2.spectrum.values)
    assert r2.diagnostics["gamma_mirror_peak"] == r.diagnostics["gamma_mirror_peak"]
    assert r2.provenance["model"] == "rom"


def test_save_bgp_without_time_field(tmp_path):
    r = simulate(Laser(a0=10.0), Target.overdense(100.0), model="bgp")
    path = tmp_path / "bgp.h5"
    r.save(path)
    r2 = load_result(path)
    assert r2.time_field is None
    assert r2.attosecond_pulse is None
    assert r2.spectrum.size == r.spectrum.size
