"""Save/load roundtrip tests."""

import numpy as np
import xarray as xr

from harmonyemissions import Laser, Target, load_result, simulate
from harmonyemissions.models.base import Result


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


def test_legacy_result_loads_chf3d_fields_as_none(tmp_path):
    """Old runs that predate chf3d must still load — new fields default to None."""
    r = simulate(Laser(a0=10.0), Target.overdense(100.0), model="rom")
    path = tmp_path / "legacy.h5"
    r.save(path)
    r2 = load_result(path)
    assert r2.chf_focal_volume is None
    assert r2.per_beam_far_field is None
    assert r2.beam_array_geometry is None


def test_chf3d_focal_volume_roundtrip(tmp_path):
    """Synthetic Result with all chf3d fields populated must round-trip."""
    spec = xr.DataArray(
        np.array([1.0, 0.5, 0.25]),
        coords={"harmonic": np.array([1, 2, 3])},
        dims=["harmonic"],
        name="spectrum",
    )
    n_diag, nx, ny, nz, n_beams = 2, 4, 4, 4, 3
    focal = xr.DataArray(
        np.random.default_rng(0).random((n_diag, nx, ny, nz)).astype(np.float64),
        coords={
            "harmonic_diag": np.array([1, 15]),
            "x": np.linspace(-1.0, 1.0, nx),
            "y": np.linspace(-1.0, 1.0, ny),
            "z": np.linspace(-1.0, 1.0, nz),
        },
        dims=["harmonic_diag", "x", "y", "z"],
        name="chf_focal_volume",
        attrs={"units": "arb.", "extent_um": 1.0},
    )
    per_beam = xr.DataArray(
        np.random.default_rng(1).random((n_beams, n_diag, 8, 8)).astype(np.float64),
        coords={
            "beam_index": np.arange(n_beams),
            "harmonic_diag": np.array([1, 15]),
            "yi": np.arange(8),
            "xi": np.arange(8),
        },
        dims=["beam_index", "harmonic_diag", "yi", "xi"],
        name="per_beam_far_field",
    )
    geom = {
        "geometry": "tetrahedral",
        "placement": "faces",
        "n_beams": 3,
        "directions": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "polarization_mode": "uniform_p",
    }
    r = Result(
        spectrum=spec,
        chf_focal_volume=focal,
        per_beam_far_field=per_beam,
        beam_array_geometry=geom,
        chf_gain={"Gamma_D": 12.0, "Gamma_3D_coherent": 144.0, "n_beams": 3},
    )

    path = tmp_path / "chf3d.h5"
    r.save(path)
    r2 = load_result(path)

    assert r2.chf_focal_volume is not None
    np.testing.assert_allclose(r2.chf_focal_volume.values, focal.values)
    assert r2.chf_focal_volume.attrs["units"] == "arb."

    assert r2.per_beam_far_field is not None
    assert r2.per_beam_far_field.shape == (n_beams, n_diag, 8, 8)
    np.testing.assert_allclose(r2.per_beam_far_field.values, per_beam.values)

    assert r2.beam_array_geometry is not None
    assert r2.beam_array_geometry["geometry"] == "tetrahedral"
    assert r2.beam_array_geometry["n_beams"] == 3
    assert r2.beam_array_geometry["directions"] == [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ]
    assert r2.chf_gain["Gamma_3D_coherent"] == 144.0


def test_chf3d_partial_population(tmp_path):
    """Setting only beam_array_geometry (not the heavy arrays) must still round-trip."""
    spec = xr.DataArray(
        np.array([1.0]), coords={"harmonic": np.array([1])}, dims=["harmonic"],
        name="spectrum",
    )
    r = Result(
        spectrum=spec,
        beam_array_geometry={"geometry": "icosahedral", "n_beams": 20},
    )
    path = tmp_path / "geom_only.h5"
    r.save(path)
    r2 = load_result(path)
    assert r2.chf_focal_volume is None
    assert r2.per_beam_far_field is None
    assert r2.beam_array_geometry == {"geometry": "icosahedral", "n_beams": 20}
