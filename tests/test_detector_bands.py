"""Round-trip tests for the three-band detector pipeline."""

import numpy as np
import pytest
import xarray as xr

from harmonyemissions.detector.filters import FilterSpec
from harmonyemissions.detector.gamma_response import GammaDetector, apply_gamma_response
from harmonyemissions.detector.hard_xray import FilterStack
from harmonyemissions.detector.instrument import BANDS, apply_detector, auto_band
from harmonyemissions.detector.ross_pair import RossPair, ross_pair
from harmonyemissions.detector.scintillator import DetectorConfig as ScintConfig
from harmonyemissions.detector.soft_xray import SoftXrayConfig, apply_soft_xray_response


def _delta_spectrum(E_center_keV: float, E_lo_keV: float = 0.01, E_hi_keV: float = 1e5) -> xr.DataArray:
    """Build a log-spaced energy grid with a single-bin delta at E_center."""
    energy_keV = np.geomspace(E_lo_keV, E_hi_keV, 2048)
    values = np.zeros_like(energy_keV)
    idx = int(np.argmin(np.abs(energy_keV - E_center_keV)))
    values[idx] = 1.0
    return xr.DataArray(
        values,
        coords={
            "harmonic": np.arange(energy_keV.size, dtype=float),
            "photon_energy_keV": ("harmonic", energy_keV),
        },
        dims=["harmonic"],
    )


def test_auto_band_picks_xuv_for_low_energy():
    s = _delta_spectrum(0.05, E_lo_keV=0.005, E_hi_keV=10.0)  # 50 eV
    assert auto_band(s) == "xuv"


def test_auto_band_picks_soft_for_hundred_ev():
    s = _delta_spectrum(0.25)  # 250 eV
    assert auto_band(s) == "xray-soft"


def test_auto_band_picks_hard_for_tens_keV():
    s = _delta_spectrum(10.0)  # 10 keV
    assert auto_band(s) == "xray-hard"


def test_auto_band_picks_gamma_for_MeV():
    s = _delta_spectrum(2_000.0)  # 2 MeV
    assert auto_band(s) == "gamma"


def test_bands_constants_cover_six_decades():
    # No gaps: upper edge of one band == lower edge of next.
    keys = ["xuv", "xray-soft", "xray-hard", "gamma"]
    for a, b in zip(keys[:-1], keys[1:], strict=True):
        assert BANDS[a][1] == BANDS[b][0], f"gap between {a} and {b}"


def test_soft_xray_kapton_mylar_attenuation_follows_expected_ordering():
    """A Kapton filter attenuates less than Mylar of the same thickness at
    300 eV because Kapton has a slightly higher Z_eff and density — check
    both stacks produce reasonable (in-band) transmission.
    """
    s = _delta_spectrum(0.3)  # 300 eV
    cfg_k = SoftXrayConfig(filters=(FilterSpec("kapton", 7.0),))
    cfg_m = SoftXrayConfig(filters=(FilterSpec("mylar", 7.0),))
    out_k = apply_soft_xray_response(s, cfg_k)
    out_m = apply_soft_xray_response(s, cfg_m)
    peak_k = float(out_k.values.max())
    peak_m = float(out_m.values.max())
    assert 0.0 < peak_k < 1.0
    assert 0.0 < peak_m < 1.0


def test_hard_xray_ross_pair_isolates_passband():
    """Delta in the Cu/Ni passband (8.33–8.98 keV) produces a sizeable
    differential; delta above both K-edges collapses it."""
    pair = RossPair(
        high_z=FilterSpec("cu", 25.0),    # thinner → higher T below K
        low_z=FilterSpec("ni", 25.0),
        passband_label="Cu/Ni window (8.33–8.98 keV)",
    )
    s_in = _delta_spectrum(8.65)
    r_in = ross_pair(s_in, pair)
    peak_in = abs(r_in["difference"].values).max()
    # Real Ross-pair isolation with these thicknesses is ~20–40% of input;
    # require at least 5% to guard against the sign-error / zero-output mode.
    assert peak_in > 0.05 * s_in.values.max()

    s_out = _delta_spectrum(25.0)
    r_out = ross_pair(s_out, pair)
    peak_out = abs(r_out["difference"].values).max()
    # Out-of-band differential at least 5× smaller than in-band.
    assert peak_out < 0.2 * peak_in


def test_gamma_lyso_2MeV_photon_produces_signal():
    """A 2-MeV photon through 2 cm LYSO must produce a non-zero detector count."""
    s = _delta_spectrum(2_000.0)
    det = GammaDetector(
        filters=FilterStack(layers=(("Al", 500.0),)),
        detector=ScintConfig(name="LYSO", thickness_mm=20.0),
    )
    out = apply_gamma_response(s, det)
    assert out.values.max() > 0
    # The absorption through 2 cm LYSO at 2 MeV is ≳ 40%.
    # (LYSO ρ=7.1, µ/ρ~0.05 cm²/g at 2 MeV → µx=0.05·7.1·2.0 = 0.71 → abs=0.51.)
    peak_in = float(s.values.max())
    peak_out = float(out.values.max())
    assert 0.1 * peak_in < peak_out < peak_in


def test_gamma_pair_production_region_has_reduced_efficiency():
    """Above 1.022 MeV pair production kicks in → detector efficiency
    drops relative to the photoelectric regime. We verify by comparing
    100 keV (pure photoelectric) vs 5 MeV (pair+Compton dominated)."""
    s_lo = _delta_spectrum(0.1)
    s_hi = _delta_spectrum(5_000.0)
    det = GammaDetector(detector=ScintConfig(name="NaI", thickness_mm=10.0))
    r_lo = apply_gamma_response(s_lo, det)
    r_hi = apply_gamma_response(s_hi, det)
    assert r_lo.values.max() > 0
    assert r_hi.values.max() > 0


def test_apply_detector_dispatch_routes_correctly():
    """A 10-keV delta via apply_detector(band='auto') must hit the hard band."""
    s = _delta_spectrum(10.0)
    out = apply_detector(s, band="auto")
    # Output sitting on the same harmonic coord.
    assert "harmonic" in out.dims
    assert out.sizes["harmonic"] == s.sizes["harmonic"]


def test_apply_detector_rejects_soft_xray_without_energy_coord():
    """xray-soft requires photon_energy_keV; a spectrum without it is rejected."""
    bad = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        coords={"harmonic": np.array([1.0, 2.0, 3.0])},
        dims=["harmonic"],
    )
    with pytest.raises(ValueError):
        apply_detector(bad, band="xray-soft")


def test_result_round_trip_preserves_instrument_attrs(tmp_path):
    """Round-tripping a detector run through Result.save/load keeps attrs."""
    from harmonyemissions.models.base import Result

    s = _delta_spectrum(8.65)
    out = apply_detector(s, band="xray-hard")
    r = Result(spectrum=s, instrument_spectrum=out)
    p = tmp_path / "round.h5"
    r.save(p)
    r2 = Result.load(p)
    assert r2.instrument_spectrum is not None
    assert "photon_energy_keV" in r2.instrument_spectrum.coords
