"""Tests for FFT / bandpass / pulse-synthesis utilities."""

import numpy as np

from harmonyemissions.spectrum import (
    attosecond_pulse,
    bandpass_field,
    field_to_spectrum,
    time_field,
)


def test_single_frequency_field_gives_peak_at_its_harmonic():
    t = np.linspace(0, 10, 1024, endpoint=False)  # 10 periods
    field = np.cos(2 * np.pi * 7.0 * t)  # 7th harmonic
    spec = field_to_spectrum(t, field)
    peak_idx = int(np.argmax(spec.values))
    peak_n = float(spec.coords["harmonic"].values[peak_idx])
    assert abs(peak_n - 7.0) < 0.5


def test_bandpass_keeps_in_band_rejects_out_of_band():
    t = np.linspace(0, 20, 2048, endpoint=False)
    f_in = np.cos(2 * np.pi * 12.0 * t)
    f_out = np.cos(2 * np.pi * 3.0 * t)
    total = f_in + f_out
    filtered = bandpass_field(t, total, n_low=9.0, n_high=15.0)
    # Filtered should have almost no 3-ω content; easy check via inner product.
    assert np.abs(np.mean(filtered * f_out)) < np.abs(np.mean(filtered * f_in)) * 0.1


def test_attosecond_pulse_roundtrip_shape():
    t = np.linspace(0, 10, 1024, endpoint=False)
    f = sum(np.cos(2 * np.pi * n * t) for n in range(15, 30))
    pulse = attosecond_pulse(t, f, (15.0, 30.0))
    assert pulse is not None
    # Pulse should be concentrated — peak much larger than RMS.
    p = pulse.values
    assert np.max(np.abs(p)) > 3.0 * np.sqrt(np.mean(p**2))


def test_attosecond_pulse_none_without_window():
    t = np.linspace(0, 5, 512, endpoint=False)
    f = np.cos(2 * np.pi * 5 * t)
    assert attosecond_pulse(t, f, None) is None


def test_time_field_wraps_as_dataarray():
    t = np.array([0.0, 0.1, 0.2])
    f = np.array([1.0, 2.0, 3.0])
    da = time_field(t, f)
    assert da.dims == ("t_over_T0",)
    assert list(da.values) == [1.0, 2.0, 3.0]
