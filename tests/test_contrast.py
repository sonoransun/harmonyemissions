"""Tests for the DPM contrast model."""


from harmonyemissions.contrast import (
    L0_LAMBDA,
    L_INF_LAMBDA,
    ContrastInputs,
    optimum_prepulse_delay_window,
    scale_length,
    scale_length_from_prepulse,
    scale_length_from_thdr,
)


def test_thdr_monotone_and_bounded():
    lengths = [scale_length_from_thdr(t) for t in [100, 250, 400, 711, 2000, 10000]]
    # Monotonically increasing with t_HDR.
    for a, b in zip(lengths, lengths[1:], strict=False):
        assert b >= a
    # Bounded between L0 and L_INF.
    for L in lengths:
        assert L0_LAMBDA - 1e-12 <= L <= L_INF_LAMBDA + 1e-12


def test_paper_optimum_near_351_fs():
    """351 fs t_HDR — the paper's sweet spot — should yield L ≈ 0.1–0.2 λ."""
    L = scale_length_from_thdr(351)
    assert 0.08 <= L <= 0.20


def test_long_thdr_over_expanded():
    """711 fs t_HDR should give noticeably larger L than 351 fs."""
    short = scale_length_from_thdr(351)
    long = scale_length_from_thdr(711)
    assert long > short * 1.5


def test_prepulse_adds_scale_length():
    base = scale_length_from_thdr(351)
    total = scale_length(
        ContrastInputs(
            t_HDR_fs=351,
            prepulse_intensity_rel=1e-3,
            prepulse_delay_fs=1000.0,
        )
    )
    assert total > base
    # Sensible magnitude: nanosecond prepulses blow the surface apart
    # (L grows by > fraction of a wavelength).
    assert total - base < 1.5


def test_zero_prepulse():
    assert scale_length_from_prepulse(0.0, 500.0) == 0.0
    assert scale_length_from_prepulse(1e-3, 0.0) == 0.0


def test_optimum_window_shrinks_at_low_intensity():
    low = optimum_prepulse_delay_window(1e19)
    high = optimum_prepulse_delay_window(1e21)
    # Window at higher intensity should be wider.
    assert (high[1] - high[0]) > (low[1] - low[0])
