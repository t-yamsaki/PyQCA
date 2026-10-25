"""Independent anchor and continuity checks for direct logistic calibration."""

import numpy as np
import pandas as pd
import pytest

from qca.calibration import calibrate_logistic, logistic_calibrate_series
from qca.calibration.logistic import _estimate_logistic_params


@pytest.mark.parametrize("anchors", [(0, 1, 10), (-10, -1, 0), (1, 2, 3)])
@pytest.mark.parametrize("bounds", [(0.05, 0.95), (0.1, 0.8), (1e-12, 1 - 1e-12)])
def test_all_three_anchors_hit_requested_memberships(anchors, bounds):
    result, _ = logistic_calibrate_series(pd.Series(anchors), *anchors, *bounds)
    np.testing.assert_allclose(result, [bounds[0], 0.5, bounds[1]], rtol=1e-10, atol=0)


def test_asymmetric_metadata_reconstructs_curve():
    raw = pd.Series([-2.0, 0.0, 0.5, 1.0, 5.5, 10.0, 20.0])
    result, anchor = logistic_calibrate_series(raw, 0, 1, 10)
    metadata = anchor.to_dict()
    assert anchor.slope is None and anchor.intercept is None
    slopes = np.where(
        raw <= anchor.crossover, metadata["slope_below"], metadata["slope_above"]
    )
    expected = 1 / (1 + np.exp(-slopes * (raw - anchor.crossover)))
    np.testing.assert_allclose(result, expected)
    # Midpoints on each log-odds segment have odds sqrt(1/19), sqrt(19).
    assert result.iloc[2] == pytest.approx(1 / (1 + np.sqrt(19)))
    assert result.iloc[4] == pytest.approx(np.sqrt(19) / (1 + np.sqrt(19)))


def test_symmetric_metadata_preserves_single_curve():
    result, anchor = logistic_calibrate_series(pd.Series([1.0, 2.0, 3.0]), 1, 2, 3)
    intercept, slope = _estimate_logistic_params(1, 2, 3)
    assert anchor.slope == pytest.approx(slope)
    assert anchor.intercept == pytest.approx(intercept)
    np.testing.assert_allclose(result, [0.05, 0.5, 0.95])
    with pytest.raises(ValueError, match="two slopes"):
        _estimate_logistic_params(0, 1, 10)


def test_quantile_anchors_and_metadata_survive_high_level_api():
    data = pd.DataFrame({"raw": [0.0, 0.5, 1.0, 4.0, 10.0, np.nan, np.inf]})
    result = calibrate_logistic(data, "raw", anchor_quantiles=(0, 0.5, 1))
    assert result.anchor.anchor_quantiles == (0, 0.5, 1)
    np.testing.assert_allclose(result.df.loc[[0, 2, 4], "raw_fuzzy"], [0.05, 0.5, 0.95])
    assert result.anchor.slope_below == pytest.approx(np.log(19))
    assert result.anchor.slope_above == pytest.approx(np.log(19) / 9)
    assert result.n_calibrated == 5 and result.n_skipped == 2
    assert "raw_fuzzy" not in data


@pytest.mark.parametrize(
    "bounds",
    [(0.5, 0.9), (0.6, 0.9), (0.1, 0.5), (0.1, 0.4), (np.nan, 0.9), (0.1, np.inf)],
)
def test_memberships_must_straddle_crossover(bounds):
    with pytest.raises(ValueError):
        logistic_calibrate_series(pd.Series([0, 1, 10]), 0, 1, 10, *bounds)
    with pytest.raises(ValueError):
        calibrate_logistic(
            pd.DataFrame({"raw": [0, 1, 10]}),
            "raw",
            full_out=0,
            crossover=1,
            full_in=10,
            mem_full_out=bounds[0],
            mem_full_in=bounds[1],
            min_nonnull=1,
        )


def test_continuity_monotonicity_and_nonfinite_inputs():
    raw = pd.Series(
        [-1e300, 0, 1 - 1e-9, 1, 1 + 1e-9, 10, 1e300, np.nan, np.inf, -np.inf],
        index=range(10, 20),
    )
    result, _ = logistic_calibrate_series(raw, 0, 1, 10)
    assert result.index.equals(raw.index)
    assert result.iloc[:7].is_monotonic_increasing
    assert np.isfinite(result.iloc[:7]).all()
    assert result.iloc[7:].isna().all()
    np.testing.assert_allclose(result.iloc[2:5], 0.5, atol=1e-8)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_large_finite_centering_does_not_cause_false_saturation(sign):
    anchors = sorted(sign * np.array([-1.5e308, -1e308, -5e307]))
    raw = pd.Series([sign * 1e308, anchors[1], np.nan, np.inf, -np.inf])
    result, _ = logistic_calibrate_series(raw, *anchors, 0.49, 0.51)
    # Distance/gap is exactly 4 in real arithmetic. Compute reference odds
    # without either the production slope or the overflowing subtraction.
    odds = (0.51 / 0.49) ** (4 * sign)
    assert result.iloc[0] == pytest.approx(odds / (1 + odds), abs=1e-14)
    assert result.iloc[1] == 0.5
    assert result.iloc[2:].isna().all()


def test_overflowing_centering_with_true_tail_saturation():
    crossover = -1e308
    anchors = (
        np.nextafter(crossover, -np.inf),
        crossover,
        np.nextafter(crossover, np.inf),
    )
    result, _ = logistic_calibrate_series(pd.Series([1e308]), *anchors)
    assert result.iloc[0] == np.nextafter(1.0, 0.0)
