"""Piecewise calibration respects representable anchor spacing (#33)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from qca import AnchorSensitivity
from qca.calibration import (
    calibrate_piecewise,
    piecewise_fuzzy_scalar,
    piecewise_fuzzy_series,
)


@pytest.mark.parametrize("scale", [1.0, 1e-13, 1e-300, 1e300])
def test_scale_invariant_anchors_interiors_and_continuity(scale):
    anchors = np.array([0.0, 1.0, 2.0]) * scale
    values = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]) * scale
    expected = [0.05, 0.05, 0.275, 0.5, 0.725, 0.95, 0.95]
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = piecewise_fuzzy_series(pd.Series(values), *anchors)
        scalar = [piecewise_fuzzy_scalar(x, *anchors) for x in values]
    assert actual.tolist() == pytest.approx(expected)
    assert scalar == pytest.approx(expected)
    assert actual.is_monotonic_increasing
    for anchor, membership in zip(anchors, [0.05, 0.5, 0.95], strict=True):
        neighbors = [
            np.nextafter(anchor, -np.inf),
            anchor,
            np.nextafter(anchor, np.inf),
        ]
        result = piecewise_fuzzy_series(pd.Series(neighbors), *anchors)
        assert result.tolist() == pytest.approx([membership] * 3, abs=1e-14)
        assert [
            piecewise_fuzzy_scalar(x, *anchors) for x in neighbors
        ] == pytest.approx(result.tolist())


@pytest.mark.parametrize(
    "anchors",
    [
        (0.0, np.nextafter(0.0, 1.0), 2 * np.nextafter(0.0, 1.0)),
        (-1.7e308, 1e308, 1.7e308),
        (-1.7e308, -1e308, 1.7e308),
    ],
)
def test_extreme_representable_anchors_are_exact(anchors):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = piecewise_fuzzy_series(pd.Series(anchors), *anchors)
        scalar = [piecewise_fuzzy_scalar(x, *anchors) for x in anchors]
    assert result.tolist() == pytest.approx([0.05, 0.5, 0.95])
    assert scalar == pytest.approx([0.05, 0.5, 0.95])


def test_overflowing_width_interpolation_and_nan():
    anchors = (-1.7e308, 1e308, 1.7e308)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = piecewise_fuzzy_series(pd.Series([0.0, np.nan]), *anchors)
        scalar = piecewise_fuzzy_scalar(0.0, *anchors)
    assert result.iloc[0] == pytest.approx(0.05 + (1.7 / 2.7) * 0.45)
    assert scalar == pytest.approx(result.iloc[0])
    assert np.isnan(result.iloc[1])


@pytest.mark.parametrize(
    "anchors",
    [
        (0, 0, 1),
        (0, 1, 1),
        (2, 1, 0),
        (0, np.nan, 2),
        (-np.inf, 0, 1),
        (0, 1, np.inf),
        (1.0, 1.0 + 1e-20, 2.0),
    ],
)
def test_invalid_or_rounded_together_anchors_are_rejected(anchors):
    with pytest.raises(ValueError, match="anchors"):
        piecewise_fuzzy_scalar(0.5, *anchors)
    with pytest.raises(ValueError, match="anchors"):
        piecewise_fuzzy_series(pd.Series([0.5]), *anchors)


@pytest.mark.parametrize("scale", [1.0, 1e-13])
def test_high_level_quantile_calibration(scale):
    result = calibrate_piecewise(
        pd.DataFrame({"x": np.array([0.0, 1.0, 2.0]) * scale}),
        "x",
        quantiles=(0, 0.5, 1),
        min_nonnull=3,
    )
    assert result.df.x_fuzzy.tolist() == pytest.approx([0.05, 0.5, 0.95])
    assert result.n_calibrated == 3


def test_anchor_sensitivity_retains_memberships_after_unit_conversion():
    frames = []
    for scale in [1.0, 1e-13]:
        sweep = AnchorSensitivity(
            pd.DataFrame(
                {"A": np.array([0.0, 1.0, 2.0]) * scale, "Y": [0.0, 0.5, 1.0]}
            ),
            outcome="Y",
            conditions=["A"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = sweep.full_membership(
                "A",
                values=[2 * scale],
                full_out=0,
                crossover=scale,
            )
        assert result.calibrated_frames
        frames.append(next(iter(result.calibrated_frames.values())))
    assert frames[0]["A_fuzzy"].tolist() == pytest.approx([0.05, 0.5, 0.95])
    pd.testing.assert_series_equal(frames[0]["A_fuzzy"], frames[1]["A_fuzzy"])
