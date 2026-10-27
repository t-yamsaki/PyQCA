"""Piecewise membership endpoints must straddle the crossover (#34)."""

import numpy as np
import pandas as pd
import pytest

from qca.calibration import (
    calibrate_piecewise,
    piecewise_fuzzy_scalar,
    piecewise_fuzzy_series,
)


@pytest.mark.parametrize(
    "bounds",
    [
        (0.6, 0.9),
        (0.1, 0.4),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.5, 0.5),
        (0.9, 0.1),
        (-0.1, 0.9),
        (0.1, 1.1),
        (np.nan, 0.9),
        (0.1, np.nan),
        (-np.inf, 0.9),
        (np.inf, 0.9),
        (0.1, np.inf),
        (0.1, -np.inf),
    ],
)
@pytest.mark.parametrize("api", ["scalar", "series", "calibrate"])
def test_invalid_bounds_are_rejected_in_every_piecewise_api(bounds, api):
    kwargs = dict(lower=bounds[0], upper=bounds[1])
    with pytest.raises(ValueError, match="Piecewise membership bounds"):
        if api == "scalar":
            piecewise_fuzzy_scalar(1, 0, 1, 2, **kwargs)
        elif api == "series":
            piecewise_fuzzy_series(pd.Series([0, 1, 2]), 0, 1, 2, **kwargs)
        else:
            calibrate_piecewise(
                pd.DataFrame({"x": [0, 1, 2]}),
                "x",
                quantiles=(0, 0.5, 1),
                min_nonnull=3,
                **kwargs,
            )


@pytest.mark.parametrize(
    "bounds",
    [
        (0.05, 0.95),
        (0.1, 0.8),
        (0.0, 1.0),
        (np.nextafter(0.5, 0), np.nextafter(0.5, 1)),
    ],
)
def test_valid_bounds_give_bounded_monotone_memberships(bounds):
    lower, upper = bounds
    values = pd.Series([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    result = piecewise_fuzzy_series(values, 0, 1, 2, lower, upper)
    scalar = [piecewise_fuzzy_scalar(x, 0, 1, 2, lower, upper) for x in values]
    assert result.tolist() == pytest.approx(scalar)
    assert result.is_monotonic_increasing
    assert result.between(lower, upper).all()
    assert result.iloc[[1, 3, 5]].tolist() == pytest.approx([lower, 0.5, upper])
    calibrated = calibrate_piecewise(
        pd.DataFrame({"x": [0, 1, 2]}),
        "x",
        quantiles=(0, 0.5, 1),
        min_nonnull=3,
        lower=lower,
        upper=upper,
    )
    assert calibrated.df.x_fuzzy.tolist() == pytest.approx([lower, 0.5, upper])


def test_default_bounds_are_unchanged():
    assert piecewise_fuzzy_series(
        pd.Series([0, 1, 2]), 0, 1, 2
    ).tolist() == pytest.approx([0.05, 0.5, 0.95])
    assert piecewise_fuzzy_scalar(1, 0, 1, 2) == 0.5


def test_invalid_bounds_are_rejected_even_with_no_calibratable_rows():
    with pytest.raises(ValueError, match="Piecewise membership bounds"):
        calibrate_piecewise(pd.DataFrame({"x": [np.nan]}), "x", lower=0.6, upper=0.9)
    with pytest.raises(ValueError, match="Piecewise membership bounds"):
        piecewise_fuzzy_series(pd.Series([], dtype=float), 0, 1, 2, 0.1, 0.4)
