"""Unit changes must not alter feature-effect directions or calibration."""

import numpy as np
import pandas as pd
import pytest

from qca import (
    MLConditionSpec,
    MLQCAConfig,
    PredictorFitResult,
    calibrate_crisp_conditions,
    calibrate_fuzzy_conditions,
    extract_xgboost_importance,
    fit_xgboost_predictor,
    propose_crisp_calibrations,
    propose_fuzzy_anchors,
    validate_mlqca_input,
)


class EmptyBooster:
    feature_names = ["A"]

    def get_score(self, importance_type):
        return {}


def importance(x, y):
    return extract_xgboost_importance(
        EmptyBooster(),
        ["A"],
        contributions=np.column_stack([y, np.zeros(len(y))]),
        feature_values=pd.DataFrame({"A": x}),
    ).iloc[0]


@pytest.mark.parametrize("x_scale", [1.0, 1e-10, 1e-200, 1e200])
@pytest.mark.parametrize("y_scale", [1.0, 1e-200, 1e200])
@pytest.mark.parametrize("sign", [1, -1])
def test_independent_positive_scales_preserve_correlation(x_scale, y_scale, sign):
    row = importance(np.arange(4.0) * x_scale, sign * np.arange(4.0) * y_scale)
    assert row.shap_direction_score == pytest.approx(sign, abs=1e-14)
    assert row.shap_direction == ("high" if sign == 1 else "low")


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 1, 1], [1, 2, 3], None),
        ([1, 2, 3], [1, 1, 1], None),
        ([-1, 0, 1], [1, -2, 1], 0.0),
        ([1, np.nan, 2, np.inf], [3, 1, 4, 5], 1.0),
        ([1, 2, 3], [np.inf, 2, np.nan], None),
        ([np.nan, np.inf], [1, 2], None),
        ([1], [2], None),
        ([-1e308, 0, 1e308], [-1e308, 0, 1e308], 1.0),
        ([0, np.nextafter(0.0, 1.0)], [0, np.nextafter(0.0, 1.0)], 1.0),
        ([1e200, np.nextafter(1e200, np.inf)], [1, 2], 1.0),
    ],
)
def test_constant_zero_missing_and_extreme_boundaries(x, y, expected):
    with np.errstate(over="ignore", invalid="ignore"):
        row = importance(x, y)
    if expected is None:
        assert np.isnan(row.shap_direction_score)
        assert row.shap_direction is None
    else:
        assert row.shap_direction_score == pytest.approx(expected, abs=1e-14)
        assert row.shap_direction == ("high" if expected else None)


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("mean", [-1.0, 1.0, -1e-20, 1e-20])
def test_undefined_direction_never_uses_mean_and_explicit_priority(fuzzy, mean):
    data = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0], "Y": [0, 0, 1, 1]})
    predictor = PredictorFitResult(
        backend="test",
        model=object(),
        feature_importance=pd.DataFrame(
            {"feature": ["A"], "shap_mean": [mean], "shap_direction": [None]}
        ),
        cutoff_candidates=pd.DataFrame(
            {"feature": ["A"] * 3, "cutoff": [0.5, 1.5, 2.5], "cutoff_rank": [1, 2, 3]}
        ),
        predictions=pd.DataFrame(),
    )
    propose = propose_fuzzy_anchors if fuzzy else propose_crisp_calibrations
    validated = validate_mlqca_input(data, "Y")
    with pytest.raises(ValueError, match="Cannot infer calibration direction"):
        propose(validated, predictor)
    predictor.feature_importance["shap_direction"] = "high"
    schema = validate_mlqca_input(
        data, "Y", condition_specs=[MLConditionSpec("A", direction="low")]
    )
    assert propose(schema, predictor).iloc[0].direction == "low"
    result = propose(schema, predictor, direction_overrides={"A": "high"})
    assert result.iloc[0].direction == "high"
    assert result.iloc[0].direction_source == "override"


@pytest.mark.parametrize("sign", [1, -1])
def test_real_xgboost_proposals_and_calibration_preserve_units(sign):
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    outputs = []
    for scale in [1.0, 1e-10]:
        x = np.arange(40, dtype=float)
        target = (x >= 20).astype(int)
        if sign < 0:
            target = 1 - target
        validated = validate_mlqca_input(
            pd.DataFrame({"A": x * scale, "Y": target}), "Y"
        )
        predictor = fit_xgboost_predictor(
            validated,
            MLQCAConfig(
                n_jobs=1,
                random_state=7,
                model_params={"n_estimators": 12, "max_depth": 2},
            ),
        )
        direction = "high" if sign > 0 else "low"
        assert predictor.feature_importance.iloc[0].shap_direction == direction
        crisp = propose_crisp_calibrations(validated, predictor)
        fuzzy = propose_fuzzy_anchors(
            validated, predictor, anchor_overrides={"A": np.array([5, 20, 35]) * scale}
        )
        assert crisp.iloc[0].direction == fuzzy.iloc[0].direction == direction
        assert crisp.iloc[0].threshold / scale == pytest.approx(20, rel=1e-6)
        outputs.append(
            (
                calibrate_crisp_conditions(validated, crisp).data["A__crisp"],
                calibrate_fuzzy_conditions(validated, fuzzy).data["A__fuzzy"],
                predictor.predictions.probability,
            )
        )
    for baseline, scaled in zip(*outputs, strict=True):
        np.testing.assert_allclose(baseline, scaled, rtol=1e-6, atol=1e-12)
