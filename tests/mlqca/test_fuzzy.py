"""mlQCA fuzzy anchor and sensitivity tests."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from qca import (
    FuzzyCalibrationSet,
    MLConditionSpec,
    MLQCAAnchorSensitivityResult,
    MLQCAFSQCAResult,
    PredictorFitResult,
    calibrate_fuzzy_conditions,
    fit_fsqca_from_calibration,
    fit_fsqca_from_predictor,
    proposal_anchor_grid,
    propose_fuzzy_anchors,
    run_anchor_sensitivity,
    validate_mlqca_input,
)


@pytest.fixture
def fuzzy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 13)],
            "resources": list(range(1, 13)),
            "burden": list(range(12, 0, -1)),
            "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def fuzzy_predictor() -> PredictorFitResult:
    return PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=pd.DataFrame(
            {
                "feature": ["resources", "burden"],
                "shap_mean_abs": [1.0, 0.8],
                "shap_mean": [0.4, -0.3],
                "shap_direction": ["high", "low"],
                "used": [True, True],
            }
        ),
        cutoff_candidates=pd.DataFrame(
            {
                "feature": [
                    "resources",
                    "resources",
                    "resources",
                    "burden",
                    "burden",
                    "burden",
                ],
                "cutoff": [3.0, 6.0, 10.0, 3.0, 6.0, 10.0],
                "cutoff_rank": [1, 2, 3, 1, 2, 3],
                "count": [5, 4, 3, 5, 4, 3],
                "frequency": [0.2, 0.16, 0.12, 0.2, 0.16, 0.12],
            }
        ),
        predictions=pd.DataFrame(),
    )


def _validated(fuzzy_df, specs=None):
    return validate_mlqca_input(
        fuzzy_df,
        "Y",
        candidates=None if specs else ["resources", "burden"],
        condition_specs=specs,
        case_id="case",
    )


def test_propose_fuzzy_anchors_uses_xgboost_and_direction(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    proposals = propose_fuzzy_anchors(
        _validated(fuzzy_df),
        fuzzy_predictor,
    )

    assert proposals["full_out"].tolist() == [3.0, 3.0]
    assert proposals["crossover"].tolist() == [6.0, 6.0]
    assert proposals["full_in"].tolist() == [10.0, 10.0]
    assert proposals["direction"].tolist() == ["high", "low"]
    assert set(proposals["anchor_source"]) == {"xgboost"}


def test_theoretical_and_override_anchors_take_precedence(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    specs = [
        MLConditionSpec(
            "resources",
            direction="high",
            theoretical_cutoffs=(2, 7, 11),
        ),
        MLConditionSpec("burden", direction="low"),
    ]
    proposals = propose_fuzzy_anchors(
        _validated(fuzzy_df, specs),
        fuzzy_predictor,
        anchor_overrides={"burden": (1, 5, 9)},
    )

    assert proposals.loc[0, "anchor_source"] == "theoretical"
    assert proposals.loc[0, "crossover"] == pytest.approx(7.0)
    assert proposals.loc[1, "anchor_source"] == "override"
    assert proposals.loc[1, "crossover"] == pytest.approx(5.0)


def test_fuzzy_proposal_requires_three_distinct_anchors(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    validated = validate_mlqca_input(
        fuzzy_df,
        "Y",
        candidates=["resources"],
    )
    limited = PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=fuzzy_predictor.feature_importance,
        cutoff_candidates=fuzzy_predictor.cutoff_candidates.head(2),
        predictions=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="Three fuzzy anchors"):
        propose_fuzzy_anchors(validated, limited)


def test_calibration_handles_high_and_low_direction(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    validated = _validated(fuzzy_df)
    calibrated = calibrate_fuzzy_conditions(
        validated,
        propose_fuzzy_anchors(validated, fuzzy_predictor),
    )

    assert isinstance(calibrated, FuzzyCalibrationSet)
    assert calibrated.data["resources__fuzzy"].is_monotonic_increasing
    assert calibrated.data["burden__fuzzy"].is_monotonic_increasing
    fuzzy = calibrated.data[list(calibrated.conditions)]
    assert fuzzy.min().min() >= 0.0
    assert fuzzy.max().max() <= 1.0


def test_fsqca_connections(fuzzy_df, fuzzy_predictor) -> None:
    validated = _validated(fuzzy_df)
    proposals = propose_fuzzy_anchors(validated, fuzzy_predictor)
    calibrated = calibrate_fuzzy_conditions(validated, proposals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = fit_fsqca_from_calibration(
            calibrated,
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )
        combined = fit_fsqca_from_predictor(
            validated,
            fuzzy_predictor,
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(direct, MLQCAFSQCAResult)
    assert direct.engine.qca_type == "fsQCA"
    assert combined.calibration.conditions == (
        "resources__fuzzy",
        "burden__fuzzy",
    )


def test_proposal_anchor_grid_preserves_all_orderings(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    validated = _validated(fuzzy_df)
    proposals = propose_fuzzy_anchors(validated, fuzzy_predictor)
    grid = proposal_anchor_grid(proposals, "resources", span=0.2)

    assert max(grid["full_out"]) < min(grid["crossover"])
    assert max(grid["crossover"]) < min(grid["full_in"])
    assert all(len(values) == 3 for values in grid.values())


def test_anchor_sensitivity_connects_high_direction(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    validated = _validated(fuzzy_df)
    calibration = calibrate_fuzzy_conditions(
        validated,
        propose_fuzzy_anchors(validated, fuzzy_predictor),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_anchor_sensitivity(
            validated,
            calibration,
            "resources",
            anchor_grid={
                "full_out": [3],
                "crossover": [6, 7],
                "full_in": [10],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(result, MLQCAAnchorSensitivityResult)
    assert not result.transformed_for_low_direction
    assert len(result.summary_df) == 2
    assert result.result.settings["mlqca_feature"] == "resources"


def test_anchor_sensitivity_transforms_low_direction(
    fuzzy_df,
    fuzzy_predictor,
) -> None:
    validated = _validated(fuzzy_df)
    calibration = calibrate_fuzzy_conditions(
        validated,
        propose_fuzzy_anchors(validated, fuzzy_predictor),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_anchor_sensitivity(
            validated,
            calibration,
            "burden",
            anchor_grid={
                "full_out": [3],
                "crossover": [6],
                "full_in": [10],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert result.transformed_for_low_direction
    assert result.analysis_anchor_grid == {
        "full_out": [-10.0],
        "crossover": [-6.0],
        "full_in": [-3.0],
    }
    assert len(result.summary_df) == 1


@pytest.mark.parametrize("feature", ["resources", "burden"])
def test_anchor_adapter_accepts_calibrated_direction_name(
    fuzzy_df, fuzzy_predictor, feature
):
    validated = _validated(fuzzy_df)
    calibration = calibrate_fuzzy_conditions(
        validated, propose_fuzzy_anchors(validated, fuzzy_predictor)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_anchor_sensitivity(
            validated, calibration, feature,
            anchor_grid={"full_out": [3], "crossover": [6], "full_in": [10]},
            dir_exp={calibration.condition_map[feature]: 1},
            incl_cut=0.5, outcome_threshold=0.5,
        )
    fit = result.result.results[0]
    assert fit is not None
    assert fit.settings["directional_expectations"] == {
        calibration.condition_map[feature]: 1
    }


@pytest.mark.parametrize("feature", ["resources", "burden"])
def test_anchor_adapter_recalibrates_existing_output_without_mutating_source(
    fuzzy_df, fuzzy_predictor, feature
):
    validated = _validated(fuzzy_df)
    calibration = calibrate_fuzzy_conditions(
        validated, propose_fuzzy_anchors(validated, fuzzy_predictor)
    )
    before = calibration.data.copy(deep=True)
    output = calibration.condition_map[feature]
    # A stale output must be overwritten in the run, not rejected as a collision.
    calibration.data[output] = 0.123
    stale = calibration.data.copy(deep=True)
    result = run_anchor_sensitivity(
        validated,
        calibration,
        feature,
        anchor_grid={"full_out": [3], "crossover": [6], "full_in": [10]},
        dir_exp={},
        incl_cut=0.5,
        outcome_threshold=0.5,
    )
    assert result.result.results[0] is not None
    frame = result.result.details[0]["calibrated_data"]
    assert not frame[output].eq(0.123).all()
    pd.testing.assert_series_equal(
        frame[calibration.outcome], before[calibration.outcome]
    )
    for other in calibration.conditions:
        if other != output:
            pd.testing.assert_series_equal(frame[other], before[other])
    if calibration.case_id is not None:
        pd.testing.assert_series_equal(
            frame[calibration.case_id], before[calibration.case_id]
        )
    pd.testing.assert_frame_equal(calibration.data, stale)
