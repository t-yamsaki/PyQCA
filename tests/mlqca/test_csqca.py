"""mlQCA crisp calibration and csQCA tests."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from qca import (
    MLConditionSpec,
    MLQCACSQCAResult,
    PredictorFitResult,
    QCAFitResult,
    build_necessity_table,
    calibrate_crisp_conditions,
    fit_csqca_from_calibration,
    fit_csqca_from_predictor,
    propose_crisp_calibrations,
    validate_mlqca_input,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 9)],
            "education": [1, 2, 3, 4, 5, 6, 7, 8],
            "income": [80, 70, 60, 50, 40, 30, 20, 10],
            "parent_vote": [0, 0, 0, 1, 1, 1, 1, 0],
            "vote": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def predictor_result() -> PredictorFitResult:
    importance = pd.DataFrame(
        {
            "feature": ["education", "income", "parent_vote"],
            "shap_mean_abs": [1.0, 0.8, 0.2],
            "shap_mean": [0.7, -0.5, 0.1],
            "used": [True, True, False],
        }
    )
    cutoffs = pd.DataFrame(
        {
            "feature": ["education", "education", "income"],
            "cutoff": [4.5, 6.5, 45.0],
            "count": [5, 2, 4],
            "frequency": [5 / 11, 2 / 11, 4 / 11],
            "cutoff_rank": [1, 2, 1],
        }
    )
    return PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=importance,
        cutoff_candidates=cutoffs,
        predictions=pd.DataFrame(),
    )


def test_propose_crisp_calibrations_uses_source_priority(
    raw_df,
    predictor_result,
) -> None:
    specs = [
        MLConditionSpec("education", direction="infer"),
        MLConditionSpec(
            "income",
            direction="low",
            theoretical_cutoffs=(25, 50, 75),
        ),
        MLConditionSpec("parent_vote", data_type="binary", direction="high"),
    ]
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        case_id="case",
        condition_specs=specs,
    )

    proposals = propose_crisp_calibrations(validated, predictor_result)

    education = proposals.loc[proposals["feature"] == "education"].iloc[0]
    income = proposals.loc[proposals["feature"] == "income"].iloc[0]
    parent = proposals.loc[proposals["feature"] == "parent_vote"].iloc[0]
    assert education["threshold"] == pytest.approx(4.5)
    assert education["direction"] == "high"
    assert education["threshold_source"] == "xgboost"
    assert education["direction_source"] == "shap"
    assert income["threshold"] == pytest.approx(50.0)
    assert income["direction"] == "low"
    assert income["threshold_source"] == "theoretical"
    assert parent["threshold"] == pytest.approx(0.5)
    assert parent["threshold_source"] == "binary_default"


def test_proposal_overrides_take_precedence(raw_df, predictor_result) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["education"],
    )

    proposals = propose_crisp_calibrations(
        validated,
        predictor_result,
        threshold_overrides={"education": 5.5},
        direction_overrides={"education": "low"},
    )

    assert proposals.loc[0, "threshold"] == pytest.approx(5.5)
    assert proposals.loc[0, "direction"] == "low"
    assert proposals.loc[0, "threshold_source"] == "override"
    assert proposals.loc[0, "direction_source"] == "override"


def test_direction_inference_requires_nonzero_shap(raw_df, predictor_result) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["parent_vote"],
    )
    predictor_result.feature_importance.loc[
        predictor_result.feature_importance["feature"] == "parent_vote",
        "shap_mean",
    ] = 0.0

    with pytest.raises(ValueError, match="Cannot infer"):
        propose_crisp_calibrations(validated, predictor_result)


def test_calibrate_crisp_conditions_preserves_raw_columns(
    raw_df,
    predictor_result,
) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["education", "income"],
    )
    proposals = propose_crisp_calibrations(validated, predictor_result)

    calibrated = calibrate_crisp_conditions(validated, proposals)

    assert calibrated.condition_map == {
        "education": "education__crisp",
        "income": "income__crisp",
    }
    assert calibrated.data["education"].equals(raw_df["education"])
    assert calibrated.data["education__crisp"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert calibrated.data["income__crisp"].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_calibration_rejects_missing_for_csqca(raw_df, predictor_result) -> None:
    data = raw_df.copy()
    data.loc[0, "education"] = None
    validated = validate_mlqca_input(
        data,
        "vote",
        candidates=["education"],
    )
    proposals = propose_crisp_calibrations(validated, predictor_result)

    with pytest.raises(ValueError, match="missing values"):
        calibrate_crisp_conditions(validated, proposals)


def test_fit_csqca_from_calibration_returns_necessity_and_fit(
    raw_df,
    predictor_result,
) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["education", "income"],
        case_id="case",
    )
    proposals = propose_crisp_calibrations(validated, predictor_result)
    calibrated = calibrate_crisp_conditions(validated, proposals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_csqca_from_calibration(
            calibrated,
            minimizer="standard",
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(result, MLQCACSQCAResult)
    assert isinstance(result.fit_result, QCAFitResult)
    assert result.engine.qca_type == "csQCA"
    assert result.engine.conditions == [
        "education__crisp",
        "income__crisp",
    ]
    assert len(result.necessity_table) == 4
    assert set(result.necessity_table["presence"]) == {"present", "absent"}
    assert set(result.necessity_table["source_feature"]) == {
        "education",
        "income",
    }


def test_build_necessity_table_can_exclude_negations(
    raw_df,
    predictor_result,
) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["education"],
    )
    proposals = propose_crisp_calibrations(validated, predictor_result)
    calibrated = calibrate_crisp_conditions(validated, proposals)
    result = fit_csqca_from_calibration(
        calibrated,
        consistency_cutoff=0.5,
        outcome_threshold=0.5,
    )

    necessity = build_necessity_table(
        result.engine,
        calibrated,
        include_negations=False,
    )

    assert len(necessity) == 1
    assert not necessity.loc[0, "negated"]


def test_fit_csqca_from_predictor_runs_full_p2_pipeline(
    raw_df,
    predictor_result,
) -> None:
    validated = validate_mlqca_input(
        raw_df,
        "vote",
        candidates=["education", "income"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_csqca_from_predictor(
            validated,
            predictor_result,
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )

    assert result.calibration.proposals["threshold"].tolist() == [4.5, 45.0]
    assert isinstance(result.fit_result, QCAFitResult)
    assert len(result.necessity_table) == 4
