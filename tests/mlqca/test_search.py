"""mlQCA combination search and Pareto tests."""

from __future__ import annotations

import importlib.util
import warnings

import pandas as pd
import pytest

from qca import (
    MLConditionSpec,
    MLQCAConfig,
    MLQCAResult,
    PredictorFitResult,
    fit_xgboost_predictor,
    generate_condition_combinations,
    pareto_frontier,
    search_csqca_models,
    validate_mlqca_input,
)


@pytest.fixture
def search_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 17)],
            "A": list(range(1, 17)),
            "B": list(range(16, 0, -1)),
            "C": [0, 1] * 8,
            "D": [0, 0, 1, 1] * 4,
            "Y": [0] * 8 + [1] * 8,
        }
    )


@pytest.fixture
def search_predictor() -> PredictorFitResult:
    importance = pd.DataFrame(
        {
            "feature": ["A", "B", "C", "D"],
            "shap_mean_abs": [1.0, 0.8, 0.4, 0.2],
            "shap_mean": [0.7, -0.6, 0.2, -0.1],
            "shap_direction": ["high", "low", "high", "low"],
            "shap_rank": [1, 2, 3, 4],
            "used": [True, True, True, True],
        }
    )
    cutoffs = pd.DataFrame(
        {
            "feature": ["A", "B"],
            "cutoff": [8.5, 8.5],
            "count": [8, 8],
            "frequency": [0.5, 0.5],
            "cutoff_rank": [1, 1],
        }
    )
    return PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=importance,
        cutoff_candidates=cutoffs,
        predictions=pd.DataFrame(),
    )


def test_radical_generates_all_top_ranked_combinations(
    search_df,
    search_predictor,
) -> None:
    validated = validate_mlqca_input(search_df, "Y", case_id="case")
    config = MLQCAConfig(
        mode="radical",
        top_k=4,
        model_size=2,
    )

    models = generate_condition_combinations(
        validated,
        search_predictor,
        config,
    )

    assert len(models) == 6
    assert ("A", "B") in models
    assert ("C", "D") in models


def test_conservative_requires_theoretical_constraint(
    search_df,
    search_predictor,
) -> None:
    validated = validate_mlqca_input(search_df, "Y", case_id="case")
    config = MLQCAConfig(
        mode="conservative",
        top_k=4,
        model_size=2,
    )

    with pytest.raises(ValueError, match="Conservative search requires"):
        generate_condition_combinations(
            validated,
            search_predictor,
            config,
        )


def test_conservative_required_condition_is_in_every_model(
    search_df,
    search_predictor,
) -> None:
    specs = [
        MLConditionSpec("A", direction="high", required=True),
        MLConditionSpec("B", direction="low"),
        MLConditionSpec("C", data_type="binary", direction="high"),
        MLConditionSpec("D", data_type="binary", direction="low"),
    ]
    validated = validate_mlqca_input(
        search_df,
        "Y",
        case_id="case",
        condition_specs=specs,
    )
    config = MLQCAConfig(
        mode="conservative",
        top_k=4,
        model_size=2,
    )

    models = generate_condition_combinations(
        validated,
        search_predictor,
        config,
    )

    assert len(models) == 3
    assert all("A" in model for model in models)


def test_conservative_accepts_explicit_theory_models(
    search_df,
    search_predictor,
) -> None:
    validated = validate_mlqca_input(search_df, "Y", case_id="case")
    config = MLQCAConfig(
        mode="conservative",
        top_k=4,
        model_size=2,
        excluded_conditions=("D",),
    )

    models = generate_condition_combinations(
        validated,
        search_predictor,
        config,
        conservative_models=[["A", "C"], ["B", "C"]],
    )

    assert models == (("A", "C"), ("B", "C"))


def test_model_generation_respects_max_models(
    search_df,
    search_predictor,
) -> None:
    validated = validate_mlqca_input(search_df, "Y", case_id="case")
    config = MLQCAConfig(
        mode="radical",
        top_k=4,
        model_size=2,
        max_models=5,
    )

    with pytest.raises(ValueError, match="exceeding max_models"):
        generate_condition_combinations(
            validated,
            search_predictor,
            config,
        )


def test_pareto_frontier_removes_dominated_models() -> None:
    models = pd.DataFrame(
        {
            "model_id": ["m1", "m2", "m3", "m4"],
            "valid": [True, True, True, False],
            "consistency": [0.90, 0.90, 0.85, 1.0],
            "coverage": [0.70, 0.60, 0.80, 1.0],
            "complexity": [3, 4, 2, 1],
        }
    )

    frontier = pareto_frontier(models)

    assert set(frontier["model_id"]) == {"m1", "m3"}
    assert frontier["pareto_rank"].tolist() == [1, 2]


def test_radical_search_returns_ranked_result_and_best_model(
    search_df,
    search_predictor,
) -> None:
    specs = [
        MLConditionSpec("A", direction="high"),
        MLConditionSpec("B", direction="low"),
        MLConditionSpec("C", data_type="binary", direction="high"),
        MLConditionSpec("D", data_type="binary", direction="low"),
    ]
    validated = validate_mlqca_input(
        search_df,
        "Y",
        case_id="case",
        condition_specs=specs,
    )
    config = MLQCAConfig(
        mode="radical",
        top_k=3,
        model_size=2,
        consistency_cutoff=0.5,
        minimizer="standard",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = search_csqca_models(
            validated,
            search_predictor,
            config,
            outcome_threshold=0.5,
        )

    assert isinstance(result, MLQCAResult)
    assert result.n_models == 3
    assert result.n_valid_models >= 1
    assert result.best_model_id is not None
    assert result.best_qca_result is not None
    assert not result.pareto_models.empty
    assert set(result.calibration_proposals["model_id"]) == {
        "m0001",
        "m0002",
        "m0003",
    }
    assert result.settings["pareto_objectives"]["minimize"] == ["complexity"]
    assert result.metadata["data"]["fingerprint_sha256"]
    assert result.metadata["predictor"]["backend"] == "xgboost"
    assert result.metadata["config"]["mode"] == "radical"


def test_conservative_search_evaluates_explicit_theory_models(
    search_df,
    search_predictor,
) -> None:
    specs = [
        MLConditionSpec("A", direction="high"),
        MLConditionSpec("B", direction="low"),
        MLConditionSpec("C", data_type="binary", direction="high"),
        MLConditionSpec("D", data_type="binary", direction="low"),
    ]
    validated = validate_mlqca_input(
        search_df,
        "Y",
        case_id="case",
        condition_specs=specs,
    )
    config = MLQCAConfig(
        mode="conservative",
        top_k=4,
        model_size=2,
        consistency_cutoff=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = search_csqca_models(
            validated,
            search_predictor,
            config,
            conservative_models=[["A", "C"], ["B", "D"]],
            outcome_threshold=0.5,
        )

    assert result.candidate_models["mode"].tolist() == [
        "conservative",
        "conservative",
    ]
    assert result.candidate_models["conditions"].tolist() == [
        ("A", "C"),
        ("B", "D"),
    ]
    assert result.n_models == 2


def test_search_records_model_level_errors(
    search_df,
    search_predictor,
) -> None:
    search_predictor.feature_importance.loc[
        search_predictor.feature_importance["feature"] == "C",
        "shap_direction",
    ] = None
    validated = validate_mlqca_input(search_df, "Y", case_id="case")
    config = MLQCAConfig(
        mode="radical",
        top_k=3,
        model_size=2,
        consistency_cutoff=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = search_csqca_models(
            validated,
            search_predictor,
            config,
            direction_overrides={"A": "high", "B": "low"},
            outcome_threshold=0.5,
        )

    failed = result.candidate_models.loc[
        ~result.candidate_models["valid"].astype(bool)
    ]
    assert not failed.empty
    assert failed["error"].notna().all()


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None
    or importlib.util.find_spec("sklearn") is None,
    reason="mlQCA optional dependencies are not installed",
)
def test_real_xgboost_to_pareto_end_to_end(search_df) -> None:
    specs = [
        MLConditionSpec("A", direction="high"),
        MLConditionSpec("B", direction="low"),
        MLConditionSpec("C", data_type="binary", direction="high"),
    ]
    validated = validate_mlqca_input(
        search_df,
        "Y",
        case_id="case",
        condition_specs=specs,
    )
    config = MLQCAConfig(
        mode="radical",
        top_k=3,
        model_size=2,
        consistency_cutoff=0.5,
        n_jobs=1,
        model_params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.2,
        },
    )
    predictor = fit_xgboost_predictor(validated, config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = search_csqca_models(
            validated,
            predictor,
            config,
            outcome_threshold=0.5,
        )

    assert result.n_models == 3
    assert result.n_valid_models >= 1
    assert result.best_qca_result is not None
    assert not result.pareto_models.empty
