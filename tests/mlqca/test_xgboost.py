"""mlQCA XGBoost backend tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from qca import (
    MLQCAConfig,
    PredictorFitResult,
    XGBoostBackend,
    extract_split_cutoffs,
    extract_xgboost_importance,
    fit_xgboost_predictor,
    validate_mlqca_input,
)
from qca.mlqca.cutoffs import xgboost_split_nodes


class FakeBooster:
    def __init__(self) -> None:
        self.scores = {
            "weight": {"f0": 3.0, "income": 1.0},
            "gain": {"f0": 4.0, "income": 2.0},
            "cover": {"f0": 5.0, "income": 3.0},
            "total_gain": {"f0": 12.0, "income": 2.0},
            "total_cover": {"f0": 15.0, "income": 3.0},
        }

    def get_score(self, importance_type):
        return self.scores[importance_type]

    def trees_to_dataframe(self):
        return pd.DataFrame(
            {
                "Tree": [0, 0, 0, 1, 1, 1],
                "Node": [0, 1, 2, 0, 1, 2],
                "Feature": [
                    "f0",
                    "Leaf",
                    "income",
                    "f0",
                    "Leaf",
                    "f0",
                ],
                "Split": [2.5, np.nan, 15.0, 2.5, np.nan, 4.5],
                "Gain": [5.0, 0.0, 2.0, 4.0, 0.0, 3.0],
                "Cover": [8.0, 4.0, 3.0, 7.0, 3.0, 5.0],
            }
        )


@pytest.fixture
def feature_names() -> tuple[str, ...]:
    return ("education", "income", "parent_vote")


def test_xgboost_split_nodes_normalizes_feature_keys(feature_names) -> None:
    nodes = xgboost_split_nodes(FakeBooster(), feature_names)

    assert nodes["feature"].tolist() == [
        "education",
        "income",
        "education",
        "education",
    ]
    assert nodes["cutoff"].tolist() == [2.5, 15.0, 2.5, 4.5]


def test_extract_split_cutoffs_ranks_frequency_and_gain(feature_names) -> None:
    cutoffs = extract_split_cutoffs(
        FakeBooster(),
        feature_names,
        max_cutoffs_per_condition=1,
    )

    education = cutoffs.loc[cutoffs["feature"] == "education"].iloc[0]
    income = cutoffs.loc[cutoffs["feature"] == "income"].iloc[0]
    assert education["cutoff"] == pytest.approx(2.5)
    assert education["count"] == 2
    assert education["frequency"] == pytest.approx(0.5)
    assert education["feature_frequency"] == pytest.approx(2 / 3)
    assert education["total_gain"] == pytest.approx(9.0)
    assert income["cutoff_rank"] == 1


def test_extract_xgboost_importance_combines_shap_and_native_scores(
    feature_names,
) -> None:
    booster = FakeBooster()
    nodes = xgboost_split_nodes(booster, feature_names)
    contributions = np.array(
        [
            [1.0, 0.2, 0.0, -0.1],
            [-1.0, 0.4, 0.0, -0.1],
            [0.5, -0.2, 0.0, -0.1],
        ]
    )

    result = extract_xgboost_importance(
        booster,
        feature_names,
        contributions=contributions,
        feature_values=pd.DataFrame(
            {
                "education": [1.0, 2.0, 3.0],
                "income": [30.0, 20.0, 10.0],
                "parent_vote": [0.0, 1.0, 0.0],
            }
        ),
        split_nodes=nodes,
    )

    assert result["feature"].tolist() == [
        "education",
        "income",
        "parent_vote",
    ]
    education = result.iloc[0]
    assert education["shap_mean_abs"] == pytest.approx(2.5 / 3)
    assert education["shap_direction"] == "low"
    assert education["weight"] == pytest.approx(3.0)
    assert education["split_count"] == 3
    assert bool(education["used"])
    assert not bool(result.iloc[-1]["used"])


def test_extract_xgboost_importance_rejects_bad_contribution_shape(
    feature_names,
) -> None:
    with pytest.raises(ValueError, match="bias term"):
        extract_xgboost_importance(
            FakeBooster(),
            feature_names,
            contributions=np.zeros((4, 3)),
        )


def test_native_importance_works_without_shap_contributions(feature_names) -> None:
    result = extract_xgboost_importance(FakeBooster(), feature_names)

    assert result["feature"].tolist()[:2] == ["education", "income"]
    assert result["shap_rank"].isna().all()
    assert result.loc[0, "split_count"] == 3
    assert result.loc[0, "split_frequency"] == pytest.approx(0.75)


def test_predictor_fit_result_defensively_copies_tables() -> None:
    importance = pd.DataFrame({"feature": ["A"], "used": [True]})
    result = PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=importance,
        cutoff_candidates=pd.DataFrame(),
        predictions=pd.DataFrame({"observed": [0], "predicted": [0]}),
        metrics={"accuracy": 1},
    )
    importance.loc[0, "feature"] = "changed"

    assert result.condition_ranking.loc[0, "feature"] == "A"
    assert result.used_features == ("A",)
    assert result.metrics["accuracy"] == 1.0


def test_xgboost_backend_rejects_unvalidated_input() -> None:
    with pytest.raises(TypeError, match="ValidatedMLQCAInput"):
        XGBoostBackend().fit(pd.DataFrame({"A": [0, 1]}))


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None
    or importlib.util.find_spec("sklearn") is None,
    reason="mlQCA optional dependencies are not installed",
)
def test_xgboost_backend_end_to_end() -> None:
    data = pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(20)],
            "education": list(range(20)),
            "income": [value * 2.0 for value in range(20)],
            "parent_vote": [value % 2 for value in range(20)],
            "vote": [int(value >= 10) for value in range(20)],
        }
    )
    validated = validate_mlqca_input(data, "vote", case_id="case")
    config = MLQCAConfig(
        top_k=3,
        model_size=2,
        max_cutoffs_per_condition=2,
        random_state=7,
        n_jobs=1,
        model_params={
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.2,
        },
    )

    result = fit_xgboost_predictor(validated, config)

    assert result.backend == "xgboost"
    assert len(result.predictions) == len(data)
    assert set(result.feature_importance["feature"]) == {
        "education",
        "income",
        "parent_vote",
    }
    assert result.feature_importance["shap_mean_abs"].notna().all()
    assert result.feature_importance.loc[
        result.feature_importance["feature"] == "education",
        "shap_direction",
    ].iloc[0] == "high"
    assert result.metrics["accuracy"] >= 0.5
    assert result.settings["evaluation_scope"] == "training"
    assert result.settings["xgboost_version"]
    assert result.settings["scikit_learn_version"]
