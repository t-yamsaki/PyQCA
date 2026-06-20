"""Phase 22: mlQCA P6 reproduction and stability tests."""

from __future__ import annotations

import importlib.util
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qca import (
    MLQCAConfig,
    MLQCAStabilityResult,
    bootstrap_mlqca,
    cross_validate_mlqca,
    fit_xgboost_predictor,
    generate_condition_combinations,
    validate_mlqca_input,
)

HAS_MLQCA = (
    importlib.util.find_spec("xgboost") is not None
    and importlib.util.find_spec("sklearn") is not None
)
PAPER_DATA = Path(__file__).parents[1] / "data" / "mlqca_vote_data.csv"
PAPER_TOP_FEATURES = {
    "V648",
    "P_V195",
    "P_V530",
    "V191",
    "V207",
    "P_V44",
}


def _paper_data() -> pd.DataFrame:
    return pd.read_csv(PAPER_DATA)


def _eligible_paper_candidates(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column != "vote" and data[column].nunique(dropna=True) > 1
    ]


def _stability_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=72)
    secondary = rng.normal(size=72)
    return pd.DataFrame(
        {
            "signal": signal,
            "secondary": secondary,
            "noise_a": rng.normal(size=72),
            "noise_b": rng.normal(size=72),
            "Y": (signal + 0.25 * secondary > 0).astype(int),
        }
    )


def _fast_config(validation: str) -> MLQCAConfig:
    return MLQCAConfig(
        mode="radical",
        top_k=2,
        model_size=2,
        random_state=201,
        validation=validation,
        n_splits=3,
        n_bootstrap=4,
        n_jobs=1,
        model_params={
            "n_estimators": 12,
            "max_depth": 2,
            "learning_rate": 0.25,
        },
    )


def test_published_vote_data_integrity() -> None:
    data = _paper_data()

    assert data.shape == (427, 77)
    assert data["vote"].value_counts().to_dict() == {1.0: 319, 0.0: 108}
    assert len(data.columns.drop("vote")) == 76
    assert data["P_V181"].nunique(dropna=True) == 1
    assert len(_eligible_paper_candidates(data)) == 75


@pytest.mark.skipif(not HAS_MLQCA, reason="mlQCA dependencies are not installed")
def test_published_example_reproduces_condition_ranking_and_accuracy() -> None:
    data = _paper_data()
    validated = validate_mlqca_input(
        data,
        "vote",
        candidates=_eligible_paper_candidates(data),
    )
    config = MLQCAConfig(
        mode="radical",
        top_k=10,
        model_size=4,
        random_state=201,
        n_jobs=1,
        model_params={
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.3,
            "colsample_bytree": 0.8,
            "subsample": 0.85,
            "min_child_weight": 1,
            "gamma": 0,
        },
    )

    predictor = fit_xgboost_predictor(validated, config)
    top_ten = set(predictor.feature_importance.head(10)["feature"])
    combinations = generate_condition_combinations(
        validated,
        predictor,
        config,
    )

    assert predictor.metrics["accuracy"] == pytest.approx(1.0)
    assert predictor.metrics["roc_auc"] == pytest.approx(1.0)
    assert top_ten >= PAPER_TOP_FEATURES
    assert len(predictor.used_features) >= 60
    assert len(predictor.cutoff_candidates) >= 100
    assert len(combinations) == comb(10, 4) == 210


@pytest.mark.skipif(not HAS_MLQCA, reason="mlQCA dependencies are not installed")
def test_cross_validation_reports_holdout_stability() -> None:
    validated = validate_mlqca_input(_stability_data(), "Y")

    result = cross_validate_mlqca(validated, _fast_config("cv"))

    assert isinstance(result, MLQCAStabilityResult)
    assert result.method == "cv"
    assert result.n_runs == 3
    assert set(result.run_summary["evaluation_scope"]) == {"holdout"}
    assert result.run_summary["n_evaluation"].sum() == validated.n_cases
    signal = result.feature_stability.set_index("feature").loc["signal"]
    assert signal["top_k_rate"] >= 2 / 3
    assert result.feature_stability["top_k_rate"].between(0, 1).all()
    assert result.cutoff_stability["selection_rate"].between(0, 1).all()


@pytest.mark.skipif(not HAS_MLQCA, reason="mlQCA dependencies are not installed")
def test_bootstrap_is_reproducible_and_uses_oob_evaluation() -> None:
    validated = validate_mlqca_input(_stability_data(), "Y")
    config = _fast_config("bootstrap")

    first = bootstrap_mlqca(validated, config)
    second = bootstrap_mlqca(validated, config)

    assert first.method == "bootstrap"
    assert first.n_runs == 4
    assert set(first.run_summary["evaluation_scope"]) == {"holdout"}
    assert first.run_summary["n_train"].eq(validated.n_cases).all()
    assert first.run_summary["n_evaluation"].between(1, validated.n_cases - 1).all()
    pd.testing.assert_frame_equal(first.run_summary, second.run_summary)
    pd.testing.assert_frame_equal(first.feature_stability, second.feature_stability)
    pd.testing.assert_frame_equal(first.cutoff_stability, second.cutoff_stability)


@pytest.mark.skipif(not HAS_MLQCA, reason="mlQCA dependencies are not installed")
def test_stability_markdown_and_validation(tmp_path: Path) -> None:
    validated = validate_mlqca_input(_stability_data(), "Y")
    result = cross_validate_mlqca(validated, _fast_config("cv"))
    report_path = tmp_path / "stability.md"

    markdown = result.to_markdown(report_path)

    assert "# mlQCA Stability Report" in markdown
    assert "## Feature Stability" in markdown
    assert report_path.read_text(encoding="utf-8") == markdown
    with pytest.raises(ValueError, match="smallest class"):
        cross_validate_mlqca(validated, n_splits=40)
    with pytest.raises(ValueError, match="top_k"):
        bootstrap_mlqca(validated, _fast_config("bootstrap"), top_k=0)
