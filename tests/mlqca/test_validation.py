"""mlQCA configuration, results, and validation tests."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from qca import (
    MLConditionSpec,
    MLQCAConfig,
    MLQCAResult,
    ValidatedMLQCAInput,
    validate_mlqca_input,
)
from qca.mlqca import (
    mlqca_dependencies_available,
    optional_dependency_status,
    require_mlqca_dependencies,
)


@pytest.fixture
def mlqca_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "education": [1, 2, 3, 4, 5, 6],
            "income": [10.0, 12.0, 15.0, 20.0, 22.0, 30.0],
            "parent_vote": [0, 0, 1, 1, 1, 0],
            "vote": [0, 0, 0, 1, 1, 1],
        }
    )


def test_mlqca_import_does_not_require_ml_dependencies() -> None:
    module = importlib.import_module("qca.mlqca")

    assert module.MLQCAConfig is MLQCAConfig
    assert isinstance(optional_dependency_status(), dict)
    assert isinstance(mlqca_dependencies_available(), bool)


def test_mlqca_config_normalizes_and_serializes() -> None:
    config = MLQCAConfig(
        mode="RADICAL",
        engine="CSQCA",
        top_k=8,
        model_size=3,
        required_conditions=("parent_vote",),
        model_params={"max_depth": 4},
    )

    assert config.mode == "radical"
    assert config.engine == "csqca"
    assert config.to_dict()["model_params"] == {"max_depth": 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 2, "model_size": 3}, "model_size"),
        (
            {
                "required_conditions": ("A",),
                "excluded_conditions": ("A",),
            },
            "overlap",
        ),
        ({"consistency_cutoff": 1.1}, "consistency_cutoff"),
        ({"n_jobs": 0}, "n_jobs"),
    ],
)
def test_mlqca_config_rejects_invalid_settings(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        MLQCAConfig(**kwargs)


def test_ml_condition_spec_validates_and_sorts_cutoffs() -> None:
    spec = MLConditionSpec(
        "income",
        data_type="continuous",
        direction="high",
        theoretical_cutoffs=(30, 10, 20),
    )

    assert spec.theoretical_cutoffs == (10.0, 20.0, 30.0)
    assert spec.to_dict()["name"] == "income"


def test_validate_mlqca_input_infers_candidates_and_schema(mlqca_df) -> None:
    validated = validate_mlqca_input(
        mlqca_df,
        outcome="vote",
        case_id="case",
    )

    assert isinstance(validated, ValidatedMLQCAInput)
    assert validated.candidates == ("education", "income", "parent_vote")
    assert validated.n_cases == 6
    assert validated.n_candidates == 3
    assert validated.target.tolist() == [0, 0, 0, 1, 1, 1]
    assert [spec.data_type for spec in validated.condition_specs] == [
        "ordinal",
        "ordinal",
        "binary",
    ]


def test_validate_mlqca_input_uses_enabled_condition_specs(mlqca_df) -> None:
    specs = [
        MLConditionSpec("education", data_type="ordinal", required=True),
        MLConditionSpec("income", enabled=False),
        MLConditionSpec("parent_vote", data_type="binary"),
    ]

    validated = validate_mlqca_input(
        mlqca_df,
        outcome="vote",
        case_id="case",
        condition_specs=specs,
    )

    assert validated.candidates == ("education", "parent_vote")
    assert [spec.name for spec in validated.condition_specs] == [
        "education",
        "parent_vote",
    ]


@pytest.mark.parametrize(
    ("mutator", "error", "message"),
    [
        (
            lambda df: df.assign(vote=1),
            ValueError,
            "both outcome classes",
        ),
        (
            lambda df: df.assign(income="category"),
            TypeError,
            "numeric or boolean",
        ),
        (
            lambda df: df.assign(income=1.0),
            ValueError,
            "constant",
        ),
    ],
)
def test_validate_mlqca_input_rejects_invalid_data(
    mlqca_df,
    mutator,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        validate_mlqca_input(
            mutator(mlqca_df),
            outcome="vote",
            candidates=["education", "income"],
        )


def test_mlqca_result_exposes_summary_and_markdown() -> None:
    candidates = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "coverage": [0.8, 0.7],
            "valid": [True, False],
        }
    )
    result = MLQCAResult(
        condition_ranking=pd.DataFrame(
            {"feature": ["education"], "shap_rank": [1]}
        ),
        candidate_models=candidates,
        qca_results=(None, None),
        best_model_id="m1",
        settings={"mode": "radical"},
    )

    assert result.n_models == 2
    assert result.n_valid_models == 1
    assert result.best_qca_result is None
    assert result.get_qca_result("m2") is None
    assert result.to_dataframe().equals(candidates)
    assert "# mlQCA Result" in result.to_markdown()


def test_optional_dependency_error_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(
        "qca.mlqca._optional.optional_dependency_status",
        lambda extra="mlqca": {"xgboost": False, "sklearn": True},
    )

    with pytest.raises(ImportError, match=r"pip install pyqca\[mlqca\]"):
        require_mlqca_dependencies()
