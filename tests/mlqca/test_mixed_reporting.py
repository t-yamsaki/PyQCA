"""Phase 21: mlQCA P5 MixedQCA, reporting, viz, and reproducibility tests."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import (
    MixedCalibrationSet,
    MLQCAConfig,
    MLQCAMixedResult,
    MLQCAResult,
    PredictorFitResult,
    calibrate_mixed_conditions,
    collect_mlqca_reproducibility,
    collect_reproducibility_metadata,
    dataframe_fingerprint,
    fit_mixedqca_from_calibration,
    fit_mixedqca_from_predictor,
    generate_markdown_report,
    generate_mlqca_markdown_report,
    validate_mlqca_input,
)


@pytest.fixture
def mixed_ml_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 13)],
            "policy": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "capacity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "region": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def mixed_predictor() -> PredictorFitResult:
    return PredictorFitResult(
        backend="xgboost",
        model=object(),
        feature_importance=pd.DataFrame(
            {
                "feature": ["policy", "capacity", "region"],
                "shap_mean_abs": [1.0, 0.8, 0.2],
                "shap_mean": [0.5, 0.4, 0.1],
                "shap_direction": ["high", "high", "high"],
                "shap_rank": [1, 2, 3],
                "used": [True, True, True],
            }
        ),
        cutoff_candidates=pd.DataFrame(
            {
                "feature": ["policy", "capacity", "capacity", "capacity"],
                "cutoff": [6.5, 3.0, 6.0, 10.0],
                "count": [6, 5, 4, 3],
                "frequency": [0.33, 0.28, 0.22, 0.17],
                "cutoff_rank": [1, 1, 2, 3],
            }
        ),
        predictions=pd.DataFrame(),
        metrics={"accuracy": 1.0},
        settings={"xgboost_version": "test"},
    )


def _validated(mixed_ml_df):
    return validate_mlqca_input(
        mixed_ml_df,
        "Y",
        candidates=["policy", "capacity", "region"],
        case_id="case",
    )


def test_mixed_calibration_builds_unified_schema(
    mixed_ml_df,
    mixed_predictor,
) -> None:
    calibration = calibrate_mixed_conditions(
        _validated(mixed_ml_df),
        mixed_predictor,
        {
            "policy": "crisp",
            "capacity": "fuzzy",
            "region": "multi",
        },
    )

    assert isinstance(calibration, MixedCalibrationSet)
    assert calibration.conditions == (
        "policy__crisp",
        "capacity__fuzzy",
        "region",
    )
    assert calibration.condition_types == {
        "policy__crisp": "crisp",
        "capacity__fuzzy": "fuzzy",
        "region": "multi",
    }
    assert calibration.workflow == "mixed-crisp-fuzzy-multi"
    assert set(calibration.proposals["kind"]) == {"crisp", "fuzzy", "multi"}


def test_mixedqca_connections(mixed_ml_df, mixed_predictor) -> None:
    validated = _validated(mixed_ml_df)
    calibration = calibrate_mixed_conditions(
        validated,
        mixed_predictor,
        {
            "policy": "crisp",
            "capacity": "fuzzy",
            "region": "multi",
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = fit_mixedqca_from_calibration(
            calibration,
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )
        combined = fit_mixedqca_from_predictor(
            validated,
            mixed_predictor,
            {
                "policy": "crisp",
                "capacity": "fuzzy",
                "region": "multi",
            },
            consistency_cutoff=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(direct, MLQCAMixedResult)
    assert direct.engine.qca_type == "MixedQCA"
    assert direct.engine.workflow == "mixed-crisp-fuzzy-multi"
    assert combined.fit_result.qca_type == "MixedQCA"


def test_mixed_calibration_validates_complete_kind_map(
    mixed_ml_df,
    mixed_predictor,
) -> None:
    with pytest.raises(ValueError, match="missing entries"):
        calibrate_mixed_conditions(
            _validated(mixed_ml_df),
            mixed_predictor,
            {"policy": "crisp"},
        )


def test_dataframe_fingerprint_is_stable_and_sensitive(mixed_ml_df) -> None:
    first = dataframe_fingerprint(mixed_ml_df)
    second = dataframe_fingerprint(mixed_ml_df.copy())
    changed = mixed_ml_df.copy()
    changed.loc[0, "policy"] = 99

    assert first == second
    assert first != dataframe_fingerprint(changed)
    assert len(first) == 64


def test_mlqca_reproducibility_metadata(
    mixed_ml_df,
    mixed_predictor,
) -> None:
    validated = _validated(mixed_ml_df)
    config = MLQCAConfig(
        mode="radical",
        top_k=3,
        model_size=2,
        random_state=42,
    )
    metadata = collect_mlqca_reproducibility(
        validated,
        mixed_predictor,
        config,
        extra={"study": "acceptance"},
    )

    assert metadata["data"]["fingerprint_sha256"]
    assert metadata["data"]["shape"] == [12, 5]
    assert metadata["predictor"]["backend"] == "xgboost"
    assert metadata["config"]["random_state"] == 42
    assert metadata["extra"]["study"] == "acceptance"


@pytest.fixture
def mlqca_result() -> MLQCAResult:
    importance = pd.DataFrame(
        {
            "feature": ["A", "B"],
            "shap_mean_abs": [0.8, 0.4],
        }
    )
    cutoffs = pd.DataFrame(
        {
            "feature": ["A", "A"],
            "cutoff": [2.0, 4.0],
            "count": [5, 2],
        }
    )
    models = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "valid": [True, True],
            "coverage": [0.8, 0.7],
            "consistency": [0.9, 0.95],
            "complexity": [3, 2],
        }
    )
    return MLQCAResult(
        feature_importance=importance,
        condition_ranking=importance.assign(selection_rank=[1, 2]),
        cutoff_candidates=cutoffs,
        calibration_proposals=pd.DataFrame(
            {"feature": ["A"], "kind": ["crisp"]}
        ),
        candidate_models=models,
        qca_results=(None, None),
        best_model_id="m1",
        pareto_models=models.copy(),
        settings={"mode": "radical"},
        metadata={"data": {"fingerprint_sha256": "abc123"}},
    )


def test_mlqca_markdown_and_generic_reporting(mlqca_result) -> None:
    markdown = generate_mlqca_markdown_report(mlqca_result)
    method_markdown = mlqca_result.to_markdown_report()
    generic = generate_markdown_report(mlqca_result)
    metadata = collect_reproducibility_metadata(mlqca_result)

    assert "# mlQCA Analysis Report" in markdown
    assert "## Pareto Frontier" in markdown
    assert "`fingerprint_sha256`" not in markdown
    assert method_markdown == markdown
    assert "## Candidate Models" in generic
    assert metadata["n_models"] == 2
    assert metadata["best_model_id"] == "m1"
    assert metadata["analysis_metadata"]["data"]["fingerprint_sha256"] == "abc123"


def test_mlqca_markdown_writes_file(mlqca_result) -> None:
    path = Path.cwd() / ".pyqca_mlqca_p5_report.md"
    try:
        markdown = mlqca_result.to_markdown_report(path)
        assert path.read_text(encoding="utf-8") == markdown
    finally:
        if path.exists():
            path.unlink()


def test_mlqca_jupyter_summary(mlqca_result) -> None:
    summary = mlqca_result.to_jupyter_summary()

    assert "n_models: 2" in str(summary)
    assert "best_model_id" in summary._repr_html_()
    assert "<table" in mlqca_result._repr_html_()


def test_mlqca_visualizations_are_optional(mlqca_result) -> None:
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    if not has_matplotlib:
        with pytest.raises(ImportError, match=r"pyqca\[viz\]"):
            mlqca_result.plot_importance()
        return

    import matplotlib

    matplotlib.use("Agg")
    importance = mlqca_result.plot_importance()
    cutoffs = mlqca_result.plot_cutoffs("A")
    frontier = mlqca_result.plot_frontier()

    assert importance.axes
    assert cutoffs.axes
    assert frontier.axes
