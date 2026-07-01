"""Core QCA engine acceptance tests."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from qca import (
    CSQCA,
    FSQCA,
    GSQCA,
    MVQCA,
    calibrate_crisp,
    calibrate_piecewise,
)
from qca.core.literals import SetLiteral
from qca.results import QCAFitResult


@pytest.fixture
def core_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 9)],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "quality": [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0],
            "channel": ["a", "a", "b", "b", "a", "c", "c", "c"],
            "outcome": [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
        }
    )


def prepared_core_df(core_df: pd.DataFrame) -> pd.DataFrame:
    crisp = calibrate_crisp(
        core_df,
        "score",
        threshold=5.5,
        out_col="score_high",
    )
    fuzzy = calibrate_piecewise(
        crisp.df,
        "quality",
        out_col="quality_fuzzy",
        min_nonnull=2,
    )
    return fuzzy.df


def test_v01_data_validation(core_df):
    with pytest.raises(ValueError, match="Required columns|DataFrame"):
        GSQCA(
            data=core_df,
            case_id="case",
            conditions=["missing"],
            condition_types={"missing": "crisp"},
            outcome="outcome",
        )

    invalid = core_df.copy()
    invalid["outcome"] = invalid["outcome"] + 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        GSQCA(
            data=invalid,
            case_id="case",
            conditions=["quality"],
            condition_types={"quality": "fuzzy"},
            outcome="outcome",
        )


def test_v01_calibration_outputs_are_qca_ready(core_df):
    df = prepared_core_df(core_df)

    assert set(df["score_high"].unique()).issubset({0.0, 1.0})
    assert df["quality_fuzzy"].dropna().between(0.0, 1.0).all()


def test_v01_generalized_core_analysis(core_df):
    df = prepared_core_df(core_df)
    model = GSQCA(
        data=df,
        case_id="case",
        conditions=["score_high", "quality_fuzzy", "channel"],
        condition_types={
            "score_high": "crisp",
            "quality_fuzzy": "fuzzy",
            "channel": "multi",
        },
        outcome="outcome",
    )

    truth_table = model.build_truth_table(outcome_threshold=0.75)
    assert sum(row.n_cases for row in truth_table) == model.n_cases
    assert all(set(row.config) == set(model.all_condition_names) for row in truth_table)

    sufficiency = model.evaluate_sufficiency([SetLiteral("score_high")])
    necessity = model.evaluate_necessity(SetLiteral("score_high"))
    assert 0.0 <= sufficiency.consistency <= 1.0
    assert 0.0 <= sufficiency.raw_coverage <= 1.0
    assert 0.0 <= necessity.consistency <= 1.0
    assert 0.0 <= necessity.coverage <= 1.0

    sufficiency_df = model.search_sufficient_configurations(
        max_depth=2,
        min_consistency=0.5,
        min_coverage=0.0,
        min_cases=1,
    )
    necessity_df = model.search_necessary_conditions(
        min_consistency=0.5,
        min_coverage=0.0,
    )
    assert {"antecedent", "consistency", "raw_coverage"}.issubset(
        sufficiency_df.columns
    )
    assert {"condition", "consistency", "coverage"}.issubset(necessity_df.columns)


def test_v01_basic_fsqca_fit(core_df):
    df = prepared_core_df(core_df)
    model = FSQCA(
        data=df,
        case_id="case",
        outcome="outcome",
        conditions=["score_high", "quality_fuzzy"],
        condition_types={"score_high": "crisp", "quality_fuzzy": "fuzzy"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )
    assert isinstance(result, QCAFitResult)
    assert isinstance(result.truth_table, pd.DataFrame)
    assert isinstance(result.solutions, pd.DataFrame)


def test_v01_basic_csqca_fit(core_df):
    df = calibrate_crisp(
        prepared_core_df(core_df),
        "outcome",
        threshold=0.75,
        out_col="outcome_high",
    ).df
    model = CSQCA(
        data=df,
        case_id="case",
        outcome="outcome_high",
        conditions=["score_high"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )
    assert isinstance(result, QCAFitResult)
    assert model.condition_types == {"score_high": "crisp"}
    assert isinstance(result.truth_table, pd.DataFrame)


def test_v01_basic_mvqca_fit(core_df):
    model = MVQCA(
        data=core_df,
        case_id="case",
        outcome="outcome",
        conditions=["channel"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )
    assert isinstance(result, QCAFitResult)
    assert model.multivalue_conditions == ["channel"]
    assert isinstance(result.truth_table, pd.DataFrame)


def test_v01_fit_coverage_cutoff_filters_selected_solution(core_df):
    df = prepared_core_df(core_df)
    model = FSQCA(
        data=df,
        case_id="case",
        outcome="outcome",
        conditions=["score_high", "quality_fuzzy"],
        condition_types={"score_high": "crisp", "quality_fuzzy": "fuzzy"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )
        max_coverage = float(base.solutions["coverage"].dropna().max())
        strict = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=min(1.0, max_coverage + 0.01),
            outcome_threshold=0.75,
        )

    assert base.coverage is not None
    if max_coverage < 1.0:
        assert strict.selected_solution is None
        assert strict.selected_solution_type == "none"
        assert strict.solutions.empty
