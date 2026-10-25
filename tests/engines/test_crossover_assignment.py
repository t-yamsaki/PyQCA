"""Crossover cases affect fuzzy scores, but not empirical row frequencies."""

import warnings

import numpy as np
import pandas as pd
import pytest

from qca import FSQCA, GSQCA


@pytest.fixture(params=[FSQCA, GSQCA])
def engine(request):
    return request.param


def make_model(engine, data, conditions):
    kwargs = (
        {"condition_types": dict.fromkeys(conditions, "fuzzy")}
        if engine is GSQCA
        else {}
    )
    return engine(
        data=data, outcome="Y", conditions=conditions, case_id="case", **kwargs
    )


def test_crossover_only_case_creates_no_observed_row(engine):
    model = make_model(
        engine, pd.DataFrame({"case": ["ambiguous"], "A": [0.5], "Y": [1.0]}), ["A"]
    )
    with pytest.warns(UserWarning, match="excluded from truth-table row assignment"):
        assert model.build_truth_table() == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = model.fit()
    assert result.minimization.n_positive == 0
    assert result.minimization.n_remainder == 2
    assert result.formula is None
    assert model.data["A"].tolist() == [0.5]


def test_immediate_neighbors_are_assigned_but_exact_half_is_not(engine):
    data = pd.DataFrame(
        {
            "case": ["below", "equal", "above"],
            "A": [np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0)],
            "Y": [1.0, 1.0, 1.0],
        }
    )
    model = make_model(engine, data, ["A"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rows = {r.config["A"]: r for r in model.build_truth_table()}
    assert rows[0].case_ids == ["below"]
    assert rows[1].case_ids == ["above"]
    assert rows[0].n_cases == rows[1].n_cases == 1
    pd.testing.assert_frame_equal(model.data, data)


def test_ambiguous_cases_still_contribute_to_consistency_and_pri(engine):
    data = pd.DataFrame(
        {"case": ["observed", "ambiguous"], "A": [0.8, 0.5], "Y": [1.0, 0.0]}
    )
    model = make_model(engine, data, ["A"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (row,) = model.build_truth_table(outcome_threshold=0.7)
        permissive = model.fit(consistency_cutoff=0.6, frequency_cutoff=1)
        strict = model.fit(consistency_cutoff=0.6, frequency_cutoff=2)
    assert row.n_cases == 1 and row.case_ids == ["observed"]
    assert row.outcome_mean == 1.0
    assert row.outcome_raw_consist == pytest.approx(8 / 13)
    assert row.pri_consistency == pytest.approx(8 / 13)
    assert not row.include
    assert permissive.minimization.n_positive == 1
    assert permissive.minimization.n_remainder == 1
    assert strict.minimization.n_positive == 0
    assert strict.minimization.n_remainder == 2


def test_any_ambiguous_condition_excludes_whole_case_not_half_outcome(engine):
    data = pd.DataFrame(
        {
            "case": ["a_half", "b_half", "both_half", "outcome_half"],
            "A": [0.5, 0.8, 0.5, 0.8],
            "B": [0.8, 0.5, 0.5, 0.8],
            "Y": [1.0, 1.0, 1.0, 0.5],
        }
    )
    model = make_model(engine, data, ["A", "B"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (row,) = model.build_truth_table()
    assert row.config == {"A": 1, "B": 1}
    assert row.case_ids == ["outcome_half"]
    assert row.n_cases == 1


def test_mixed_gsqca_preserves_multivalue_alignment():
    model = GSQCA(
        data=pd.DataFrame(
            {
                "case": ["ambiguous", "kept"],
                "A": [0.5, 0.9],
                "m0": [0.9, 0.1],
                "m1": [0.1, 0.9],
                "Y": [1.0, 1.0],
            }
        ),
        case_id="case",
        outcome="Y",
        schema=[
            {"name": "A", "type": "fuzzy"},
            {"name": "M", "type": "multi-value", "value_columns": {0: "m0", 1: "m1"}},
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (row,) = model.build_truth_table()
    assert row.config == {"A": 1, "M": 1}
    assert row.case_ids == ["kept"]
