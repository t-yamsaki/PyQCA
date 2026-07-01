"""Algorithmic regression tests for GSQCA and generalized set behavior."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from qca import GSQCA, ConditionSpec
from qca.core.literals import MultiValueLiteral


def test_positive_rows_use_consistency_not_outcome_mean() -> None:
    data = pd.DataFrame(
        {
            "case": ["c1", "c2", "c3", "c4"],
            "A": [0.9, 0.8, 0.0, 0.0],
            "Y": [0.7, 0.7, 0.0, 0.0],
        }
    )
    model = GSQCA(
        data=data,
        case_id="case",
        outcome="Y",
        conditions=["A"],
        condition_types={"A": "fuzzy"},
    )

    row = next(
        r
        for r in model.build_truth_table(outcome_threshold=0.8)
        if r.config["A"] == 1
    )

    assert row.outcome_mean == pytest.approx(0.7)
    assert row.outcome_raw_consist == pytest.approx(1.4 / 1.7)
    assert row.include is True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(consistency_cutoff=0.8, outcome_threshold=0.8)

    assert result.formula is not None
    assert result.consistency == pytest.approx(1.4 / 1.7)


def test_fit_does_not_return_solution_below_consistency_cutoff() -> None:
    data = pd.DataFrame(
        {
            "case": ["c1", "c2"],
            "A": [0.9, 0.4],
            "Y": [0.9, 0.0],
        }
    )
    model = GSQCA(
        data=data,
        case_id="case",
        outcome="Y",
        conditions=["A"],
        condition_types={"A": "fuzzy"},
    )

    row = next(
        r
        for r in model.build_truth_table(outcome_threshold=0.8)
        if r.config["A"] == 1
    )

    assert row.outcome_raw_consist == pytest.approx(0.9 / 1.3)
    assert row.include is False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(consistency_cutoff=0.8, outcome_threshold=0.8)

    assert result.formula is None
    assert result.consistency is None


def test_fit_outcome_threshold_alias_controls_row_inclusion() -> None:
    data = pd.DataFrame(
        {
            "case": ["c1", "c2"],
            "A": [0.9, 0.4],
            "Y": [0.9, 0.0],
        }
    )
    model = GSQCA(
        data=data,
        case_id="case",
        outcome="Y",
        conditions=["A"],
        condition_types={"A": "fuzzy"},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = model.fit(outcome_threshold=0.8)
        permissive = model.fit(outcome_threshold=0.6)

    assert strict.formula is None
    assert permissive.formula is not None
    assert strict.settings["truth_table_inclusion_cutoff"] == pytest.approx(0.8)
    assert permissive.settings["truth_table_inclusion_cutoff"] == pytest.approx(0.6)


def test_fit_pri_cut_filters_rows_after_consistency_cutoff() -> None:
    data = pd.DataFrame(
        {
            "case": ["c1", "c2", "c3"],
            "A": [1.0, 1.0, 0.0],
            "Y": [0.75, 0.75, 0.0],
        }
    )
    model = GSQCA(
        data=data,
        case_id="case",
        outcome="Y",
        conditions=["A"],
        condition_types={"A": "crisp"},
    )
    row = next(
        r
        for r in model.build_truth_table(outcome_threshold=0.7, pri_cut=0.8)
        if r.config["A"] == 1
    )

    assert row.outcome_raw_consist == pytest.approx(0.75)
    assert row.pri_consistency == pytest.approx(2 / 3)
    assert row.include is False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = model.fit(consistency_cutoff=0.7, pri_cut=0.8)
        retained = model.fit(consistency_cutoff=0.7, pri_cut=0.6)

    assert filtered.formula is None
    assert retained.formula is not None


def test_gsqca_rejects_non_binary_crisp_condition() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b", "c"],
            "A": [1.0, 0.7, 0.0],
            "Y": [1.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="crisp condition 'A'.*0/1"):
        GSQCA(
            data=data,
            case_id="case",
            outcome="Y",
            conditions=["A"],
            condition_types={"A": "crisp"},
        )


def test_declared_multivalue_domain_contributes_logical_remainders() -> None:
    data = pd.DataFrame(
        {
            "case": ["a"],
            "M": ["x"],
            "Y": [1.0],
        }
    )
    model = GSQCA.from_condition_specs(
        data=data,
        case_id="case",
        outcome="Y",
        condition_specs=[ConditionSpec("M", "multi", domain=("x", "y"))],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(consistency_cutoff=0.8, outcome_threshold=0.8)

    assert result.qm_solution.n_remainder == 1


def test_declared_multivalue_domain_must_cover_observed_values() -> None:
    data = pd.DataFrame(
        {
            "case": ["a"],
            "M": ["z"],
            "Y": [1.0],
        }
    )

    with pytest.raises(ValueError, match="declared domain"):
        GSQCA.from_condition_specs(
            data=data,
            case_id="case",
            outcome="Y",
            condition_specs=[ConditionSpec("M", "multi", domain=("x", "y"))],
        )


def test_multivalue_condition_rejects_missing_categories() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b"],
            "M": ["x", None],
            "Y": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        GSQCA.from_condition_specs(
            data=data,
            case_id="case",
            outcome="Y",
            condition_specs=[ConditionSpec("M", "multi", domain=("x", "y"))],
        )


def test_multivalue_generalization_drops_condition_when_domain_is_exhausted() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b"],
            "M": ["x", "y"],
            "Y": [1.0, 1.0],
        }
    )
    model = GSQCA.from_condition_specs(
        data=data,
        case_id="case",
        outcome="Y",
        condition_specs=[ConditionSpec("M", "multi", domain=("x", "y"))],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            consistency_cutoff=0.8,
            outcome_threshold=0.8,
            include_remainders_in_parsimonious=False,
        )

    assert result.formula is not None
    assert "tautology" in result.formula
    assert "M=" not in result.formula


def test_multivalue_generalization_keeps_partial_value_sets() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b", "c"],
            "M": ["x", "y", "z"],
            "Y": [1.0, 1.0, 0.0],
        }
    )
    model = GSQCA.from_condition_specs(
        data=data,
        case_id="case",
        outcome="Y",
        condition_specs=[ConditionSpec("M", "multi", domain=("x", "y", "z"))],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            consistency_cutoff=0.8,
            outcome_threshold=0.8,
            include_remainders_in_parsimonious=False,
        )

    assert result.formula is not None
    assert "M={x,y}" in result.formula
    assert "tautology" not in result.formula
    assert result.consistency == pytest.approx(1.0)


def test_gsqca_accepts_multivalent_fuzzy_membership_columns() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b", "c", "d"],
            "sector_public": [0.9, 0.8, 0.1, 0.0],
            "sector_private": [0.1, 0.2, 0.9, 0.8],
            "capacity": [0.9, 0.8, 0.2, 0.1],
            "Y": [1.0, 0.9, 0.0, 0.0],
        }
    )
    model = GSQCA.from_condition_specs(
        data=data,
        case_id="case",
        outcome="Y",
        condition_specs=[
            ConditionSpec("capacity", "fuzzy", calibrated=True),
            ConditionSpec(
                "sector",
                "multi",
                calibrated=True,
                value_columns={
                    "public": "sector_public",
                    "private": "sector_private",
                },
            ),
        ],
    )

    public = MultiValueLiteral("sector", "public").membership(model.data)
    public_or_private = MultiValueLiteral(
        "sector", frozenset({"public", "private"})
    ).membership(model.data)

    assert "sector=public" in model.data.columns
    assert "sector=private" in model.data.columns
    assert public.tolist() == pytest.approx([0.9, 0.8, 0.1, 0.0])
    assert public_or_private.tolist() == pytest.approx([0.9, 0.8, 0.9, 0.8])
    assert model.condition_schema.set_index("name").loc["sector", "calibrated"]
    assert tuple(model.condition_spec_map["sector"].domain) == ("public", "private")

    rows = model.build_truth_table(outcome_threshold=0.8)
    assert {row.config["sector"] for row in rows} == {"public", "private"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            consistency_cutoff=0.8,
            outcome_threshold=0.8,
            include_remainders_in_parsimonious=False,
        )

    assert result.qca_type == "GSQCA"
    assert result.settings["workflow"] == "GSQCA-fuzzy-multi"
    assert result.formula is not None


def test_gsqca_accepts_value_columns_from_dataframe_schema() -> None:
    data = pd.DataFrame(
        {
            "case": ["a"],
            "sector_public": [1.0],
            "sector_private": [0.0],
            "Y": [1.0],
        }
    )
    schema = pd.DataFrame(
        [
            {
                "name": "sector",
                "type": "multi-value",
                "calibrated": True,
                "value_columns": {
                    "public": "sector_public",
                    "private": "sector_private",
                },
            }
        ]
    )

    model = GSQCA.from_schema(
        data=data,
        case_id="case",
        outcome="Y",
        schema=schema,
    )

    assert tuple(model.condition_spec_map["sector"].domain) == ("public", "private")
    assert model.condition_schema.set_index("name").loc["sector", "value_columns"] == {
        "public": "sector_public",
        "private": "sector_private",
    }


def test_gsqca_reports_official_engine_name() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b"],
            "A": [1.0, 0.0],
            "Y": [1.0, 0.0],
        }
    )

    model = GSQCA(
        data=data,
        case_id="case",
        outcome="Y",
        conditions=["A"],
        condition_types={"A": "crisp"},
    )

    assert isinstance(model, GSQCA)
    assert model.qca_type == "GSQCA"
    assert model.workflow == "GSQCA-crisp-only"
