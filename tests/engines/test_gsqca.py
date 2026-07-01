"""GSQCA high-level workflow acceptance tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import GSQCA, ConditionSpec, condition_specs_from_schema
from qca.results import QCAFitResult


@pytest.fixture
def gsqca_v02_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D", "E", "F"],
            "policy": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "capacity": [0.9, 0.8, 0.2, 0.9, 0.1, 0.2],
            "region": ["north", "north", "south", "north", "south", "south"],
            "outcome": [1.0, 0.9, 0.1, 0.2, 0.1, 0.0],
        }
    )


def gsqca_specs() -> list[ConditionSpec]:
    return [
        ConditionSpec("policy", "crisp", domain=(0.0, 1.0), calibrated=True),
        ConditionSpec("capacity", "fuzzy", domain=(0.0, 1.0), calibrated=True),
        ConditionSpec("region", "multi", domain=("north", "south"), calibrated=False),
    ]


def gsqca_schema_records() -> list[dict[str, object]]:
    return [
        {
            "name": "policy",
            "type": "crisp",
            "domain": [0.0, 1.0],
            "calibrated": True,
        },
        {
            "name": "capacity",
            "type": "fuzzy",
            "domain": [0.0, 1.0],
            "calibrated": True,
        },
        {
            "name": "region",
            "type": "multi-value",
            "domain": ["north", "south"],
            "calibrated": False,
        },
    ]


def test_v02_accepts_unified_condition_specs(gsqca_v02_df) -> None:
    model = GSQCA(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        condition_specs=gsqca_specs(),
    )

    assert model.conditions == ["policy", "capacity", "region"]
    assert model.set_conditions == ["policy", "capacity"]
    assert model.multivalue_conditions == ["region"]
    assert model.condition_types == {
        "policy": "crisp",
        "capacity": "fuzzy",
        "region": "multi",
    }

    schema = model.condition_schema.set_index("name")
    assert schema.loc["policy", "kind"] == "crisp"
    assert schema.loc["capacity", "role"] == "set"
    assert schema.loc["region", "role"] == "multi"
    assert tuple(schema.loc["region", "domain"]) == ("north", "south")
    assert bool(schema.loc["policy", "calibrated"]) is True
    assert bool(schema.loc["region", "calibrated"]) is False


def test_v02_accepts_mapping_and_dataframe_condition_schema(gsqca_v02_df) -> None:
    records = gsqca_schema_records()

    from_records = condition_specs_from_schema(records)
    from_dataframe = condition_specs_from_schema(pd.DataFrame(records))
    model = GSQCA.from_schema(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        schema=pd.DataFrame(records),
    )

    assert [spec.kind for spec in from_records] == ["crisp", "fuzzy", "multi"]
    assert [spec.kind for spec in from_dataframe] == ["crisp", "fuzzy", "multi"]
    assert model.workflow == "GSQCA-crisp-fuzzy-multi"
    assert model.workflow_kinds == ("crisp", "fuzzy", "multi")
    assert model.is_generalized_workflow
    assert model.has_crisp_conditions
    assert model.has_fuzzy_conditions
    assert model.has_multivalue_conditions
    assert model.schema.equals(model.condition_schema)
    assert tuple(model.condition_spec_map["region"].domain) == ("north", "south")


def test_v02_from_condition_specs_classmethod(gsqca_v02_df) -> None:
    model = GSQCA.from_condition_specs(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        condition_specs=gsqca_specs(),
    )

    assert isinstance(model, GSQCA)
    assert model.condition_spec_map["region"].is_multi


def test_v02_rejects_schema_alias_combined_with_condition_specs(
    gsqca_v02_df,
) -> None:
    with pytest.raises(ValueError, match="schema and condition_specs"):
        GSQCA(
            data=gsqca_v02_df,
            case_id="case",
            outcome="outcome",
            schema=gsqca_schema_records(),
            condition_specs=gsqca_specs(),
        )


def test_v02_rejects_combined_schema_styles(gsqca_v02_df) -> None:
    with pytest.raises(ValueError, match="condition_specs cannot be combined"):
        GSQCA(
            data=gsqca_v02_df,
            case_id="case",
            outcome="outcome",
            condition_specs=gsqca_specs(),
            conditions=["policy"],
            condition_types={"policy": "crisp"},
        )


def test_v02_high_level_api_runs_gsqca_workflow(gsqca_v02_df) -> None:
    model = GSQCA(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        conditions=["policy", "capacity", "region"],
        condition_types={
            "policy": "crisp",
            "capacity": "fuzzy",
            "region": "multi",
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.analyze(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )

    assert isinstance(result, QCAFitResult)
    assert result.qca_type == "GSQCA"
    assert result.outcome == "outcome"
    assert result.case_id == "case"
    assert isinstance(result.truth_table, pd.DataFrame)
    assert isinstance(result.solutions, pd.DataFrame)
    assert isinstance(result.case_coverage, pd.DataFrame)
    assert result.qm_solution.condition_names == ["policy", "capacity", "region"]
    assert result.workflow == "GSQCA-crisp-fuzzy-multi"
    assert result.settings["workflow"] == "GSQCA-crisp-fuzzy-multi"

    schema = result.condition_schema.set_index("name")
    assert schema.loc["policy", "kind"] == "crisp"
    assert schema.loc["capacity", "kind"] == "fuzzy"
    assert schema.loc["region", "kind"] == "multi"


def test_v02_formula_export(gsqca_v02_df) -> None:
    model = GSQCA.from_condition_specs(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        condition_specs=gsqca_specs(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )

    assert result.formula == result.to_formula()
    assert result.selected_formula == result.formula
    assert result.formula is not None
    assert result.selected_solution_type in result.formulas
    assert result.formulas[result.selected_solution_type] == result.formula
    assert "policy" in result.to_formula("complex")

    with pytest.raises(ValueError, match="Unknown solution_type"):
        result.to_formula("minimal")


def test_v02_formula_export_writes_selected_formula(gsqca_v02_df) -> None:
    model = GSQCA.from_schema(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        schema=gsqca_schema_records(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )

    path = Path.cwd() / ".pyqca_v02_formula_test.txt"
    try:
        formula = result.export_formula(path, "complex")

        assert formula is not None
        assert path.read_text(encoding="utf-8") == formula
    finally:
        if path.exists():
            path.unlink()


def test_v02_markdown_export_includes_schema_and_formulas(
    gsqca_v02_df,
) -> None:
    model = GSQCA.from_condition_specs(
        data=gsqca_v02_df,
        case_id="case",
        outcome="outcome",
        condition_specs=gsqca_specs(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            minimizer="qmc",
            consistency_cutoff=0.5,
            coverage_cutoff=0.0,
            outcome_threshold=0.75,
        )

    path = Path.cwd() / ".pyqca_v02_result_test.md"
    try:
        markdown = result.to_markdown(path)

        assert path.read_text(encoding="utf-8") == markdown
        assert "# QCA Fit Result" in markdown
        assert "## Formulas" in markdown
        assert "## Condition Schema" in markdown
        assert "policy" in markdown
        assert "region" in markdown
    finally:
        if path.exists():
            path.unlink()
