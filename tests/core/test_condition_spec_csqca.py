from __future__ import annotations

import warnings

import pandas as pd
import pytest

import qca
from qca import ConditionSpec
from qca.engines import CSQCA, MixedQCA, QCAEngineBase
from qca.results import QCAFitResult


def crisp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D"],
            "skill": [1.0, 1.0, 0.0, 0.0],
            "resources": [1.0, 0.0, 1.0, 0.0],
            "outcome": [1.0, 1.0, 0.0, 0.0],
        }
    )


def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D"],
            "skill": [1.0, 1.0, 0.0, 0.0],
            "quality": [0.9, 0.7, 0.2, 0.1],
            "region": ["north", "south", "north", "south"],
            "outcome": [1.0, 0.8, 0.1, 0.0],
        }
    )


def test_condition_spec_normalizes_aliases_and_metadata() -> None:
    spec = ConditionSpec(
        name="skill",
        kind="csqca",
        domain=[0, 1],
        calibrated=True,
    )

    assert spec.kind == "crisp"
    assert spec.domain == (0, 1)
    assert spec.is_crisp
    assert spec.is_set
    assert not spec.is_multi
    assert spec.to_dict() == {
        "name": "skill",
        "kind": "crisp",
        "domain": (0, 1),
        "calibrated": True,
    }


def test_mixedqca_exposes_condition_specs_for_pyqca_schema() -> None:
    model = MixedQCA(
        data=mixed_df(),
        case_id="case",
        outcome="outcome",
        conditions=["skill", "quality", "region"],
        condition_types={
            "skill": "csqca",
            "quality": "fuzzy-set",
            "region": "mvqca",
        },
    )

    assert [spec.name for spec in model.condition_specs] == [
        "skill",
        "quality",
        "region",
    ]
    assert [spec.kind for spec in model.condition_specs] == [
        "crisp",
        "fuzzy",
        "multi",
    ]
    assert model.condition_spec_map["skill"].is_crisp
    assert model.condition_spec_map["quality"].is_fuzzy
    assert model.condition_spec_map["region"].is_multi
    assert model.set_conditions == ["skill", "quality"]
    assert model.multivalue_conditions == ["region"]


def test_mixedqca_builds_condition_specs_for_legacy_schema() -> None:
    model = MixedQCA(
        data=mixed_df(),
        case_id="case",
        outcome="outcome",
        set_conditions=["skill", "quality"],
        multivalue_conditions=["region"],
    )

    assert model.condition_spec_map["skill"].kind == "crisp"
    assert model.condition_spec_map["quality"].kind == "fuzzy"
    assert model.condition_spec_map["region"].kind == "multi"
    assert model.condition_types == {
        "skill": "crisp",
        "quality": "fuzzy",
        "region": "multi",
    }


def test_csqca_is_exported_from_public_modules() -> None:
    assert qca.CSQCA is CSQCA
    assert qca.ConditionSpec is ConditionSpec


def test_csqca_maps_all_conditions_to_crisp() -> None:
    model = CSQCA(
        data=crisp_df(),
        outcome="outcome",
        conditions=["skill", "resources"],
        case_id="case",
    )

    assert isinstance(model, QCAEngineBase)
    assert not isinstance(model, MixedQCA)
    assert model.qca_type == "csQCA"
    assert model.set_conditions == ["skill", "resources"]
    assert model.multivalue_conditions == []
    assert model.condition_types == {"skill": "crisp", "resources": "crisp"}
    assert model.condition_spec_map["skill"].is_crisp
    assert "CSQCA" in repr(model)


def test_csqca_accepts_legacy_set_conditions_alias() -> None:
    model = CSQCA(
        data=crisp_df(),
        outcome="outcome",
        set_conditions=["skill", "resources"],
        case_id="case",
    )

    assert model.conditions == ["skill", "resources"]
    assert model.condition_types == {"skill": "crisp", "resources": "crisp"}


def test_csqca_fit_returns_unified_result() -> None:
    model = CSQCA(
        data=crisp_df(),
        outcome="outcome",
        conditions=["skill", "resources"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(minimizer="qmc", consistency_cutoff=0.5)

    assert isinstance(result, QCAFitResult)
    assert result.settings["minimizer"] == "qmc"
    assert isinstance(result.truth_table, pd.DataFrame)


def test_csqca_rejects_non_crisp_condition_type() -> None:
    with pytest.raises(ValueError, match="CSQCA"):
        CSQCA(
            data=crisp_df(),
            outcome="outcome",
            conditions=["skill"],
            case_id="case",
            condition_types={"skill": "fuzzy"},
        )


def test_csqca_rejects_non_binary_condition_values() -> None:
    df = crisp_df()
    df.loc[1, "skill"] = 0.5

    with pytest.raises(ValueError, match="CSQCA requires crisp/binary"):
        CSQCA(
            data=df,
            outcome="outcome",
            conditions=["skill", "resources"],
            case_id="case",
        )


def test_csqca_rejects_non_binary_outcome_values() -> None:
    df = crisp_df()
    df.loc[1, "outcome"] = 0.5

    with pytest.raises(ValueError, match="CSQCA requires crisp/binary"):
        CSQCA(
            data=df,
            outcome="outcome",
            conditions=["skill", "resources"],
            case_id="case",
        )
