from __future__ import annotations

import pandas as pd
import pytest

import qca
from qca.engines import FSQCA, MVQCA, MixedQCA, QCAEngineBase
from qca.results import QCAFitResult


def fs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D"],
            "skill": [0.9, 0.8, 0.2, 0.1],
            "resources": [1.0, 0.0, 1.0, 0.0],
            "outcome": [0.9, 0.7, 0.1, 0.2],
        }
    )


def mv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D", "E", "F"],
            "sector": ["public", "public", "private", "private", "hybrid", "hybrid"],
            "region": ["north", "south", "north", "south", "north", "south"],
            "outcome": [1.0, 0.8, 0.1, 0.2, 0.9, 0.1],
        }
    )


def test_engines_are_exported_from_public_modules() -> None:
    assert qca.FSQCA is FSQCA
    assert qca.MVQCA is MVQCA


def test_fsqca_maps_conditions_to_set_conditions() -> None:
    model = FSQCA(
        data=fs_df(),
        outcome="outcome",
        conditions=["skill", "resources"],
        case_id="case",
    )

    assert isinstance(model, QCAEngineBase)
    assert not isinstance(model, MixedQCA)
    assert model.qca_type == "fsQCA"
    assert model.set_conditions == ["skill", "resources"]
    assert model.multivalue_conditions == []
    assert model.conditions == ["skill", "resources"]
    assert model.condition_types == {"skill": "fuzzy", "resources": "crisp"}
    assert "FSQCA" in repr(model)


def test_fsqca_accepts_explicit_set_condition_types() -> None:
    model = FSQCA(
        data=fs_df(),
        outcome="outcome",
        conditions=["skill", "resources"],
        case_id="case",
        condition_types={"skill": "fsqca", "resources": "csqca"},
    )

    assert model.set_conditions == ["skill", "resources"]
    assert model.multivalue_conditions == []
    assert model.condition_types == {"skill": "fuzzy", "resources": "crisp"}


def test_fsqca_accepts_legacy_set_conditions_alias() -> None:
    model = FSQCA(
        data=fs_df(),
        outcome="outcome",
        set_conditions=["skill", "resources"],
        case_id="case",
    )

    assert model.conditions == ["skill", "resources"]
    assert model.set_conditions == ["skill", "resources"]


def test_fsqca_rejects_multi_condition_type() -> None:
    with pytest.raises(ValueError, match="FSQCA"):
        FSQCA(
            data=fs_df(),
            outcome="outcome",
            conditions=["skill"],
            case_id="case",
            condition_types={"skill": "multi"},
        )


def test_fsqca_fit_returns_unified_result() -> None:
    result = FSQCA(
        data=fs_df(),
        outcome="outcome",
        conditions=["skill", "resources"],
        case_id="case",
    ).fit(minimizer="qmc", consistency_cutoff=0.6)

    assert isinstance(result, QCAFitResult)
    assert result.settings["minimizer"] == "qmc"
    assert list(result.truth_table.columns)


def test_mvqca_maps_conditions_to_multivalue_conditions() -> None:
    model = MVQCA(
        data=mv_df(),
        outcome="outcome",
        conditions=["sector", "region"],
        case_id="case",
    )

    assert isinstance(model, QCAEngineBase)
    assert not isinstance(model, MixedQCA)
    assert model.qca_type == "mvQCA"
    assert model.set_conditions == []
    assert model.multivalue_conditions == ["sector", "region"]
    assert model.conditions == ["sector", "region"]
    assert model.condition_types == {"sector": "multi", "region": "multi"}
    assert "MVQCA" in repr(model)


def test_mvqca_accepts_explicit_multi_condition_types() -> None:
    model = MVQCA(
        data=mv_df(),
        outcome="outcome",
        conditions=["sector", "region"],
        case_id="case",
        condition_types={"sector": "mvqca", "region": "multi-value"},
    )

    assert model.multivalue_conditions == ["sector", "region"]
    assert model.condition_types == {"sector": "multi", "region": "multi"}


def test_mvqca_accepts_legacy_multivalue_conditions_alias() -> None:
    model = MVQCA(
        data=mv_df(),
        outcome="outcome",
        multivalue_conditions=["sector", "region"],
        case_id="case",
    )

    assert model.conditions == ["sector", "region"]
    assert model.multivalue_conditions == ["sector", "region"]


def test_mvqca_rejects_set_condition_type() -> None:
    with pytest.raises(ValueError, match="MVQCA"):
        MVQCA(
            data=mv_df(),
            outcome="outcome",
            conditions=["sector"],
            case_id="case",
            condition_types={"sector": "fuzzy"},
        )


def test_mvqca_fit_returns_unified_result() -> None:
    result = MVQCA(
        data=mv_df(),
        outcome="outcome",
        conditions=["sector", "region"],
        case_id="case",
    ).fit(minimizer="qmc", consistency_cutoff=0.6)

    assert isinstance(result, QCAFitResult)
    assert result.settings["minimizer"] == "qmc"
    assert list(result.truth_table.columns)


def test_engines_require_conditions() -> None:
    with pytest.raises(ValueError, match="requires conditions"):
        FSQCA(data=fs_df(), outcome="outcome", case_id="case")

    with pytest.raises(ValueError, match="at least one condition"):
        MVQCA(data=mv_df(), outcome="outcome", conditions=[], case_id="case")


def test_engines_reject_duplicate_condition_aliases() -> None:
    with pytest.raises(ValueError, match="either conditions or set_conditions"):
        FSQCA(
            data=fs_df(),
            outcome="outcome",
            conditions=["skill"],
            set_conditions=["resources"],
            case_id="case",
        )

    with pytest.raises(ValueError, match="either conditions or multivalue_conditions"):
        MVQCA(
            data=mv_df(),
            outcome="outcome",
            conditions=["sector"],
            multivalue_conditions=["region"],
            case_id="case",
        )
