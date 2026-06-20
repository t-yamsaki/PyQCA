"""Engine architecture tests for first-class csQCA/fsQCA/mvQCA modules."""

from __future__ import annotations

import pandas as pd
import pytest

import qca
from qca.engines import CSQCA, FSQCA, MVQCA, MixedQCA, QCAEngineBase
from qca.engines.csqca import CSQCA as ModuleCSQCA
from qca.engines.fsqca import FSQCA as ModuleFSQCA
from qca.engines.mvqca import MVQCA as ModuleMVQCA


def test_single_family_engines_are_first_class_modules() -> None:
    assert CSQCA is ModuleCSQCA
    assert FSQCA is ModuleFSQCA
    assert MVQCA is ModuleMVQCA
    assert qca.CSQCA is ModuleCSQCA
    assert qca.FSQCA is ModuleFSQCA
    assert qca.MVQCA is ModuleMVQCA


def test_engines_share_core_without_inheriting_from_mixed_facade() -> None:
    for engine_cls in (CSQCA, FSQCA, MVQCA):
        assert issubclass(engine_cls, QCAEngineBase)
        assert not issubclass(engine_cls, MixedQCA)

    assert issubclass(MixedQCA, QCAEngineBase)


def test_shared_core_lives_in_base_module() -> None:
    assert QCAEngineBase.__module__ == "qca.engines.base"
    assert MixedQCA.__module__ == "qca.engines.mixed"
    assert CSQCA.__module__ == "qca.engines.csqca"
    assert FSQCA.__module__ == "qca.engines.fsqca"
    assert MVQCA.__module__ == "qca.engines.mvqca"


def test_csqca_validates_crisp_binary_outcome() -> None:
    df = pd.DataFrame(
        {
            "case": ["A", "B", "C"],
            "A": [1.0, 0.0, 1.0],
            "outcome": [1.0, 0.5, 0.0],
        }
    )

    with pytest.raises(ValueError, match="CSQCA requires crisp/binary"):
        CSQCA(data=df, case_id="case", outcome="outcome", conditions=["A"])


def test_fsqca_validates_set_only_schema() -> None:
    df = pd.DataFrame(
        {
            "case": ["A", "B", "C"],
            "A": [0.9, 0.2, 0.8],
            "region": ["north", "south", "north"],
            "outcome": [0.8, 0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="FSQCA"):
        FSQCA(
            data=df,
            case_id="case",
            outcome="outcome",
            conditions=["A", "region"],
            condition_types={"A": "fuzzy", "region": "multi"},
        )


def test_mvqca_validates_multivalue_only_schema() -> None:
    df = pd.DataFrame(
        {
            "case": ["A", "B", "C"],
            "A": [1.0, 0.0, 1.0],
            "region": ["north", "south", "north"],
            "outcome": [0.8, 0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="MVQCA"):
        MVQCA(
            data=df,
            case_id="case",
            outcome="outcome",
            conditions=["A", "region"],
            condition_types={"A": "crisp", "region": "multi"},
        )


def test_mvqca_rejects_degenerate_multivalue_domain() -> None:
    df = pd.DataFrame(
        {
            "case": ["A", "B", "C"],
            "region": ["north", "north", "north"],
            "outcome": [0.8, 0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="at least two observed values"):
        MVQCA(data=df, case_id="case", outcome="outcome", conditions=["region"])
