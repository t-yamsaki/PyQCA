"""Independent examples for exact universal necessity diagnostics (#19)."""

import numpy as np
import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA, MultiValueLiteral, SetLiteral
from qca.core.results import NecessityResult, necessity_results_to_df
from qca.mlqca.csqca import build_necessity_table


@pytest.mark.parametrize("engine", [CSQCA, FSQCA, GSQCA])
@pytest.mark.parametrize("x,y,expected", [
    ([1] * 10, [1] + [0] * 9, True),
    ([0, 1], [0, 1], False),
    ([1, 1, 0], [1, 0, 0], False),
    ([1, 1], [1, 1], True),
    ([0, 0], [0, 1], False),
    ([1, 1], [0, 0], False),
    ([0, 0], [0, 0], False),
    ([0, 1], [1, 0], False),
])
def test_exact_universal_diagnostic(engine, x, y, expected):
    kwargs = {"conditions": ["A"], "outcome": "Y"}
    if engine is GSQCA:
        kwargs["condition_types"] = {"A": "crisp"}
    model = engine(pd.DataFrame({"A": x, "Y": y}), **kwargs)
    result = model.evaluate_necessity(SetLiteral("A"))
    assert result.is_trivial is expected
    assert result.is_universal is all(v == 1 for v in x)
    assert result.to_dict()["is_trivial"] is expected
    assert result.to_series()["is_trivial"] == expected
    assert necessity_results_to_df([result]).iloc[0]["is_trivial"] == expected
    assert ("Trivial necessary condition" in str(result)) is expected
    assert result.consistency == pytest.approx(
        sum(min(a, b) for a, b in zip(x, y, strict=True)) / sum(y) if sum(y) else 0
    )
    assert result.coverage == pytest.approx(
        sum(min(a, b) for a, b in zip(x, y, strict=True)) / sum(x) if sum(x) else 0
    )


@pytest.mark.parametrize("x", [[0.9, 0.9], [1, np.nextafter(1., 0.)], [0.2, 0.8]])
def test_fuzzy_nonuniversal_conditions_not_flagged(x):
    model = FSQCA(pd.DataFrame({"A": x, "Y": [0.1, 0.2]}),
                  outcome="Y", conditions=["A"])
    result = model.evaluate_necessity(SetLiteral("A"))
    assert result.is_necessary is True
    assert result.is_trivial is False


def test_low_coverage_is_not_universality():
    model = FSQCA(pd.DataFrame({"A": [1, 1, 0], "Y": [0.01, 0, 0]}),
                  outcome="Y", conditions=["A"])
    result = model.evaluate_necessity(SetLiteral("A"))
    assert result.coverage == pytest.approx(0.005)
    assert result.is_trivial is False


def test_negated_universal_condition_and_ml_table():
    model = CSQCA(pd.DataFrame({"A": [0, 0], "Y": [0, 1]}),
                  outcome="Y", conditions=["A"])
    assert model.evaluate_necessity(SetLiteral("A", negated=True)).is_trivial is True
    table = build_necessity_table(model).set_index("condition")
    assert bool(table.loc["~A", "is_trivial"]) is True
    assert bool(table.loc["A", "is_trivial"]) is False


def test_multivalue_union_can_be_universal():
    model = MVQCA(pd.DataFrame({"A": [0, 1], "Y": [0, 1]}),
                  outcome="Y", conditions=["A"])
    assert model.evaluate_necessity(MultiValueLiteral("A", 1)).is_trivial is False
    union = MultiValueLiteral("A", frozenset({0, 1}))
    assert model.evaluate_necessity(union).is_trivial is True


def test_manual_results_without_evidence_are_unknown():
    for coverage in [0.01, 0.9, 1.0]:
        result = NecessityResult("A", 1.0, coverage, 2)
        assert result.is_trivial is None
        assert result.to_dict()["is_trivial"] is None
        assert "Triviality unknown" in str(result)
        assert necessity_results_to_df([result]).iloc[0]["is_trivial"] is None
    assert NecessityResult("A", 0.8, 0.01, 2).is_trivial is False
    assert NecessityResult("A", 0.8, 0.01, 2, is_universal=True).is_trivial is False
