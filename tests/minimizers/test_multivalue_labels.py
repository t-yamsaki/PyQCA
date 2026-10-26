"""Keep category identities in formulas and exported solution labels (#18)."""

from itertools import product

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA
from qca.minimizers.implicant import Implicant, QMSolution


@pytest.mark.parametrize("backend", [
    "standard", "qmc", "set_cover", "exact_set_cover", "greedy_set_cover"
])
@pytest.mark.parametrize("category", [0, 1, 2])
def test_numeric_category_survives_all_outputs(category, backend, tmp_path):
    data = pd.DataFrame({"A": [0, 1, 2], "Y": [int(i == category) for i in range(3)]})
    result = MVQCA(data, outcome="Y", conditions=["A"]).fit(minimizer=backend)
    term = f"A={category}"
    assert set(result.formulas.values()) == {f"({term})"}
    assert set(result.solutions["term"]) == {term}
    assert set(result.qm_solution.summary()["term"]) == {term}
    assert result.qm_solution.coverage_table.index.tolist() == [term]
    assert result.qm_solution.complex_solution[0].pattern == (category,)
    assert result.consistency == pytest.approx(1)
    assert result.coverage == pytest.approx(1)
    assert term in result.to_markdown()
    path = tmp_path / "formula.txt"
    result.export_formula(path)
    assert path.read_text(encoding="utf-8") == f"({term})"


@pytest.mark.parametrize("engine", [CSQCA, FSQCA])
@pytest.mark.parametrize("positive", [0, 1])
def test_set_condition_notation_unchanged(engine, positive):
    data = pd.DataFrame({"A": [0, 1], "Y": [int(i == positive) for i in range(2)]})
    result = engine(data, outcome="Y", conditions=["A"]).fit()
    assert set(result.formulas.values()) == {"(A)" if positive else "(~A)"}


def test_mixed_gsqca_keeps_set_negation_and_category():
    data = pd.DataFrame(product([0, 1], [0, 1, 2]), columns=["S", "M"])
    data["Y"] = ((data.S == 0) & (data.M == 1)).astype(int)
    result = GSQCA(
        data, outcome="Y", conditions=["S", "M"],
        condition_types={"S": "crisp", "M": "multi"},
    ).fit()
    assert set(result.formulas.values()) == {"(~S * M=1)"}
    assert result.qm_solution.coverage_table.index.tolist() == ["~S * M=1"]
    assert result.consistency == result.coverage == 1


def test_category_union_and_tautology():
    for outcome, term in [([1, 1, 0], "A={0,1}"), ([1, 1, 1], "(tautology)")]:
        result = MVQCA(
            pd.DataFrame({"A": [0, 1, 2], "Y": outcome}),
            outcome="Y", conditions=["A"],
        ).fit()
        assert set(result.solutions["term"]) == {term}


def test_low_level_compatibility_and_explicit_types():
    imp = Implicant((0, 1, 2), frozenset({0}))
    assert imp.label(["A", "B", "C"]) == "~A * B * C=2"
    assert imp.label(["A", "B", "C"], [True] * 3) == "A=0 * B=1 * C=2"
    legacy = QMSolution(complex_solution=[imp], condition_names=["A", "B", "C"])
    assert legacy.summary().iloc[0]["term"] == "~A * B * C=2"
    with pytest.raises(ValueError, match="is_multivalue length"):
        imp.label(["A", "B", "C"], [True])
