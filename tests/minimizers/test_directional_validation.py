"""Validate theory before minimization, including empty truth tables (#30)."""

import pandas as pd
import pytest

from qca import CSQCA, GSQCA, ConditionSpec
from qca.minimizers.engine import QCAMinimizer
from qca.minimizers.set_cover import SetCoverMinimizer


@pytest.mark.parametrize(
    "backend", ["qmc", "standard", "set_cover", "exact_set_cover", "greedy_set_cover"]
)
@pytest.mark.parametrize("expectation", [{"AA": 1}, {"A": 2}, {"A": None}, {"A": [1]}])
def test_fit_rejects_invalid_theory(backend, expectation):
    model = CSQCA(
        pd.DataFrame({"A": [0, 1], "B": [0, 1], "Y": [0, 1]}),
        outcome="Y",
        conditions=["A", "B"],
    )
    with pytest.raises(ValueError, match="directional_expectations"):
        model.fit(minimizer=backend, directional_expectations=expectation)


@pytest.mark.parametrize("minimizer", [QCAMinimizer, SetCoverMinimizer])
@pytest.mark.parametrize("method", ["minimize", "remainder_summary"])
@pytest.mark.parametrize("expectation", [{"typo": 1}, {"A": 2}, {"M": "missing"}])
def test_low_level_validates_even_without_positive_rows(minimizer, method, expectation):
    engine = minimizer(
        ["A", "M"], {"A": False, "M": True}, {"A": [0, 1], "M": ["north", "south"]}
    )
    with pytest.raises(ValueError, match="directional_expectations"):
        getattr(engine, method)([], directional_expectations=expectation)


def test_valid_partial_none_and_empty_theory_semantics():
    model = CSQCA(
        pd.DataFrame({"A": [0, 1], "B": [0, 1], "Y": [0, 1]}),
        outcome="Y",
        conditions=["A", "B"],
    )
    partial = model.fit(directional_expectations={"A": 1}).qm_solution
    # The selected parsimonious solution is B; dropping B would belong to
    # the alternative parsimonious solution A, not this selected pair.
    assert [imp.pattern for imp in partial.parsimonious_solution] == [("*", 1)]
    assert [imp.pattern for imp in partial.intermediate_solution] == [(1, 1)]
    with pytest.warns(UserWarning, match="directional_expectations was not provided"):
        unspecified = model.fit(directional_expectations=None).qm_solution
    assert unspecified.intermediate_solution == unspecified.complex_solution
    unconstrained = model.fit(directional_expectations={}).qm_solution
    assert unconstrained.intermediate_solution == unconstrained.parsimonious_solution


def test_multivalue_theory_uses_declared_domain_including_unobserved_values():
    model = GSQCA(
        pd.DataFrame({"A": ["north", "south"], "B": [0, 1], "Y": [0, 1]}),
        outcome="Y",
        condition_specs=[
            ConditionSpec("A", "multi", domain=("north", "south", "east")),
            ConditionSpec("B", "crisp"),
        ],
    )
    model.fit(directional_expectations={"A": "east"})
    with pytest.raises(ValueError, match="outside its condition domain"):
        model.fit(directional_expectations={"A": "west"})
    engine = QCAMinimizer(
        ["A", "B"],
        {"A": True, "B": False},
        {"A": ["north", "south", "east"], "B": [0, 1]},
    )
    summary = engine.remainder_summary([], directional_expectations={"A": "east"})
    assert summary.loc[summary.theory_consistent, "A"].tolist() == ["east", "east"]
    assert engine.remainder_summary(
        [], directional_expectations={}
    ).theory_consistent.all()
    assert (
        engine.remainder_summary([], directional_expectations=None)
        .theory_consistent.isna()
        .all()
    )


def test_non_mapping_theory_is_rejected():
    engine = QCAMinimizer(["A"], {"A": False}, {"A": [0, 1]})
    with pytest.raises(TypeError, match="mapping or None"):
        engine.minimize([], directional_expectations=[("A", 1)])
