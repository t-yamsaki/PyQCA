"""Hand-calculated necessity ratios for nonempty tiny sets (#21)."""

import numpy as np
import pandas as pd
import pytest

from qca import FSQCA, GSQCA, ConditionSpec, MultiValueLiteral, SetLiteral


@pytest.mark.parametrize("engine", [FSQCA, GSQCA])
@pytest.mark.parametrize("scale", [1.0, 1e-12, 1e-100, 1e-300])
def test_necessity_ratios_are_scale_invariant(engine, scale):
    kwargs = {"condition_types": {"A": "fuzzy"}} if engine is GSQCA else {}
    model = engine(
        pd.DataFrame({"A": [0.2 * scale, 0.4 * scale],
                      "Y": [0.3 * scale, 0.5 * scale]}),
        outcome="Y", conditions=["A"], **kwargs,
    )
    result = model.evaluate_necessity(SetLiteral("A"))
    assert result.consistency == pytest.approx(0.75)
    assert result.coverage == pytest.approx(1.0)
    assert result.n_cases_in == 0
    table = model.search_necessary_conditions(
        include_negations=False, min_consistency=0.7, min_coverage=0.9
    )
    assert table["condition"].tolist() == ["A"]
    assert table.iloc[0]["consistency"] == pytest.approx(0.75)
    assert table.iloc[0]["coverage"] == pytest.approx(1.0)
    assert table.iloc[0]["n_cases_in"] == 0


@pytest.mark.parametrize("x,y,consistency,coverage", [
    ([0., 0.], [3e-13, 5e-13], 0., 0.),
    ([2e-13, 4e-13], [0., 0.], 0., 0.),
    ([0., 0.], [0., 0.], 0., 0.),
    ([1e-13, 0.], [0., 1e-13], 0., 0.),
    ([1.], [1e-13], 1., 1e-13),
    ([1e-13], [1.], 1e-13, 1.),
    ([1e-13], [1e-13], 1., 1.),
])
def test_each_denominator_is_handled_independently(x, y, consistency, coverage):
    model = FSQCA(pd.DataFrame({"A": x, "Y": y}), outcome="Y", conditions=["A"])
    with np.errstate(divide="raise", invalid="raise"):
        result = model.evaluate_necessity(SetLiteral("A"))
    assert result.consistency == pytest.approx(consistency, abs=0)
    assert result.coverage == pytest.approx(coverage, abs=0)
    assert result.n_cases_in == sum(value > 0.5 for value in x)


def test_smallest_positive_representable_membership():
    tiny = np.nextafter(0., 1.)
    model = FSQCA(pd.DataFrame({"A": [tiny], "Y": [tiny]}),
                  outcome="Y", conditions=["A"])
    result = model.evaluate_necessity(SetLiteral("A"))
    assert result.consistency == result.coverage == 1.0
    assert result.n_cases_in == 0


def test_gsqca_value_memberships_use_the_same_ratios():
    model = GSQCA(
        pd.DataFrame({"low": [2e-13, 4e-13], "high": [0.8, 0.9],
                      "Y": [3e-13, 5e-13]}),
        outcome="Y", condition_specs=[
            ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
        ],
    )
    result = model.evaluate_necessity(MultiValueLiteral("A", 0))
    assert result.consistency == pytest.approx(0.75)
    assert result.coverage == pytest.approx(1.0)
    assert result.n_cases_in == 0
    table = model.search_necessary_conditions(min_consistency=0.7, min_coverage=0.9)
    assert table["condition"].tolist() == ["A=0"]


def test_negated_literal_and_triviality_with_tiny_outcome():
    model = FSQCA(pd.DataFrame({"A": [0., 0.], "Y": [3e-13, 5e-13]}),
                  outcome="Y", conditions=["A"])
    result = model.evaluate_necessity(SetLiteral("A", negated=True))
    assert result.consistency == 1.0
    assert result.coverage == pytest.approx(4e-13, abs=0)
    assert result.is_necessary is True
    assert result.is_trivial is True
    assert result.n_cases_in == 2
