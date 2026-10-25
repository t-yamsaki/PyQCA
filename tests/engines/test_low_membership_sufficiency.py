"""Hand-calculated regressions for nonempty low-membership sets."""

import pandas as pd
import pytest

from qca import FSQCA, GSQCA, SetLiteral
from qca.core.results import SufficiencyResult


@pytest.mark.parametrize("engine", [FSQCA, GSQCA])
@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_low_membership_metrics_across_evaluation_apis(engine, scale):
    data = pd.DataFrame(
        {"A": [0.2 * scale, 0.4 * scale], "Y": [0.3 * scale, 0.5 * scale]}
    )
    kwargs = {"condition_types": {"A": "fuzzy"}} if engine is GSQCA else {}
    model = engine(data=data, outcome="Y", conditions=["A"], **kwargs)
    term = [SetLiteral("A")]
    for result in (
        model.evaluate_sufficiency(term),
        model.evaluate_solution([term]),
        model.compute_solution_coverages([term])[0],
    ):
        assert result.consistency == pytest.approx(1.0)
        assert result.raw_coverage == pytest.approx(0.75)
        assert result.n_cases_in == 0
        assert not result.is_sufficient
        assert result.strength == "no_cases_above_crossover"
        assert "[empty]" not in str(result)
    assert model.compute_solution_coverages([term])[0].unique_coverage == pytest.approx(
        0.75
    )
    assert model._calc_row_consistency({"A": 1}) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x,y,consistency,coverage,strength",
    [
        ([0.0, 0.0], [0.3, 0.5], 0.0, 0.0, "empty"),
        ([0.2, 0.4], [0.0, 0.0], 0.0, 0.0, "no_cases_above_crossover"),
        ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0, "empty"),
    ],
)
def test_zero_denominators_and_zero_overlap(x, y, consistency, coverage, strength):
    model = FSQCA(pd.DataFrame({"A": x, "Y": y}), outcome="Y", conditions=["A"])
    term = [SetLiteral("A")]
    result = model.compute_solution_coverages([term])[0]
    assert result.consistency == consistency
    assert result.raw_coverage == coverage
    assert result.unique_coverage == 0.0
    assert result.strength == strength
    assert model.evaluate_sufficiency(term).unique_coverage is None


def test_overlap_and_candidate_filtering_are_separate_from_metrics():
    model = FSQCA(
        pd.DataFrame({"A": [0.2, 0.4], "B": [0.1, 0.3], "Y": [0.3, 0.5]}),
        outcome="Y",
        conditions=["A", "B"],
    )
    terms = [[SetLiteral("A")], [SetLiteral("B")]]
    a, b = model.compute_solution_coverages(terms)
    assert a.unique_coverage == pytest.approx(0.25)
    assert b.unique_coverage == 0.0
    assert model.evaluate_solution(terms).raw_coverage == pytest.approx(0.75)
    accepted = model.search_sufficient_configurations(
        include_negations=False, min_cases=0
    )
    assert "A" in accepted["antecedent"].tolist()
    with pytest.warns(UserWarning, match="min_cases"):
        assert model.search_sufficient_configurations(include_negations=False).empty


def test_legacy_result_without_membership_sum_does_not_claim_empty():
    result = SufficiencyResult("A", 0.0, 0.0, None, 0)
    assert result.strength == "no_cases_above_crossover"
