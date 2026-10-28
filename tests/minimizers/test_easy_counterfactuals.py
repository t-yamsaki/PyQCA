"""Independent logical checks of intermediate solutions (Issue #50)."""

from itertools import product

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA, AnchorSensitivity, ConditionSpec
from qca.minimizers.remainder import filter_easy_remainders

NAMES = ["DEV", "URB", "LIT", "IND", "STB"]
PATTERNS = list(product([0, 1], repeat=5))


def lf_table():
    # Published LF truth-table classifications, not a recalibration of raw LF.
    ids = [32, 22, 1, 2, 5, 24, 6, 23, 31]
    data = pd.DataFrame([PATTERNS[i - 1] for i in ids], columns=NAMES)
    data["Y"] = [1, 1] + [0] * 7
    return data


def truth_set(terms, universe):
    # Evaluate patterns rather than Implicant.covered (positive rows only).
    return {
        row
        for row in universe
        if any(
            all(
                term == "*"
                or (value in term if isinstance(term, frozenset) else value == term)
                for value, term in zip(row, imp.pattern, strict=True)
            )
            for imp in terms
        )
    }


@pytest.mark.parametrize("engine", [CSQCA, FSQCA, MVQCA, GSQCA])
@pytest.mark.parametrize("backend", ["standard", "set_cover", "greedy_set_cover"])
def test_lf_published_solutions(engine, backend):
    kwargs = (
        {"condition_types": dict.fromkeys(NAMES, "crisp")} if engine is GSQCA else {}
    )
    model = engine(lf_table(), outcome="Y", conditions=NAMES, **kwargs)
    result = model.fit(
        directional_expectations=dict.fromkeys(NAMES, 1), minimizer=backend
    )
    sol = result.qm_solution
    complex_set = {PATTERNS[31], PATTERNS[21]}
    intermediate_set = {
        row
        for row in PATTERNS
        if row[0] and row[2] and row[4] and (row[1] or not row[3])
    }
    parsimonious_set = {
        row for row in PATTERNS if (row[0] and not row[3]) or (row[1] and row[4])
    }
    assert truth_set(sol.complex_solution, PATTERNS) == complex_set
    assert truth_set(sol.intermediate_solution, PATTERNS) == intermediate_set
    assert truth_set(sol.parsimonious_solution, PATTERNS) == parsimonious_set
    assert intermediate_set - complex_set == {PATTERNS[29]}
    assert complex_set < intermediate_set < parsimonious_set


@pytest.mark.parametrize(
    "expectations,expected",
    [
        ({"A": 1, "B": 1, "C": 1}, {3, 6, 7}),
        ({"A": 0, "B": 0, "C": 0}, {0, 1, 4}),
        ({"A": 1}, {0, 1, 3, 4, 6, 7}),
        ({}, {0, 1, 3, 4, 6, 7}),
    ],
)
def test_easy_and_difficult_from_two_observed_positives(expectations, expected):
    # Positives 010 and 101: unchanged contrary literals are allowed;
    # moving a specified coordinate away from theory is not.
    patterns = list(product([0, 1], repeat=3))
    selected = filter_easy_remainders(
        [0, 1, 3, 4, 6, 7],
        [],
        [],
        patterns,
        ["A", "B", "C"],
        expectations,
        [2, 5],
        [("*", "*", "*")],
    )
    assert set(selected) == expected


def test_easy_remainders_exclude_negative_contradictory_and_other_solutions():
    patterns = list(product([0, 1], repeat=3))
    assert filter_easy_remainders(
        list(range(8)),
        [6],
        [7],
        patterns,
        ["A", "B", "C"],
        {},
        [4],
        [(1, "*", "*")],
    ) == [4, 5]


@pytest.mark.parametrize("backend", ["standard", "set_cover"])
def test_all_two_condition_tables_preserve_selected_pair_inclusion(backend):
    universe = list(product([0, 1], repeat=2))
    for labels in product([0, 1, None], repeat=4):
        if 1 not in labels:
            continue
        observed = [
            (*row, label)
            for row, label in zip(universe, labels, strict=True)
            if label is not None
        ]
        data = pd.DataFrame(observed, columns=["A", "B", "Y"])
        for directions in [{}, {"A": 1}, {"A": 0}, {"A": 1, "B": 0}]:
            sol = (
                CSQCA(data, "Y", conditions=["A", "B"])
                .fit(directional_expectations=directions, minimizer=backend)
                .qm_solution
            )
            c = truth_set(sol.complex_solution, universe)
            i = truth_set(sol.intermediate_solution, universe)
            p = truth_set(sol.parsimonious_solution, universe)
            assert c <= i <= p
            assert c == {
                row for row, label in zip(universe, labels, strict=True) if label == 1
            }
            if not directions:
                assert i == p


@pytest.mark.parametrize("backend", ["standard", "set_cover"])
def test_multivalue_expected_category_is_not_a_numeric_order(backend):
    data = pd.DataFrame({"M": ["middle", "other"], "B": [1, 0], "Y": [1, 0]})
    model = GSQCA(
        data,
        outcome="Y",
        condition_specs=[
            ConditionSpec("M", "multi", domain=("middle", "preferred", "other")),
            ConditionSpec("B", "crisp"),
        ],
    )
    universe = list(product([0, 1], ["middle", "preferred", "other"]))
    sol = model.fit(
        directional_expectations={"M": "preferred", "B": 1}, minimizer=backend
    ).qm_solution
    assert truth_set(sol.intermediate_solution, universe) == {
        (1, "middle"),
        (1, "preferred"),
    }


def test_anchor_sweep_uses_relational_counterfactuals():
    result = AnchorSensitivity(lf_table(), outcome="Y", conditions=NAMES).grid(
        "DEV",
        {"full_out": [0], "crossover": [0.5], "full_in": [1]},
        out_col="DEV_set",
        include="?",
        dir_exp=dict.fromkeys(NAMES, 1),
    )
    fit = result.results[0]
    assert fit is not None
    assert truth_set(fit.qm_solution.intermediate_solution, PATTERNS) == {
        PATTERNS[21],
        PATTERNS[29],
        PATTERNS[31],
    }


@pytest.mark.parametrize("backend", ["standard", "set_cover"])
def test_lf_inversion_and_permutation_preserve_logic(backend):
    data = lf_table()
    data[NAMES] = 1 - data[NAMES]
    names = NAMES[::-1]
    fit = CSQCA(data.iloc[::-1], outcome="Y", conditions=names).fit(
        directional_expectations=dict.fromkeys(names, 0), minimizer=backend
    )
    transformed = {
        tuple(1 - value for value in row[::-1])
        for row in [PATTERNS[21], PATTERNS[29], PATTERNS[31]]
    }
    assert truth_set(fit.qm_solution.intermediate_solution, PATTERNS) == transformed


@pytest.mark.parametrize("backend", ["standard", "set_cover"])
def test_disabling_remainders_bounds_intermediate_by_complex(backend):
    fit = CSQCA(lf_table(), outcome="Y", conditions=NAMES).fit(
        directional_expectations=dict.fromkeys(NAMES, 1),
        minimizer=backend,
        include_remainders_in_parsimonious=False,
    )
    sol = fit.qm_solution
    assert truth_set(sol.intermediate_solution, PATTERNS) == truth_set(
        sol.complex_solution, PATTERNS
    )
