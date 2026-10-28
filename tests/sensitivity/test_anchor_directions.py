"""Anchor calibration renames dictionary theory alongside conditions (#35)."""

import pandas as pd
import pytest

from qca import AnchorSensitivity

GRID = {"full_out": [0.0], "crossover": [1.0], "full_in": [2.0]}


def run(direction, output="A_fuzzy", conditions=None):
    data = pd.DataFrame({"A": [0.0, 2.0], "B": [0, 1], "Y": [0, 1]})
    sweep = AnchorSensitivity(data, outcome="Y", conditions=conditions or ["A", "B"])
    return sweep.grid("A", GRID, out_col=output, include="?", dir_exp=direction)


@pytest.mark.parametrize("output", ["A_fuzzy", "membership", "A"])
@pytest.mark.parametrize("other", [{}, {"B": 0}, {"B": 1}])
def test_original_and_calibrated_names_have_identical_intermediate_solutions(
    output, other
):
    original_direction = {"A": 1, **other}
    before = original_direction.copy()
    original = run(original_direction, output)
    calibrated = run({output: 1, **other}, output)
    assert original_direction == before
    assert original.results[0] is not None
    pd.testing.assert_frame_equal(original.summary_df, calibrated.summary_df)
    assert original.results[0].minimization.intermediate_solution == (
        calibrated.results[0].minimization.intermediate_solution
    )
    if not other:
        # The selected parsimonious solution is B, so intermediate A*B
        # must not use the remainder belonging only to alternative solution A.
        assert original.results[0].formulas["parsimonious"] == "(B)"
        assert original.summary_df.iloc[0]["expression"] == f"({output} * B)"
        assert original.summary_df.iloc[0]["solution_type"] == "intermediate"


@pytest.mark.parametrize(
    "direction", [1, [1], [1, 1], (1, 1), {"A": 1, "B": 1}, {"A_fuzzy": 1, "B": 1}]
)
def test_equivalent_direction_forms_match(direction):
    expected = run({"A": 1, "B": 1})
    actual = run(direction)
    pd.testing.assert_frame_equal(actual.summary_df, expected.summary_df)


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_aliases_are_reported_independent_of_order(reverse):
    entries = [("A", 1), ("A_fuzzy", 0)]
    result = run(dict(entries[::-1] if reverse else entries))
    assert result.results == [None]
    assert "Conflicting dir_exp" in result.summary_df.iloc[0]["error"]
    assert isinstance(result.details[0]["error"], ValueError)


def test_consistent_aliases_are_accepted():
    result = run({"A": 1, "A_fuzzy": 1})
    pd.testing.assert_frame_equal(result.summary_df, run({"A": 1}).summary_df)


@pytest.mark.parametrize("direction", [{"typo": 1}, {"A": 2}])
def test_invalid_theory_still_reaches_minimizer_validation(direction):
    result = run(direction)
    assert result.results == [None]
    assert "directional_expectations" in result.summary_df.iloc[0]["error"]


def test_condition_order_is_preserved_for_sequence_directions():
    expected = run({"A": 1, "B": 0}, conditions=["B", "A"])
    actual = run([0, 1], conditions=["B", "A"])
    pd.testing.assert_frame_equal(actual.summary_df, expected.summary_df)
