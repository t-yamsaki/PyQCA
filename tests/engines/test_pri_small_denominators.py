"""PRI boundaries checked against exact arithmetic on represented inputs."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from qca import FSQCA, GSQCA, ConditionSpec


def exact_pri(x, y):
    """Evaluate the published difference-of-sums formula with rational numbers."""
    xs = [Fraction(float(v)) for v in x]
    ys = [Fraction(float(v)) for v in y]
    overlap = sum(min(a, b) for a, b in zip(xs, ys, strict=True))
    triple = sum(min(a, b, 1 - b) for a, b in zip(xs, ys, strict=True))
    denominator = sum(xs) - triple
    return float((overlap - triple) / denominator) if denominator else 0.0


@pytest.mark.parametrize("engine", [FSQCA, GSQCA])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "x,y",
    [
        ([0.8, 0.6, 0.2], [0.9, 0.2, 0.1]),
        ([1e-13, 2e-13], [1.0, 0.0]),
        ([1e-100, 2e-100], [1.0, 0.0]),
        ([np.nextafter(0.0, 1.0)], [1.0]),
        ([0.0, 0.0], [0.4, 0.6]),
        ([0.2, 0.3], [0.4, 0.6]),
        # Common mass swamps the positive residual in a difference of sums.
        ([0.25, 1e-20], [0.5, 1.0]),
        ([0.25, 1e-20, 2e-20], [0.5, 1.0, 0.0]),
        ([0.25, np.nextafter(0.25, 1.0)], [0.5, 0.75]),
    ],
)
def test_pri_matches_exact_reference(engine, reverse, x, y):
    if reverse:
        x, y = x[::-1], y[::-1]
    kwargs = {"condition_types": {"A": "fuzzy"}} if engine is GSQCA else {}
    model = engine(
        pd.DataFrame({"A": x, "Y": y}), outcome="Y", conditions=["A"], **kwargs
    )
    assert model._calc_row_pri_consistency({"A": 1}) == pytest.approx(
        exact_pri(x, y), rel=1e-14, abs=0.0
    )


@pytest.mark.parametrize("minimizer", ["standard", "set_cover"])
@pytest.mark.parametrize("scale", [1.0, 1e-13, 1e-100, np.nextafter(0.0, 1.0)])
def test_tiny_membership_retains_truth_table_and_solutions(minimizer, scale):
    model = GSQCA(
        pd.DataFrame({"low": [scale, scale], "high": [0.0, 0.0], "Y": [1, 1]}),
        outcome="Y",
        condition_specs=[
            ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
        ],
    )
    (row,) = model.build_truth_table(pri_cut=0.5)
    assert row.config == {"A": 0}
    assert row.pri_consistency == 1.0
    assert row.include
    result = model.fit(pri_cut=0.5, directional_expectations={}, minimizer=minimizer)
    assert result.minimization.n_positive == 1
    assert result.formulas == {
        "complex": "(A=0)",
        "parsimonious": "((tautology))",
        "intermediate": "((tautology))",
    }


@pytest.mark.parametrize("minimizer", ["standard", "set_cover"])
def test_cancellation_pri_cut_controls_final_solution(minimizer):
    model = GSQCA(
        pd.DataFrame(
            {"low": [0.25, 1e-20, 2e-20], "high": [0.0] * 3, "Y": [0.5, 1.0, 0.0]}
        ),
        outcome="Y",
        condition_specs=[
            ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
        ],
    )
    (row,) = model.build_truth_table(pri_cut=0.3)
    assert row.pri_consistency == pytest.approx(1 / 3)
    assert row.include
    for cutoff, expected in [(0.3, 1), (0.4, 0)]:
        result = model.fit(
            pri_cut=cutoff, directional_expectations={}, minimizer=minimizer
        )
        assert result.minimization.n_positive == expected
        assert (result.formula is not None) == bool(expected)


@pytest.mark.parametrize("minimizer", ["standard", "set_cover"])
def test_exact_zero_pri_denominator_still_excludes_row(minimizer):
    model = GSQCA(
        pd.DataFrame({"low": [0.2, 0.3], "high": [0.0, 0.0], "Y": [0.4, 0.6]}),
        outcome="Y",
        condition_specs=[
            ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
        ],
    )
    (row,) = model.build_truth_table(pri_cut=0.5)
    assert row.pri_consistency == 0.0
    assert not row.include
    with pytest.warns(UserWarning, match="No positive rows"):
        result = model.fit(
            pri_cut=0.5, directional_expectations={}, minimizer=minimizer
        )
    assert result.minimization.n_positive == 0
    assert all(value is None for value in result.formulas.values())
