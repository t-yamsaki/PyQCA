"""Independent numerical regressions for the standard PRI definition."""

import warnings

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA


@pytest.mark.parametrize("engine", [CSQCA, MVQCA, FSQCA, GSQCA])
@pytest.mark.parametrize("minimizer", ["standard", "set_cover"])
def test_crisp_pri_and_fit_inclusion(engine, minimizer):
    data = pd.DataFrame({"A": [1, 1, 1, 1, 1, 0], "Y": [1, 1, 1, 1, 0, 0]})
    kwargs = {"condition_types": {"A": "crisp"}} if engine is GSQCA else {}
    model = engine(data=data, outcome="Y", conditions=["A"], **kwargs)
    row = next(r for r in model.build_truth_table(pri_cut=0.78) if r.config["A"] == 1)
    # Binary Y makes the triple overlap zero: PRI = 4 / 5.
    assert row.pri_consistency == pytest.approx(0.8)
    assert row.include
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        retained = model.fit(consistency_cutoff=0.75, pri_cut=0.78, minimizer=minimizer)
        excluded = model.fit(consistency_cutoff=0.75, pri_cut=0.81, minimizer=minimizer)
    assert retained.minimization.n_positive == 1
    assert retained.formula is not None
    assert excluded.minimization.n_positive == 0
    assert excluded.formula is None


def test_fuzzy_pri_uses_triple_overlap_over_all_cases():
    model = FSQCA(
        pd.DataFrame({"A": [0.8, 0.6, 0.2], "Y": [0.9, 0.2, 0.1]}),
        outcome="Y", conditions=["A"],
    )
    row = next(r for r in model.build_truth_table() if r.config["A"] == 1)
    # All three cases: sum(X)=1.6, sum(min(X,Y))=1.1,
    # triple overlap=0.1+0.2+0.1=0.4; PRI=(1.1-0.4)/(1.6-0.4).
    assert row.outcome_raw_consist == pytest.approx(11 / 16)
    assert row.pri_consistency == pytest.approx(7 / 12)


@pytest.mark.parametrize("memberships", [[0.0, 0.0], [0.2, 0.3]])
def test_zero_pri_denominator_returns_finite_zero(memberships):
    model = FSQCA(
        pd.DataFrame({"A": memberships, "Y": [0.4, 0.6]}),
        outcome="Y", conditions=["A"],
    )
    # Either empty X, or X entirely inside both Y and ~Y.
    assert model._calc_row_pri_consistency({"A": 1}) == 0.0
