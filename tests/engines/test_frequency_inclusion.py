"""Truth-table display must use the minimizer's frequency cutoff (#20)."""

from io import StringIO

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA
from qca.core.results import truth_table_to_df
from qca.minimizers.remainder import classify_truth_table_rows


def make_model(engine, data):
    kwargs = {"outcome": "Y", "conditions": ["A", "B"]}
    if engine is GSQCA:
        kwargs["condition_types"] = {"A": "crisp", "B": "multi"}
    return engine(data, **kwargs)


@pytest.mark.parametrize("engine", [CSQCA, FSQCA, MVQCA, GSQCA])
@pytest.mark.parametrize("cutoff", [1, 2, 3])
def test_frequency_boundaries_match_minimization_and_exports(engine, cutoff):
    data = pd.DataFrame(
        [(0, 0, 1)] + [(0, 1, 1)] * 2
        + [(1, 0, 0)] * 2 + [(1, 1, 1)] * 3,
        columns=["A", "B", "Y"],
    )
    model = make_model(engine, data)
    result = model.fit(frequency_cutoff=cutoff)
    rows = model.build_truth_table(outcome_threshold=0.75, frequency_cutoff=cutoff)
    assert result.truth_table_rows == rows
    expected = {(0, 0): cutoff <= 1, (0, 1): cutoff <= 2,
                (1, 0): False, (1, 1): True}
    for row in rows:
        assert row.include is expected[(row.config["A"], row.config["B"])]
    assert result.qm_solution.n_positive == sum(expected.values())
    assert result.qm_solution.n_remainder == sum(row.n_cases < cutoff for row in rows)
    assert result.settings["frequency_cutoff"] == cutoff
    baseline = truth_table_to_df(model.build_truth_table(outcome_threshold=0.75))
    pd.testing.assert_frame_equal(
        result.truth_table.drop(columns="include"), baseline.drop(columns="include")
    )
    exported = pd.read_csv(StringIO(result.truth_table.to_csv(index=False)))
    assert exported["include"].tolist() == [row.include for row in rows]
    report_path_table = result._dataframe_to_markdown(result.truth_table)
    assert report_path_table in result.to_markdown()


@pytest.mark.parametrize("consistency,pri,frequency,expected", [
    (2 / 3, 2 / 3, 3, True),
    (0.7, 0.6, 3, False),
    (0.6, 0.7, 3, False),
    (0.6, 0.6, 4, False),
])
def test_all_three_cutoffs_apply(consistency, pri, frequency, expected):
    model = CSQCA(pd.DataFrame({"A": [1, 1, 1], "Y": [1, 1, 0]}),
                  outcome="Y", conditions=["A"])
    result = model.fit(consistency_cutoff=consistency, pri_cut=pri,
                       frequency_cutoff=frequency)
    row = result.truth_table_rows[0]
    assert row.include is expected
    assert result.qm_solution.n_positive == int(expected)
    assert row.outcome_raw_consist == pytest.approx(2 / 3)
    assert row.pri_consistency == pytest.approx(2 / 3)


def test_low_frequency_remainder_is_not_an_observed_negative():
    model = CSQCA(pd.DataFrame(
        [(0, 0, 1)] + [(0, 1, 1)] * 2 + [(1, 0, 0)] * 2 + [(1, 1, 1)] * 3,
        columns=["A", "B", "Y"],
    ), outcome="Y", conditions=["A", "B"])
    rows = model.build_truth_table(outcome_threshold=0.75, frequency_cutoff=2)
    patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
    positive, negative, remainder, contradiction = classify_truth_table_rows(
        rows, patterns, {p: i for i, p in enumerate(patterns)},
        ["A", "B"], [False, False], 0.75, 0.75, frequency_cutoff=2,
    )
    assert (positive, negative, remainder, contradiction) == ([1, 3], [2], [0], [])
    assert [row.include for row in rows if row.config == {"A": 0, "B": 0}] == [False]


def test_issue_example_has_no_included_rows_or_solution():
    model = CSQCA(pd.DataFrame({"A": [0, 1], "Y": [0, 1]}),
                  outcome="Y", conditions=["A"])
    result = model.fit(frequency_cutoff=2)
    assert not any(row.include for row in result.truth_table_rows)
    assert result.formulas == {}
    assert model.build_truth_table() == model.build_truth_table(frequency_cutoff=1)
