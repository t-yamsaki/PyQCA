"""Reserved threshold outcome names must never silently replace data (#46)."""

import pandas as pd
import pytest

from qca import ThresholdSweep
from qca.sweep import prepare_threshold_data
from qca.sweep.threshold import OUTCOME_COLUMN


def run(sweep, path, pre=False):
    conditions = sweep.conditions
    kwargs = {"pre_calibrated": [conditions[0]] if pre else None, "dir_exp": {}}
    thresholds = dict.fromkeys(conditions, 0.5)
    if path == "outcome":
        return sweep.outcome([0.5], condition_thresholds=thresholds, **kwargs)
    if path == "condition":
        return sweep.condition(
            conditions[0],
            [0.5],
            outcome_threshold=0.5,
            condition_thresholds=thresholds,
            **kwargs,
        )
    grid = {name: [0.5] for name in conditions}
    if path == "multi_condition":
        return sweep.multi_condition(grid, outcome_threshold=0.5, **kwargs)
    return sweep.dual(grid, [0.5], **kwargs)


@pytest.mark.parametrize("role", ["condition", "case_id"])
@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("negate", [False, True])
def test_preparation_rejects_reserved_name_before_binarization(
    role, pre, negate, monkeypatch
):
    condition = OUTCOME_COLUMN if role == "condition" else "A"
    case_id = OUTCOME_COLUMN if role == "case_id" else "case"
    data = pd.DataFrame({condition: [0.0, 1.0], case_id: ["c1", "c2"], "Y": [1, 0]})
    before = data.copy(deep=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Reserved names must be rejected before binarization")

    monkeypatch.setattr("qca.sweep.threshold.qca_bin", forbidden)
    with pytest.raises(ValueError, match="__tsqca_Y__.*reserved"):
        prepare_threshold_data(
            data,
            outcome="Y",
            conditions=[condition],
            case_id=case_id,
            outcome_threshold=0.5,
            condition_thresholds={condition: 0.5},
            pre_calibrated=[condition] if pre else None,
            negate_outcome=negate,
        )
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("path", ["outcome", "condition", "multi_condition", "dual"])
@pytest.mark.parametrize("role", ["condition", "case_id"])
@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("negate", [False, True])
def test_all_sweep_paths_report_collision_without_solution(path, role, pre, negate):
    condition = OUTCOME_COLUMN if role == "condition" else "A"
    case_id = OUTCOME_COLUMN if role == "case_id" else "case"
    data = pd.DataFrame({condition: [0.1, 0.9], case_id: ["c1", "c2"], "Y": [1, 0]})
    before = data.copy(deep=True)
    sweep = ThresholdSweep(
        data, outcome="~Y" if negate else "Y", conditions=[condition], case_id=case_id
    )
    result = run(sweep, path, pre)
    row = result.summary_df.iloc[0]
    assert not row["valid"]
    assert "__tsqca_Y__" in row["error"] and "reserved" in row["error"]
    assert row["expression"] == "No solution"
    assert pd.isna(row["consistency"]) and pd.isna(row["coverage"])
    assert result.results == [None]
    assert result.details[0]["dat_bin"] is None
    assert isinstance(result.details[0]["error"], ValueError)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(sweep.data, before)


@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("source_outcome", ["Y", OUTCOME_COLUMN])
def test_source_outcome_or_unused_column_can_have_reserved_name(negate, source_outcome):
    data = pd.DataFrame({"A": [0.0, 1.0], source_outcome: [1, 0], "case": ["c1", "c2"]})
    if source_outcome != OUTCOME_COLUMN:
        data[OUTCOME_COLUMN] = [99, 98]
    before = data.copy(deep=True)
    sweep = ThresholdSweep(
        data,
        outcome=("~" if negate else "") + source_outcome,
        conditions=["A"],
        case_id="case",
    )
    result = run(sweep, "outcome")
    row = result.summary_df.iloc[0]
    assert row["valid"]
    assert row["expression"] == ("(A)" if negate else "(~A)")
    assert row["consistency"] == row["coverage"] == 1
    prepared = result.details[0]["dat_bin"]
    assert prepared[OUTCOME_COLUMN].tolist() == ([0, 1] if negate else [1, 0])
    assert prepared["case"].tolist() == ["c1", "c2"]
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("path", ["outcome", "condition", "multi_condition", "dual"])
@pytest.mark.parametrize("pre", [False, True])
def test_safe_renaming_and_condition_order_preserve_analysis(path, pre):
    data = pd.DataFrame(
        {
            "A": [0.0, 0.0, 1.0, 1.0],
            "B": [0.0, 1.0, 0.0, 1.0],
            "Y": [1, 1, 0, 0],
            "case": ["c1", "c2", "c3", "c4"],
        }
    )
    reference = run(
        ThresholdSweep(data, outcome="Y", conditions=["A", "B"], case_id="case"),
        path,
        pre,
    )
    renamed = data.rename(columns={"A": "renamed"})
    actual = run(
        ThresholdSweep(
            renamed, outcome="Y", conditions=["B", "renamed"], case_id="case"
        ),
        path,
        pre,
    )
    assert reference.summary_df.iloc[0]["expression"] == "(~A)"
    assert actual.summary_df.iloc[0]["expression"] == "(~renamed)"
    for metric in ["consistency", "coverage"]:
        assert (
            reference.summary_df.iloc[0][metric]
            == actual.summary_df.iloc[0][metric]
            == 1
        )

    def rows(result, name):
        return sorted(
            (
                tuple(
                    sorted(("A" if k == name else k, v) for k, v in r.config.items())
                ),
                r.n_cases,
                r.outcome_raw_consist,
                r.include,
            )
            for r in result.results[0].truth_table_rows
        )

    assert rows(reference, "A") == rows(actual, "renamed")
    assert actual.details[0]["dat_bin"][OUTCOME_COLUMN].tolist() == data.Y.tolist()
