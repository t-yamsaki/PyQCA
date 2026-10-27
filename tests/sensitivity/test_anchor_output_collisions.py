"""Protect anchor-sweep analytical columns without blocking recalibration (#45)."""

import pandas as pd
import pytest

from qca import FSQCA, AnchorSensitivity

GRID = {"full_out": [0.0], "crossover": [1.0], "full_in": [2.0]}


def run(sweep, mode, output=None):
    kwargs = {"out_col": output, "dir_exp": {}}
    if mode == "grid":
        return sweep.grid("A", GRID, **kwargs)
    if mode == "full_membership":
        return sweep.full_membership("A", [2.0], full_out=0, crossover=1, **kwargs)
    if mode == "crossover":
        return sweep.crossover("A", [1.0], full_out=0, full_in=2, **kwargs)
    return sweep.full_nonmembership("A", [0.0], crossover=1, full_in=2, **kwargs)


@pytest.mark.parametrize(
    "mode", ["grid", "full_membership", "crossover", "full_nonmembership"]
)
@pytest.mark.parametrize("role", ["outcome", "case ID", "non-swept condition"])
@pytest.mark.parametrize("default", [False, True])
def test_protected_output_is_rejected_before_calibration(
    mode, role, default, monkeypatch
):
    collision = "A_fuzzy" if default else "protected"
    outcome = collision if role == "outcome" else "Y"
    case_id = collision if role == "case ID" else "case"
    other = collision if role == "non-swept condition" else "B"
    data = pd.DataFrame(
        {"A": [0.0, 2.0], outcome: [1, 0], case_id: ["c1", "c2"], other: [0, 1]}
    )
    before = data.copy(deep=True)
    sweep = AnchorSensitivity(
        data, outcome=outcome, conditions=["A", other], case_id=case_id
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A protected output must be rejected before calibration")

    monkeypatch.setattr("qca.sweep.anchors.piecewise_fuzzy_series", forbidden)
    with pytest.raises(ValueError, match=f"conflicts with .*{role}"):
        run(sweep, mode, None if default else collision)
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(sweep.data, before)


@pytest.mark.parametrize("role", ["outcome", "case ID"])
def test_swept_column_cannot_override_protected_role(role):
    data = pd.DataFrame({"A": [0.0, 1.0], "Y": [1, 0]})
    sweep = AnchorSensitivity(
        data,
        outcome="A" if role == "outcome" else "Y",
        conditions=["A"],
        case_id="A" if role == "case ID" else None,
    )
    with pytest.raises(ValueError, match=f"conflicts with .*{role}"):
        run(sweep, "grid", "A")


def test_engine_source_preserves_its_protected_columns():
    engine = FSQCA(
        pd.DataFrame({"A": [0.0, 1.0], "Y": [1, 0]}), outcome="Y", conditions=["A"]
    )
    before = engine.data.copy(deep=True)
    sweep = AnchorSensitivity(engine)
    for output in [engine.outcome, engine.case_id]:
        with pytest.raises(ValueError, match="conflicts with"):
            run(sweep, "grid", output)
    pd.testing.assert_frame_equal(engine.data, before)


@pytest.mark.parametrize("output", ["A", "membership", "existing"])
def test_safe_recalibration_preserves_memberships_metrics_and_input(output):
    data = pd.DataFrame(
        {"A": [0.0, 2.0], "Y": [1, 0], "case": ["c1", "c2"], "existing": [42.0, 43.0]}
    )
    before = data.copy(deep=True)
    sweep = AnchorSensitivity(data, outcome="Y", conditions=["A"], case_id="case")
    reference = run(sweep, "grid")
    result = run(sweep, "grid", output)
    assert result.results[0] is not None
    assert result.summary_df.iloc[0]["expression"] == f"(~{output})"
    for metric in ["consistency", "coverage"]:
        assert result.summary_df.iloc[0][metric] == pytest.approx(0.95)
        assert result.summary_df.iloc[0][metric] == reference.summary_df.iloc[0][metric]
    frame = result.details[0]["calibrated_data"]
    assert frame[output].tolist() == pytest.approx([0.05, 0.95])
    pd.testing.assert_frame_equal(frame[["Y", "case"]], before[["Y", "case"]])
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(sweep.data, before)
