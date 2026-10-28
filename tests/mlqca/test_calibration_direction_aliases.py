"""Accepted direction spellings must yield identical sets and QCA results."""

import numpy as np
import pandas as pd
import pytest

from qca import (
    calibrate_crisp_conditions,
    calibrate_fuzzy_conditions,
    fit_csqca_from_calibration,
    fit_fsqca_from_calibration,
    validate_mlqca_input,
)

ALIASES = [
    ("high", "high"),
    ("positive", "high"),
    ("ge", "high"),
    (">=", "high"),
    ("low", "low"),
    ("negative", "low"),
    ("le", "low"),
    ("<=", "low"),
]


def inputs(fuzzy, direction):
    data = pd.DataFrame(
        {"case": list("abcde"), "A": [0.0, 0.5, 1.0, 1.5, 2.0], "Y": [0, 0, 1, 1, 1]}
    )
    validated = validate_mlqca_input(data, "Y", case_id="case")
    proposal = {
        "feature": "A",
        "output": "X",
        "direction": direction,
        "note": "user metadata",
    }
    if fuzzy:
        proposal.update(full_out=0.0, crossover=1.0, full_in=2.0)
    else:
        proposal.update(threshold=1.0, operator="stale")
    return validated, pd.DataFrame([proposal], index=[17])


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("alias,canonical", ALIASES)
@pytest.mark.parametrize("style", ["plain", "upper", "padded"])
def test_aliases_preserve_membership_metadata_and_fit(fuzzy, alias, canonical, style):
    spelling = alias if style == "plain" else alias.upper()
    if style == "padded":
        spelling = " \t" + spelling + " \n"
    validated, proposal = inputs(fuzzy, spelling)
    before = proposal.copy(deep=True)
    data_before = validated.data.copy(deep=True)
    calibrate = calibrate_fuzzy_conditions if fuzzy else calibrate_crisp_conditions
    fit = fit_fsqca_from_calibration if fuzzy else fit_csqca_from_calibration
    actual = calibrate(validated, proposal)
    reference = calibrate(validated, proposal.assign(direction=canonical))
    pd.testing.assert_frame_equal(actual.data, reference.data)
    pd.testing.assert_frame_equal(actual.proposals, reference.proposals)
    assert actual.proposals.loc[17, "direction"] == canonical
    assert actual.proposals.loc[17, "note"] == "user metadata"
    if fuzzy:
        assert actual.data["X"].iloc[[0, 2, 4]].tolist() == pytest.approx(
            [0.05, 0.5, 0.95] if canonical == "high" else [0.95, 0.5, 0.05]
        )
    else:
        assert actual.data["X"].tolist() == (
            [0, 0, 1, 1, 1] if canonical == "high" else [1, 1, 1, 0, 0]
        )
        assert actual.proposals.loc[17, "operator"] == (
            ">=" if canonical == "high" else "<="
        )
    options = dict(
        directional_expectations={}, consistency_cutoff=0.5, outcome_threshold=0.5
    )
    actual_fit, reference_fit = (
        fit(actual, **options).fit_result,
        fit(reference, **options).fit_result,
    )
    assert actual_fit.formulas == reference_fit.formulas
    pd.testing.assert_frame_equal(actual_fit.truth_table, reference_fit.truth_table)
    assert actual_fit.consistency == reference_fit.consistency
    assert actual_fit.coverage == reference_fit.coverage
    pd.testing.assert_frame_equal(proposal, before)
    pd.testing.assert_frame_equal(validated.data, data_before)


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize(
    "direction", ["", "up", "down", "infer", None, np.nan, 1, pd.NA]
)
def test_invalid_directions_are_rejected_without_mutation(fuzzy, direction):
    validated, proposal = inputs(fuzzy, direction)
    before = proposal.copy(deep=True)
    calibrate = calibrate_fuzzy_conditions if fuzzy else calibrate_crisp_conditions
    with pytest.raises(ValueError, match="Unknown crisp calibration direction"):
        calibrate(validated, proposal)
    pd.testing.assert_frame_equal(proposal, before)


@pytest.mark.parametrize("fuzzy", [False, True])
def test_categorical_proposal_directions_are_normalized_on_a_copy(fuzzy):
    validated, proposal = inputs(fuzzy, " LOW ")
    proposal["direction"] = proposal["direction"].astype("category")
    before = proposal.copy(deep=True)
    calibrate = calibrate_fuzzy_conditions if fuzzy else calibrate_crisp_conditions
    actual = calibrate(validated, proposal)
    assert actual.proposals.loc[17, "direction"] == "low"
    assert actual.data["X"].iloc[0] > actual.data["X"].iloc[-1]
    pd.testing.assert_frame_equal(proposal, before)
