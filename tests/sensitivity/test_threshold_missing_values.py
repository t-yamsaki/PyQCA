"""Missing values must never become observed absence or presence (#32)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from qca import ThresholdSweep
from qca.sweep import prepare_threshold_data, qca_bin


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_binarization_retains_missing_and_nonfinite_values(dtype):
    values = pd.Series(
        [np.nan, -np.inf, np.inf, 0, 0.5, 1],
        dtype=dtype,
        index=[4, 4, 2, 7, 1, 0],
        name="x",
    )
    result = qca_bin(values, 0.5)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].tolist() == [0, 1, 1]
    pd.testing.assert_index_equal(result.index, values.index)
    assert result.name == "x"


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_nonfinite_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold must be finite"):
        qca_bin([0, 1], threshold)


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("negate", [False, True])
def test_preparation_preserves_missingness_even_after_negation(pre, negate):
    data = pd.DataFrame({"A": [0, 1, np.nan, np.inf], "Y": [0, 1, np.nan, -np.inf]})
    original = data.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prepared, _ = prepare_threshold_data(
            data,
            outcome="Y",
            conditions=["A"],
            outcome_threshold=0.5,
            condition_thresholds={"A": 0.5},
            pre_calibrated=["A"] if pre else None,
            negate_outcome=negate,
        )
    assert prepared["A"].tolist()[:2] == [0, 1]
    assert prepared["__tsqca_Y__"].tolist()[:2] == ([1, 0] if negate else [0, 1])
    assert prepared.iloc[2:].isna().all().all()
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("path", ["outcome", "condition", "multi_condition", "dual"])
@pytest.mark.parametrize("column,pre", [("Y", False), ("A", False), ("A", True)])
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_every_sweep_reports_invalid_data(path, column, pre, missing):
    data = pd.DataFrame({"A": [0.0, 1.0, 1.0], "Y": [0.0, 1.0, 1.0]})
    data.loc[2, column] = missing
    sweep = ThresholdSweep(data, outcome="~Y", conditions=["A"])
    kwargs = {"pre_calibrated": ["A"] if pre else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if path == "outcome":
            result = sweep.outcome([0.5], **kwargs)
        elif path == "condition":
            result = sweep.condition("A", [0.5], outcome_threshold=0.5, **kwargs)
        elif path == "multi_condition":
            result = sweep.multi_condition(
                {"A": [0.5]}, outcome_threshold=0.5, **kwargs
            )
        else:
            result = sweep.dual({"A": [0.5]}, [0.5], **kwargs)
    row = result.summary_df.iloc[0]
    assert not row["valid"]
    assert "Missing or non-finite" in row["error"]
    assert f"'{column}': 1" in row["error"]
    assert result.results == [None]
    assert isinstance(result.details[0]["error"], ValueError)
