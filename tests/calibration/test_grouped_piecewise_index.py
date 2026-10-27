"""Grouped calibration uses row positions, not duplicate labels (#31)."""

import numpy as np
import pandas as pd
import pytest

from qca.calibration import calibrate_piecewise


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([0, 1, 2, 0, 1, 2], name="shared"),
        pd.Index(["a", "a", "a", "b", "b", "b"], name="within"),
        pd.Index(["x"] * 6, name="all"),
        pd.MultiIndex.from_tuples(
            [("x", 0), ("x", 0), ("y", 1)] * 2, names=["first", "second"]
        ),
    ],
)
@pytest.mark.parametrize("group_cols", [["g"], ["g", "region"]])
@pytest.mark.parametrize("minimum", [3, 5])
def test_duplicate_labels_preserve_group_memberships_and_counts(
    index, group_cols, minimum
):
    data = pd.DataFrame(
        {
            "g": ["a"] * 3 + ["b"] * 3,
            "region": ["same"] * 6,
            "x": [0, 1, 2, 100, 101, 102],
        },
        index=index,
    ).iloc[[3, 0, 4, 1, 5, 2]]
    original = data.copy(deep=True)
    kwargs = dict(group_cols=group_cols, quantiles=(0, 0.5, 1), min_nonnull=minimum)
    if minimum == 5:
        with pytest.warns(UserWarning, match="6 cases were not calibrated"):
            result = calibrate_piecewise(data, "x", **kwargs)
        with pytest.warns(UserWarning, match="6 cases were not calibrated"):
            baseline = calibrate_piecewise(data.reset_index(drop=True), "x", **kwargs)
        assert result.df.x_fuzzy.isna().all()
        assert (result.n_calibrated, result.n_skipped) == (0, 6)
        assert all(not anchor.is_valid for anchor in result.anchors)
    else:
        result = calibrate_piecewise(data, "x", **kwargs)
        baseline = calibrate_piecewise(data.reset_index(drop=True), "x", **kwargs)
        assert result.df.x_fuzzy.tolist() == pytest.approx(
            [0.05, 0.05, 0.5, 0.5, 0.95, 0.95]
        )
        assert (result.n_calibrated, result.n_skipped) == (6, 0)
        assert [(a.p_low, a.p_mid, a.p_high) for a in result.anchors] == [
            (0, 1, 2),
            (100, 101, 102),
        ]
    assert [(a.n_total, a.n_nonnull) for a in result.anchors] == [(3, 3), (3, 3)]
    pd.testing.assert_frame_equal(result.anchors_to_df(), baseline.anchors_to_df())
    pd.testing.assert_frame_equal(result.df.reset_index(drop=True), baseline.df)
    pd.testing.assert_index_equal(result.df.index, data.index)
    pd.testing.assert_frame_equal(
        result.df[data.columns], original.astype({"x": float})
    )
    pd.testing.assert_frame_equal(data, original)


def test_missing_values_count_actual_rows_with_shared_labels():
    data = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 4,
            "x": [0, 1, 2, np.nan, 100, np.nan, 102, np.nan],
        },
        index=[0] * 8,
    )
    with pytest.warns(UserWarning, match="5 cases were not calibrated"):
        result = calibrate_piecewise(
            data, "x", group_cols=["g"], quantiles=(0, 0.5, 1), min_nonnull=3
        )
    assert result.df.x_fuzzy.iloc[:3].tolist() == pytest.approx([0.05, 0.5, 0.95])
    assert result.df.x_fuzzy.iloc[3:].isna().all()
    assert [(a.n_total, a.n_nonnull, a.is_valid) for a in result.anchors] == [
        (4, 3, True),
        (4, 2, False),
    ]
    assert (result.n_calibrated, result.n_skipped) == (3, 5)


@pytest.mark.parametrize("group_cols", [["g"], ["g", "region"]])
def test_missing_group_keys_remain_a_separate_group(group_cols):
    data = pd.DataFrame(
        {
            "g": ["a", None, "a", None, "a", None],
            "region": ["same"] * 6,
            "x": [0, 100, 1, 101, 2, 102],
        },
        index=[0] * 6,
    )
    result = calibrate_piecewise(
        data, "x", group_cols=group_cols, quantiles=(0, 0.5, 1), min_nonnull=3
    )
    assert result.df.x_fuzzy.tolist() == pytest.approx(
        [0.05, 0.05, 0.5, 0.5, 0.95, 0.95]
    )
    assert [(a.n_total, a.n_nonnull) for a in result.anchors] == [(3, 3), (3, 3)]
    assert (result.n_calibrated, result.n_skipped) == (6, 0)
