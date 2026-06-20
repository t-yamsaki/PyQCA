"""Phase 5: generic calibration package tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import qca.calibration as calibration
from qca import (
    MixedQCA,
    calibrate_crisp as top_level_calibrate_crisp,
    calibrate_logistic as top_level_calibrate_logistic,
    calibrate_piecewise as top_level_calibrate_piecewise,
)
from qca.calibration import (
    CrispCalibrationResult,
    LogisticAnchor,
    LogisticCalibrationResult,
    PiecewiseAnchor,
    PiecewiseCalibrationResult,
    calibrate_crisp,
    calibrate_logistic,
    calibrate_piecewise,
    crisp_calibrate_series,
    logistic_calibrate_series,
    piecewise_fuzzy_scalar,
    piecewise_fuzzy_series,
)
from qca.calibration._validators import (
    check_sufficient_valid_values,
    validate_anchor_ordering,
    validate_column_exists,
    validate_columns_exist,
    validate_membership_bounds,
    validate_min_nonnull,
    validate_numeric_column,
    validate_quantile_triplet,
)
from qca.calibration.logistic import _estimate_logistic_params, _logit
from qca.calibration.piecewise import _estimate_anchors


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(10)],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "group": ["A"] * 5 + ["B"] * 5,
            "outcome": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0],
            "conv": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )


class TestValidators:
    def test_column_exists_ok(self, small_df):
        validate_column_exists(small_df, "value")

    def test_column_exists_raises(self, small_df):
        with pytest.raises(KeyError, match="missing"):
            validate_column_exists(small_df, "missing")

    def test_columns_exist_raises_multiple(self, small_df):
        with pytest.raises(KeyError, match="missing1"):
            validate_columns_exist(small_df, ["value", "missing1", "missing2"])

    def test_numeric_column_ok(self, small_df):
        result = validate_numeric_column(small_df, "value")
        assert result.dtype == float

    def test_numeric_column_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        with pytest.raises(TypeError, match="numeric"):
            validate_numeric_column(df, "x")

    def test_numeric_column_nan_not_allowed(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            validate_numeric_column(df, "x", allow_nan=False)

    def test_anchor_ordering_ok(self):
        validate_anchor_ordering(1.0, 2.0, 3.0)

    def test_anchor_ordering_reversed_raises(self):
        with pytest.raises(ValueError, match="full_out < crossover < full_in"):
            validate_anchor_ordering(3.0, 2.0, 1.0)

    def test_membership_bounds_ok(self):
        validate_membership_bounds(0.05, 0.95)

    def test_membership_bounds_reversed_raises(self):
        with pytest.raises(ValueError, match="lower"):
            validate_membership_bounds(0.9, 0.5)

    def test_quantile_triplet_ok(self):
        validate_quantile_triplet((0.05, 0.5, 0.95))

    def test_quantile_triplet_wrong_order_raises(self):
        with pytest.raises(ValueError, match="low < mid < high"):
            validate_quantile_triplet((0.9, 0.5, 0.1))

    def test_min_nonnull_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            validate_min_nonnull(0)

    def test_sufficient_values_raises(self):
        with pytest.raises(ValueError, match="insufficient"):
            check_sufficient_valid_values(pd.Series([np.nan, np.inf]), "x")


class TestPiecewiseCalibration:
    def test_scalar_anchor_values(self):
        assert piecewise_fuzzy_scalar(0.5, 1.0, 2.0, 3.0) == pytest.approx(0.05)
        assert piecewise_fuzzy_scalar(2.0, 1.0, 2.0, 3.0) == pytest.approx(0.5)
        assert piecewise_fuzzy_scalar(4.0, 1.0, 2.0, 3.0) == pytest.approx(0.95)

    def test_scalar_nan_returns_nan(self):
        assert np.isnan(piecewise_fuzzy_scalar(np.nan, 1.0, 2.0, 3.0))

    def test_series_matches_scalar(self):
        series = pd.Series([0.5, 1.5, 2.0, 2.5, 3.5], index=[5, 6, 7, 8, 9])
        result = piecewise_fuzzy_series(series, 1.0, 2.0, 3.0)
        expected = [piecewise_fuzzy_scalar(x, 1.0, 2.0, 3.0) for x in series]
        assert list(result.index) == list(series.index)
        assert result.tolist() == pytest.approx(expected)

    def test_estimate_anchors_valid(self, small_df):
        p_low, p_mid, p_high, is_valid = _estimate_anchors(
            small_df["value"], (0.1, 0.5, 0.9), min_nonnull=2
        )
        assert is_valid
        assert p_low < p_mid < p_high

    def test_calibrate_piecewise_returns_result(self, small_df):
        result = calibrate_piecewise(small_df, "value", min_nonnull=2)
        assert isinstance(result, PiecewiseCalibrationResult)
        assert isinstance(result.anchors[0], PiecewiseAnchor)
        assert result.out_col == "value_fuzzy"
        assert "value_fuzzy" in result.df.columns

    def test_calibrate_piecewise_grouped(self, small_df):
        result = calibrate_piecewise(
            small_df, "value", group_cols=["group"], min_nonnull=2
        )
        assert len(result.anchors) == 2
        assert {anchor.group_key for anchor in result.anchors} == {("A",), ("B",)}

    def test_calibrate_piecewise_insufficient_group_is_nan(self):
        df = pd.DataFrame({"value": [1.0, 2.0, np.nan], "group": ["A", "A", "A"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibrate_piecewise(
                df, "value", group_cols=["group"], min_nonnull=5
            )
        assert result.df["value_fuzzy"].isna().all()
        assert result.n_skipped == len(df)

    def test_calibrate_piecewise_original_df_not_modified(self, small_df):
        original_cols = list(small_df.columns)
        calibrate_piecewise(small_df, "value", min_nonnull=2)
        assert list(small_df.columns) == original_cols

    def test_anchors_to_df(self, small_df):
        result = calibrate_piecewise(small_df, "value", min_nonnull=2)
        anchors = result.anchors_to_df()
        assert {"p_low", "p_mid", "p_high", "is_valid"}.issubset(anchors.columns)


class TestCrispCalibration:
    def test_crisp_series_ge_preserves_index_and_nan(self):
        series = pd.Series([1.0, 2.0, 3.0, np.nan], index=[10, 11, 12, 13])
        result = crisp_calibrate_series(series, threshold=2.0, direction="ge")

        assert list(result.index) == list(series.index)
        assert result.iloc[:3].tolist() == [0.0, 1.0, 1.0]
        assert np.isnan(result.iloc[3])

    def test_crisp_series_lt(self):
        result = crisp_calibrate_series(
            pd.Series([1.0, 2.0, 3.0]),
            threshold=2.0,
            direction="lt",
        )

        assert result.tolist() == [1.0, 0.0, 0.0]

    def test_calibrate_crisp_returns_result(self, small_df):
        original_cols = list(small_df.columns)
        result = calibrate_crisp(small_df, "value", threshold=5.5)

        assert isinstance(result, CrispCalibrationResult)
        assert result.out_col == "value_crisp"
        assert result.direction == "ge"
        assert result.n_calibrated == len(small_df)
        assert result.n_in == 5
        assert result.n_out == 5
        assert set(result.df["value_crisp"].unique()).issubset({0.0, 1.0})
        assert list(small_df.columns) == original_cols

    def test_calibrate_crisp_custom_direction_and_output(self, small_df):
        result = calibrate_crisp(
            small_df,
            "value",
            threshold=3.0,
            direction="le",
            out_col="low_value",
        )

        assert result.out_col == "low_value"
        assert result.n_in == 3
        assert result.df["low_value"].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]

    def test_calibrate_crisp_rejects_unknown_direction(self, small_df):
        with pytest.raises(ValueError, match="Unsupported crisp calibration"):
            calibrate_crisp(small_df, "value", threshold=5.0, direction="between")

    def test_calibrate_crisp_rejects_nan_when_requested(self):
        df = pd.DataFrame({"value": [1.0, np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            calibrate_crisp(df, "value", threshold=1.0, allow_nan=False)


class TestLogisticCalibration:
    def test_logit_half_is_zero(self):
        assert _logit(0.5) == pytest.approx(0.0)

    def test_logit_rejects_boundary(self):
        with pytest.raises(ValueError, match="exclusive"):
            _logit(1.0)

    def test_estimate_logistic_params_hits_targets(self):
        intercept, slope = _estimate_logistic_params(1.0, 2.0, 3.0)
        membership = 1.0 / (1.0 + np.exp(-(intercept + slope * 2.0)))
        assert membership == pytest.approx(0.5, abs=1e-9)

    def test_logistic_series_crossover_is_half(self):
        result, anchor = logistic_calibrate_series(pd.Series([2.0]), 1.0, 2.0, 3.0)
        assert isinstance(anchor, LogisticAnchor)
        assert result.iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_logistic_series_monotone(self):
        series = pd.Series(np.linspace(0.5, 4.0, 30), index=range(30, 60))
        result, _ = logistic_calibrate_series(series, 1.0, 2.0, 3.0)
        assert list(result.index) == list(series.index)
        assert (result.diff().dropna() >= -1e-10).all()

    def test_logistic_series_nan_preserved(self):
        result, _ = logistic_calibrate_series(
            pd.Series([1.0, np.nan, 3.0]), 0.5, 2.0, 4.0
        )
        assert np.isnan(result.iloc[1])

    def test_calibrate_logistic_quantile_anchors(self, small_df):
        result = calibrate_logistic(small_df, "value", min_nonnull=2)
        assert isinstance(result, LogisticCalibrationResult)
        assert result.anchor.anchor_quantiles == (0.05, 0.5, 0.95)
        assert result.n_calibrated == len(small_df)
        assert "value_fuzzy" in result.df.columns

    def test_calibrate_logistic_explicit_anchors(self, small_df):
        result = calibrate_logistic(
            small_df,
            "value",
            full_out=1.0,
            crossover=5.5,
            full_in=10.0,
            out_col="score_fuzzy",
            min_nonnull=2,
        )
        assert result.anchor.anchor_quantiles is None
        assert "score_fuzzy" in result.df.columns

    def test_calibrate_logistic_requires_all_explicit_anchors(self, small_df):
        with pytest.raises(ValueError, match="provided together"):
            calibrate_logistic(small_df, "value", full_out=1.0, min_nonnull=2)


class TestPublicAPISplit:
    def test_top_level_exports_generic_calibration(self):
        assert top_level_calibrate_crisp is calibrate_crisp
        assert top_level_calibrate_piecewise is calibrate_piecewise
        assert top_level_calibrate_logistic is calibrate_logistic

    def test_domain_specific_helpers_not_exported_from_package(self):
        assert not hasattr(calibration, "make_ctr_lift_fuzzy_outcome")
        assert not hasattr(calibration, "make_cv_occurred_crisp_outcome")
        assert not hasattr(calibration, "calibrate_cta_area_ratio")


class TestCalibrationToQCAPipeline:
    def test_crisp_output_valid_for_qca(self, small_df):
        result = calibrate_crisp(small_df, "value", threshold=5.5)
        df = result.df.copy()
        model = MixedQCA(
            data=df,
            case_id="case",
            conditions=["value_crisp"],
            condition_types={"value_crisp": "crisp"},
            outcome="outcome",
        )
        assert model.condition_types == {"value_crisp": "crisp"}
        assert model.n_cases == len(df)

    def test_piecewise_output_valid_for_qca(self, small_df):
        result = calibrate_piecewise(small_df, "value", min_nonnull=2)
        df = result.df.copy()
        model = MixedQCA(
            data=df,
            case_id="case",
            conditions=["value_fuzzy"],
            condition_types={"value_fuzzy": "fuzzy"},
            outcome="outcome",
        )
        assert model.n_cases == len(df)

    def test_logistic_output_valid_for_qca(self, small_df):
        result = calibrate_logistic(small_df, "value", min_nonnull=2)
        df = result.df.copy()
        model = MixedQCA(
            data=df,
            case_id="case",
            conditions=["value_fuzzy"],
            condition_types={"value_fuzzy": "fuzzy"},
            outcome="outcome",
        )
        assert model.n_cases == len(df)
