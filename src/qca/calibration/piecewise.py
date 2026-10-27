"""Piecewise-linear fuzzy-set calibration."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import (
    CALIBRATION_FULL_IN,
    CALIBRATION_FULL_OUT,
    CROSSOVER_POINT,
)
from qca.calibration._validators import (
    validate_anchor_ordering,
    validate_column_exists,
    validate_columns_exist,
    validate_min_nonnull,
    validate_numeric_column,
    validate_piecewise_membership_bounds,
    validate_quantile_triplet,
)


@dataclass(frozen=True)
class PiecewiseAnchor:
    """Anchor values estimated for one calibration group."""

    group_key: tuple[Any, ...]
    p_low: float
    p_mid: float
    p_high: float
    n_total: int
    n_nonnull: int
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "p_low": self.p_low,
            "p_mid": self.p_mid,
            "p_high": self.p_high,
            "n_total": self.n_total,
            "n_nonnull": self.n_nonnull,
            "is_valid": self.is_valid,
        }


@dataclass(frozen=True)
class PiecewiseCalibrationResult:
    """Result returned by :func:`calibrate_piecewise`."""

    df: pd.DataFrame
    anchors: list[PiecewiseAnchor]
    out_col: str
    n_calibrated: int
    n_skipped: int

    def anchors_to_df(self) -> pd.DataFrame:
        """Return anchor metadata as a DataFrame."""
        if not self.anchors:
            return pd.DataFrame()
        return pd.DataFrame([anchor.to_dict() for anchor in self.anchors])

    def __repr__(self) -> str:
        return (
            "PiecewiseCalibrationResult("
            f"n_calibrated={self.n_calibrated}, "
            f"n_skipped={self.n_skipped}, "
            f"out_col={self.out_col!r}, "
            f"n_groups={len(self.anchors)}"
            ")"
        )


def piecewise_fuzzy_scalar(
    x: float,
    p_low: float,
    p_mid: float,
    p_high: float,
    lower: float = CALIBRATION_FULL_OUT,
    upper: float = CALIBRATION_FULL_IN,
) -> float:
    """Convert one numeric value into fuzzy membership with linear segments."""
    validate_piecewise_membership_bounds(lower, upper)
    validate_anchor_ordering(p_low, p_mid, p_high)
    try:
        x_value = float(x)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(x_value):
        return np.nan

    if x_value <= p_low:
        return lower
    if x_value <= p_mid:
        return _interpolate_segment(x_value, p_low, p_mid, lower, CROSSOVER_POINT)
    if x_value <= p_high:
        return _interpolate_segment(x_value, p_mid, p_high, CROSSOVER_POINT, upper)
    return upper


def piecewise_fuzzy_series(
    series: pd.Series,
    p_low: float,
    p_mid: float,
    p_high: float,
    lower: float = CALIBRATION_FULL_OUT,
    upper: float = CALIBRATION_FULL_IN,
) -> pd.Series:
    """Convert a Series into fuzzy membership with linear segments."""
    validate_piecewise_membership_bounds(lower, upper)
    validate_anchor_ordering(p_low, p_mid, p_high)
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    arr = numeric.to_numpy(dtype=float)

    result = np.full(arr.shape, np.nan, dtype=float)
    result[arr <= p_low] = lower
    result[arr >= p_high] = upper
    low_mask = (arr > p_low) & (arr <= p_mid)
    high_mask = (arr > p_mid) & (arr < p_high)
    result[low_mask] = _interpolate_segment(
        arr[low_mask], p_low, p_mid, lower, CROSSOVER_POINT
    )
    result[high_mask] = _interpolate_segment(
        arr[high_mask], p_mid, p_high, CROSSOVER_POINT, upper
    )
    return pd.Series(result, index=series.index, dtype=float)


def _interpolate_segment(
    values: float | np.ndarray,
    left: float,
    right: float,
    lower: float,
    upper: float,
) -> float | np.ndarray:
    """Interpolate inside a valid segment without an absolute width floor."""
    left, right = float(left), float(right)
    width = right - left
    if np.isfinite(width):
        fraction = (values - left) / width
    else:
        # Opposite-sign finite anchors can have an unrepresentable difference.
        # Halving first keeps both numerator and denominator representable.
        fraction = (values / 2 - left / 2) / (right / 2 - left / 2)
    return lower + fraction * (upper - lower)


def _compute_quantile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def _estimate_anchors(
    series: pd.Series,
    quantiles: tuple[float, float, float],
    min_nonnull: int,
) -> tuple[float, float, float, bool]:
    """Estimate low/mid/high anchors for a numeric series."""
    n_nonnull = int(series.notna().sum())
    if n_nonnull < min_nonnull:
        return np.nan, np.nan, np.nan, False

    q_low, q_mid, q_high = quantiles
    p_low = _compute_quantile(series, q_low)
    p_mid = _compute_quantile(series, q_mid)
    p_high = _compute_quantile(series, q_high)

    if not all(np.isfinite(value) for value in (p_low, p_mid, p_high)):
        return np.nan, np.nan, np.nan, False
    if not p_low < p_mid < p_high:
        warnings.warn(
            "Estimated anchors do not satisfy p_low < p_mid < p_high; "
            "this group will be skipped.",
            UserWarning,
            stacklevel=3,
        )
        return np.nan, np.nan, np.nan, False
    return p_low, p_mid, p_high, True


def calibrate_piecewise(
    df: pd.DataFrame,
    value_col: str,
    out_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
    lower: float = CALIBRATION_FULL_OUT,
    upper: float = CALIBRATION_FULL_IN,
    min_nonnull: int = 5,
) -> PiecewiseCalibrationResult:
    """Calibrate a numeric column to fuzzy membership by quantile anchors."""
    validate_column_exists(df, value_col)
    if group_cols:
        validate_columns_exist(df, list(group_cols))
    validate_quantile_triplet(quantiles)
    validate_piecewise_membership_bounds(lower, upper)
    validate_min_nonnull(min_nonnull)

    out_col = out_col or f"{value_col}_fuzzy"
    work = df.copy()
    work[value_col] = validate_numeric_column(work, value_col)

    anchors: list[PiecewiseAnchor] = []
    fuzzy_values = pd.Series(np.nan, index=work.index, dtype=float)

    if not group_cols:
        p_low, p_mid, p_high, is_valid = _estimate_anchors(
            work[value_col], quantiles, min_nonnull
        )
        anchors.append(
            PiecewiseAnchor(
                group_key=(),
                p_low=p_low,
                p_mid=p_mid,
                p_high=p_high,
                n_total=len(work),
                n_nonnull=int(work[value_col].notna().sum()),
                is_valid=is_valid,
            )
        )
        if is_valid:
            fuzzy_values = piecewise_fuzzy_series(
                work[value_col], p_low, p_mid, p_high, lower, upper
            )
    else:
        group_list = list(group_cols)
        # Group positions isolate rows even when index labels are duplicated.
        for keys, positions in work.groupby(
            group_list, dropna=False, sort=True
        ).indices.items():
            group_key = keys if isinstance(keys, tuple) else (keys,)
            group_series = work[value_col].iloc[positions]
            p_low, p_mid, p_high, is_valid = _estimate_anchors(
                group_series, quantiles, min_nonnull
            )
            anchors.append(
                PiecewiseAnchor(
                    group_key=group_key,
                    p_low=p_low,
                    p_mid=p_mid,
                    p_high=p_high,
                    n_total=len(positions),
                    n_nonnull=int(group_series.notna().sum()),
                    is_valid=is_valid,
                )
            )
            if is_valid:
                fuzzy_values.iloc[positions] = piecewise_fuzzy_series(
                    group_series, p_low, p_mid, p_high, lower, upper
                ).to_numpy()

    work[out_col] = fuzzy_values
    n_calibrated = int(fuzzy_values.notna().sum())
    n_skipped = int(fuzzy_values.isna().sum())
    if n_skipped > 0:
        warnings.warn(
            f"{n_skipped} cases were not calibrated and are NaN. ",
            UserWarning,
            stacklevel=2,
        )

    return PiecewiseCalibrationResult(
        df=work,
        anchors=anchors,
        out_col=out_col,
        n_calibrated=n_calibrated,
        n_skipped=n_skipped,
    )
