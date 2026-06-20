"""Logistic fuzzy-set calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import CALIBRATION_FULL_IN, CALIBRATION_FULL_OUT, ZERO_EPSILON
from qca.calibration._validators import (
    check_sufficient_valid_values,
    validate_anchor_ordering,
    validate_column_exists,
    validate_membership_bounds,
    validate_min_nonnull,
    validate_numeric_column,
    validate_quantile_triplet,
    validate_strict_membership_bounds,
)


@dataclass(frozen=True)
class LogisticAnchor:
    """Anchor and curve parameters for logistic calibration."""

    full_out: float
    crossover: float
    full_in: float
    slope: float
    intercept: float
    anchor_quantiles: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_out": self.full_out,
            "crossover": self.crossover,
            "full_in": self.full_in,
            "slope": self.slope,
            "intercept": self.intercept,
            "anchor_quantiles": self.anchor_quantiles,
        }


@dataclass(frozen=True)
class LogisticCalibrationResult:
    """Result returned by :func:`calibrate_logistic`."""

    df: pd.DataFrame
    anchor: LogisticAnchor
    out_col: str
    n_calibrated: int
    n_skipped: int

    def __repr__(self) -> str:
        return (
            "LogisticCalibrationResult("
            f"n_calibrated={self.n_calibrated}, "
            f"n_skipped={self.n_skipped}, "
            f"out_col={self.out_col!r}"
            ")"
        )


def _logit(p: float) -> float:
    """Return log(p / (1 - p))."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be between 0 and 1, exclusive. Got {p}.")
    return float(np.log(p / (1.0 - p)))


def _estimate_logistic_params(
    full_out: float,
    crossover: float,
    full_in: float,
    mem_full_out: float = CALIBRATION_FULL_OUT,
    mem_full_in: float = CALIBRATION_FULL_IN,
) -> tuple[float, float]:
    """Estimate intercept and slope from three fuzzy calibration anchors."""
    validate_anchor_ordering(full_out, crossover, full_in)
    validate_strict_membership_bounds(mem_full_out, mem_full_in)

    denominator = full_in - full_out
    if abs(denominator) < ZERO_EPSILON:
        raise ValueError("full_in - full_out is too close to zero.")

    slope = (_logit(mem_full_in) - _logit(mem_full_out)) / denominator
    intercept = -slope * crossover
    return intercept, slope


def logistic_calibrate_series(
    series: pd.Series,
    full_out: float,
    crossover: float,
    full_in: float,
    mem_full_out: float = CALIBRATION_FULL_OUT,
    mem_full_in: float = CALIBRATION_FULL_IN,
) -> tuple[pd.Series, LogisticAnchor]:
    """Convert a Series into fuzzy membership with a logistic curve."""
    intercept, slope = _estimate_logistic_params(
        full_out, crossover, full_in, mem_full_out, mem_full_in
    )

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    arr = numeric.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        z = intercept + slope * arr
        membership = 1.0 / (1.0 + np.exp(-z))
    membership = np.clip(membership, ZERO_EPSILON, 1.0 - ZERO_EPSILON)
    membership[~np.isfinite(arr)] = np.nan

    anchor = LogisticAnchor(
        full_out=full_out,
        crossover=crossover,
        full_in=full_in,
        slope=slope,
        intercept=intercept,
        anchor_quantiles=None,
    )
    return pd.Series(membership, index=series.index, dtype=float), anchor


def calibrate_logistic(
    df: pd.DataFrame,
    value_col: str,
    out_col: str | None = None,
    *,
    full_out: float | None = None,
    crossover: float | None = None,
    full_in: float | None = None,
    anchor_quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
    mem_full_out: float = CALIBRATION_FULL_OUT,
    mem_full_in: float = CALIBRATION_FULL_IN,
    min_nonnull: int = 5,
) -> LogisticCalibrationResult:
    """Calibrate a numeric column to fuzzy membership with a logistic curve.

    If explicit anchors are omitted, they are estimated from ``anchor_quantiles``
    over the finite, non-null values in ``value_col``.
    """
    validate_column_exists(df, value_col)
    validate_quantile_triplet(anchor_quantiles)
    validate_membership_bounds(mem_full_out, mem_full_in)
    validate_strict_membership_bounds(mem_full_out, mem_full_in)
    validate_min_nonnull(min_nonnull)

    explicit = (full_out, crossover, full_in)
    n_explicit = sum(value is not None for value in explicit)
    if n_explicit not in {0, 3}:
        raise ValueError("full_out, crossover, and full_in must be provided together.")

    work = df.copy()
    numeric = validate_numeric_column(work, value_col)
    valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    check_sufficient_valid_values(valid, value_col, min_count=min_nonnull)

    if n_explicit == 0:
        q_low, q_mid, q_high = anchor_quantiles
        full_out, crossover, full_in = (
            float(value) for value in valid.quantile([q_low, q_mid, q_high]).tolist()
        )
    assert full_out is not None and crossover is not None and full_in is not None
    validate_anchor_ordering(full_out, crossover, full_in)

    calibrated, anchor = logistic_calibrate_series(
        numeric.replace([np.inf, -np.inf], np.nan),
        full_out=full_out,
        crossover=crossover,
        full_in=full_in,
        mem_full_out=mem_full_out,
        mem_full_in=mem_full_in,
    )
    anchor = LogisticAnchor(
        full_out=anchor.full_out,
        crossover=anchor.crossover,
        full_in=anchor.full_in,
        slope=anchor.slope,
        intercept=anchor.intercept,
        anchor_quantiles=anchor_quantiles if n_explicit == 0 else None,
    )

    out_col = out_col or f"{value_col}_fuzzy"
    work[value_col] = numeric
    work[out_col] = calibrated
    n_calibrated = int(calibrated.notna().sum())
    n_skipped = int(calibrated.isna().sum())
    return LogisticCalibrationResult(
        df=work,
        anchor=anchor,
        out_col=out_col,
        n_calibrated=n_calibrated,
        n_skipped=n_skipped,
    )
