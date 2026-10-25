"""Logistic fuzzy-set calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import CALIBRATION_FULL_IN, CALIBRATION_FULL_OUT
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
    slope: float | None
    intercept: float | None
    anchor_quantiles: tuple[float, float, float] | None = None
    slope_below: float | None = None
    slope_above: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_out": self.full_out,
            "crossover": self.crossover,
            "full_in": self.full_in,
            "slope": self.slope,
            "intercept": self.intercept,
            "anchor_quantiles": self.anchor_quantiles,
            "slope_below": self.slope_below,
            "slope_above": self.slope_above,
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
    return float(np.log(p) - np.log1p(-p))


def _estimate_logistic_params(
    full_out: float,
    crossover: float,
    full_in: float,
    mem_full_out: float = CALIBRATION_FULL_OUT,
    mem_full_in: float = CALIBRATION_FULL_IN,
) -> tuple[float, float]:
    """Return legacy single-curve coefficients only when both slopes agree.

    Asymmetric curves require two slopes; use logistic_calibrate_series and
    its LogisticAnchor metadata instead.
    """
    below, above = _estimate_segment_slopes(
        full_out, crossover, full_in, mem_full_out, mem_full_in
    )
    if not np.isclose(below, above, rtol=1e-12, atol=0.0):
        raise ValueError(
            "Calibration requires two slopes; use LogisticAnchor metadata."
        )
    return -below * crossover, below


def _estimate_segment_slopes(
    full_out: float,
    crossover: float,
    full_in: float,
    mem_full_out: float,
    mem_full_in: float,
) -> tuple[float, float]:
    """Fit each log-odds segment through its anchor and the crossover."""
    validate_anchor_ordering(full_out, crossover, full_in)
    validate_strict_membership_bounds(mem_full_out, mem_full_in)
    if not 0.0 < mem_full_out < 0.5 < mem_full_in < 1.0:
        raise ValueError("Require 0 < mem_full_out < 0.5 < mem_full_in < 1.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        below = float(-_logit(mem_full_out) / np.float64(crossover - full_out))
        above = float(_logit(mem_full_in) / np.float64(full_in - crossover))
    if not all(np.isfinite(value) and value > 0.0 for value in (below, above)):
        raise ValueError("Anchors must yield finite positive segment slopes.")
    return below, above


def logistic_calibrate_series(
    series: pd.Series,
    full_out: float,
    crossover: float,
    full_in: float,
    mem_full_out: float = CALIBRATION_FULL_OUT,
    mem_full_in: float = CALIBRATION_FULL_IN,
) -> tuple[pd.Series, LogisticAnchor]:
    """Convert a Series into fuzzy membership with a logistic curve."""
    below, above = _estimate_segment_slopes(
        full_out, crossover, full_in, mem_full_out, mem_full_in
    )
    symmetric = bool(np.isclose(below, above, rtol=1e-12, atol=0.0))

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    arr = numeric.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        # Center before scaling to avoid cancellation at the crossover.
        z = np.where(arr <= crossover, below, above) * (arr - crossover)
        exp_negative_abs = np.exp(-np.abs(z))
        membership = np.where(
            z >= 0,
            1.0 / (1.0 + exp_negative_abs),
            exp_negative_abs / (1.0 + exp_negative_abs),
        )
    # Saturated tails stay inside (0, 1) without clipping valid custom anchors.
    membership = np.clip(membership, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    membership[~np.isfinite(arr)] = np.nan

    anchor = LogisticAnchor(
        full_out=full_out,
        crossover=crossover,
        full_in=full_in,
        slope=below if symmetric else None,
        intercept=-below * crossover if symmetric else None,
        anchor_quantiles=None,
        slope_below=below,
        slope_above=above,
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
    anchor = replace(
        anchor, anchor_quantiles=anchor_quantiles if n_explicit == 0 else None
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
