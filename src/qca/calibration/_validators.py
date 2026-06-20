"""Validation helpers for calibration routines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_column_exists(df: pd.DataFrame, col: str) -> None:
    """Raise KeyError when a DataFrame column is missing."""
    if col not in df.columns:
        raise KeyError(
            f"Column {col!r} does not exist. Available columns: {list(df.columns)}"
        )


def validate_columns_exist(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise KeyError when any DataFrame column is missing."""
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"Columns do not exist: {missing}. Available columns: {list(df.columns)}"
        )


def validate_numeric_column(
    df: pd.DataFrame,
    col: str,
    allow_nan: bool = True,
) -> pd.Series:
    """Return a column converted to float values."""
    validate_column_exists(df, col)
    try:
        series = pd.to_numeric(df[col], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Column {col!r} cannot be converted to numeric values: {exc}"
        ) from exc

    if not allow_nan and series.isna().any():
        n_nan = int(series.isna().sum())
        raise ValueError(f"Column {col!r} contains {n_nan} NaN values.")
    return series


def validate_anchor_ordering(
    full_out: float,
    crossover: float,
    full_in: float,
    *,
    param_name: str = "anchors",
) -> None:
    """Validate direct-calibration anchor ordering."""
    values = (full_out, crossover, full_in)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{param_name} must contain finite values: {values}")
    if not (full_out < crossover < full_in):
        raise ValueError(
            f"{param_name} must satisfy full_out < crossover < full_in. "
            f"Got full_out={full_out}, crossover={crossover}, full_in={full_in}."
        )


def validate_membership_bounds(lower: float, upper: float) -> None:
    """Validate lower and upper membership bounds."""
    if lower < 0.0:
        raise ValueError(f"lower must be >= 0.0. Got {lower}.")
    if upper > 1.0:
        raise ValueError(f"upper must be <= 1.0. Got {upper}.")
    if lower >= upper:
        raise ValueError(f"lower must be less than upper. Got {lower} >= {upper}.")


def validate_strict_membership_bounds(lower: float, upper: float) -> None:
    """Validate bounds suitable for logit transformation."""
    validate_membership_bounds(lower, upper)
    if lower <= 0.0 or upper >= 1.0:
        raise ValueError(
            "logistic calibration requires 0.0 < mem_full_out < mem_full_in < 1.0"
        )


def validate_quantile_triplet(quantiles: tuple[float, float, float]) -> None:
    """Validate a low/mid/high quantile triplet."""
    if len(quantiles) != 3:
        raise ValueError("quantiles must contain exactly three values.")
    q_low, q_mid, q_high = quantiles
    for value, name in ((q_low, "low"), (q_mid, "mid"), (q_high, "high")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Quantile {name} must be in [0, 1]. Got {value}.")
    if not q_low < q_mid < q_high:
        raise ValueError(f"quantiles must satisfy low < mid < high. Got {quantiles}.")


def validate_min_nonnull(min_nonnull: int) -> None:
    """Validate the minimum non-null count for anchor estimation."""
    if min_nonnull < 1:
        raise ValueError(f"min_nonnull must be >= 1. Got {min_nonnull}.")


def check_sufficient_valid_values(
    series: pd.Series,
    col: str,
    min_count: int = 1,
) -> None:
    """Validate that a series has enough finite non-null values."""
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < min_count:
        raise ValueError(
            f"Column {col!r} has insufficient valid values for calibration. "
            f"Required {min_count}, got {len(valid)}."
        )
