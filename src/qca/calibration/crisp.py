"""Crisp-set calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from qca.calibration._validators import (
    validate_column_exists,
    validate_numeric_column,
)

CrispDirection = Literal["ge", "gt", "le", "lt"]


@dataclass(frozen=True)
class CrispCalibrationResult:
    """Result returned by :func:`calibrate_crisp`."""

    df: pd.DataFrame
    out_col: str
    threshold: float
    direction: CrispDirection
    n_calibrated: int
    n_in: int
    n_out: int
    n_skipped: int

    def __repr__(self) -> str:
        return (
            "CrispCalibrationResult("
            f"n_calibrated={self.n_calibrated}, "
            f"n_in={self.n_in}, "
            f"n_out={self.n_out}, "
            f"n_skipped={self.n_skipped}, "
            f"out_col={self.out_col!r}"
            ")"
        )


def _normalize_direction(direction: str) -> CrispDirection:
    value = str(direction).strip().lower().replace("_", "-")
    aliases = {
        "ge": "ge",
        "gte": "ge",
        ">=": "ge",
        "at-least": "ge",
        "gt": "gt",
        ">": "gt",
        "greater-than": "gt",
        "above": "gt",
        "le": "le",
        "lte": "le",
        "<=": "le",
        "at-most": "le",
        "lt": "lt",
        "<": "lt",
        "less-than": "lt",
        "below": "lt",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported crisp calibration direction {direction!r}. "
            "Available directions: ge, gt, le, lt."
        )
    return normalized  # type: ignore[return-value]


def _validate_threshold(threshold: float) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"threshold must be numeric. Got {threshold!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"threshold must be finite. Got {threshold!r}.")
    return value


def crisp_calibrate_series(
    series: pd.Series,
    threshold: float,
    direction: str = "ge",
) -> pd.Series:
    """Convert a numeric series into crisp membership values.

    Non-null finite values are mapped to ``1.0`` or ``0.0``. Missing values are
    preserved as ``NaN`` so callers can decide whether to drop or inspect them.
    """
    threshold_value = _validate_threshold(threshold)
    normalized_direction = _normalize_direction(direction)

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    arr = numeric.to_numpy(dtype=float)
    valid = np.isfinite(arr)

    if normalized_direction == "ge":
        in_set = arr >= threshold_value
    elif normalized_direction == "gt":
        in_set = arr > threshold_value
    elif normalized_direction == "le":
        in_set = arr <= threshold_value
    else:
        in_set = arr < threshold_value

    calibrated = np.full(len(arr), np.nan, dtype=float)
    calibrated[valid] = np.where(in_set[valid], 1.0, 0.0)
    return pd.Series(calibrated, index=series.index, dtype=float)


def calibrate_crisp(
    df: pd.DataFrame,
    value_col: str,
    threshold: float,
    out_col: str | None = None,
    direction: str = "ge",
    *,
    allow_nan: bool = True,
) -> CrispCalibrationResult:
    """Calibrate a numeric column to crisp-set membership.

    Parameters
    ----------
    df:
        Source DataFrame. The original object is not modified.
    value_col:
        Numeric source column to calibrate.
    threshold:
        Cut point used by ``direction``.
    out_col:
        Output column name. Defaults to ``"{value_col}_crisp"``.
    direction:
        One of ``"ge"``, ``"gt"``, ``"le"``, or ``"lt"``.
    allow_nan:
        If ``False``, missing values in ``value_col`` raise ``ValueError``.
    """
    validate_column_exists(df, value_col)
    threshold_value = _validate_threshold(threshold)
    normalized_direction = _normalize_direction(direction)

    out_col = out_col or f"{value_col}_crisp"
    work = df.copy()
    numeric = validate_numeric_column(work, value_col, allow_nan=allow_nan)
    calibrated = crisp_calibrate_series(
        numeric,
        threshold=threshold_value,
        direction=normalized_direction,
    )

    work[value_col] = numeric
    work[out_col] = calibrated
    n_calibrated = int(calibrated.notna().sum())
    n_in = int((calibrated == 1.0).sum())
    n_out = int((calibrated == 0.0).sum())
    n_skipped = int(calibrated.isna().sum())

    return CrispCalibrationResult(
        df=work,
        out_col=out_col,
        threshold=threshold_value,
        direction=normalized_direction,
        n_calibrated=n_calibrated,
        n_in=n_in,
        n_out=n_out,
        n_skipped=n_skipped,
    )
