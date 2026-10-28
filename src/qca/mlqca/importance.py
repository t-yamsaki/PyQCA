"""Feature-importance extraction for XGBoost models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from qca.mlqca._features import FeatureResolver, normalize_feature_names

IMPORTANCE_COLUMNS = (
    "feature",
    "shap_mean_abs",
    "shap_mean",
    "shap_direction_score",
    "shap_direction",
    "weight",
    "gain",
    "cover",
    "total_gain",
    "total_cover",
    "split_count",
    "split_frequency",
    "shap_rank",
    "frequency_rank",
    "used",
)


def extract_xgboost_importance(
    booster: Any,
    feature_names: Sequence[str],
    *,
    contributions: Any | None = None,
    feature_values: pd.DataFrame | None = None,
    split_nodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a unified XGBoost feature-importance table.

    ``contributions`` should contain XGBoost ``pred_contribs`` output with one
    column per feature plus the bias term. If omitted, SHAP-derived columns are
    returned as missing values while native tree importance remains available.
    """
    names = normalize_feature_names(feature_names)
    resolver = FeatureResolver(booster, names)
    rows = pd.DataFrame({"feature": names})

    shap_values = _summarize_contributions(
        contributions,
        names,
        feature_values=feature_values,
    )
    rows = rows.merge(shap_values, on="feature", how="left")

    for importance_type in ("weight", "gain", "cover", "total_gain", "total_cover"):
        values = booster.get_score(importance_type=importance_type)
        normalized = {}
        for key, value in values.items():
            name = resolver.resolve(key)
            if name in normalized:
                raise ValueError(f"Multiple XGBoost feature keys resolve to {name!r}.")
            normalized[name] = float(value)
        rows[importance_type] = rows["feature"].map(normalized).fillna(0.0)

    split_counts = _split_count_table(split_nodes)
    rows = rows.merge(split_counts, on="feature", how="left")
    rows["split_count"] = rows["split_count"].fillna(rows["weight"]).astype(int)
    total_splits = int(rows["split_count"].sum())
    fallback_frequency = (
        rows["split_count"] / total_splits if total_splits else 0.0
    )
    rows["split_frequency"] = (
        rows["split_frequency"].fillna(fallback_frequency).astype(float)
    )
    rows["used"] = rows["split_count"].gt(0) | rows["weight"].gt(0)
    rows["shap_rank"] = _descending_rank(rows["shap_mean_abs"])
    rows["frequency_rank"] = _descending_rank(rows["split_count"])

    rows = rows.sort_values(
        ["shap_rank", "frequency_rank", "feature"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return rows.loc[:, list(IMPORTANCE_COLUMNS)]


def _summarize_contributions(
    contributions: Any | None,
    feature_names: tuple[str, ...],
    *,
    feature_values: pd.DataFrame | None,
) -> pd.DataFrame:
    if contributions is None:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "shap_mean_abs": np.nan,
                "shap_mean": np.nan,
                "shap_direction_score": np.nan,
                "shap_direction": None,
            }
        )

    values = np.asarray(contributions, dtype=float)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2:
        raise ValueError(
            "XGBoost contributions must be a two-dimensional array for "
            "binary classification."
        )
    expected_columns = len(feature_names) + 1
    if values.shape[1] != expected_columns:
        raise ValueError(
            "XGBoost contributions must contain one column per feature plus "
            f"the bias term. Expected {expected_columns}, got {values.shape[1]}."
        )

    contribution_values = values[:, :-1]
    direction_scores = _direction_scores(
        contribution_values,
        feature_names,
        raw_features=_feature_values_frame(
            feature_names=feature_names,
            provided=feature_values,
        ),
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "shap_mean_abs": np.mean(np.abs(contribution_values), axis=0),
            "shap_mean": np.mean(contribution_values, axis=0),
            "shap_direction_score": direction_scores,
            "shap_direction": [
                _direction_label(score) for score in direction_scores
            ],
        }
    )


def _feature_values_frame(
    *,
    feature_names: tuple[str, ...],
    provided: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Normalize raw feature values used for SHAP direction inference."""
    if provided is None:
        return None
    missing = [name for name in feature_names if name not in provided.columns]
    if missing:
        raise ValueError(f"feature_values is missing columns: {missing}")
    if len(provided) == 0:
        raise ValueError("feature_values must contain at least one row.")
    return provided.loc[:, list(feature_names)]


def _direction_scores(
    contributions: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    raw_features: pd.DataFrame | None,
) -> list[float]:
    if raw_features is None:
        return [float("nan")] * len(feature_names)
    if len(raw_features) != contributions.shape[0]:
        raise ValueError(
            "feature_values and XGBoost contributions must have the same "
            "number of rows."
        )

    scores: list[float] = []
    for index, name in enumerate(feature_names):
        values = pd.to_numeric(raw_features[name], errors="coerce").to_numpy(
            dtype=float
        )
        shap_values = contributions[:, index]
        valid = np.isfinite(values) & np.isfinite(shap_values)
        if valid.sum() < 2:
            scores.append(float("nan"))
            continue
        x = values[valid]
        y = shap_values[valid]
        x_unit = _centered_unit_vector(x)
        y_unit = _centered_unit_vector(y)
        if x_unit is None or y_unit is None:
            scores.append(float("nan"))
            continue
        scores.append(float(np.clip(np.dot(x_unit, y_unit), -1.0, 1.0)))
    return scores


def _centered_unit_vector(values: np.ndarray) -> np.ndarray | None:
    """Normalize finite values without squaring their original scale."""
    if np.all(values == values[0]):
        return None
    # Shift before scaling to preserve small representable differences around
    # a large offset. Opposite extreme values may overflow this subtraction.
    with np.errstate(over="ignore"):
        shifted = values - values[0]
    if not np.isfinite(shifted).all():
        scaled = values / np.max(np.abs(values))
        shifted = scaled - scaled[0]
    scaled = shifted / np.max(np.abs(shifted))
    centered = scaled - scaled.mean()
    return centered / np.linalg.norm(centered)


def _direction_label(score: float) -> str | None:
    if not np.isfinite(score) or np.isclose(score, 0.0):
        return None
    return "high" if score > 0.0 else "low"


def _split_count_table(split_nodes: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["feature", "split_count", "split_frequency"]
    if split_nodes is None or split_nodes.empty:
        return pd.DataFrame(columns=columns)
    if "feature" not in split_nodes:
        raise ValueError("split_nodes must contain a 'feature' column.")

    counts = (
        split_nodes.groupby("feature", sort=False)
        .size()
        .rename("split_count")
        .reset_index()
    )
    total = int(counts["split_count"].sum())
    counts["split_frequency"] = (
        counts["split_count"] / total if total else 0.0
    )
    return counts


def _descending_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        ranked = numeric.rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
        return ranked.astype("Int64")
    return pd.Series(pd.NA, index=series.index, dtype="Int64")


__all__ = ["IMPORTANCE_COLUMNS", "extract_xgboost_importance"]
