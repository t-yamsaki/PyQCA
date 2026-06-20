"""Split-threshold extraction for XGBoost tree ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

CUTOFF_COLUMNS = (
    "feature",
    "cutoff",
    "count",
    "frequency",
    "feature_frequency",
    "total_gain",
    "mean_gain",
    "total_cover",
    "mean_cover",
    "cutoff_rank",
)


def xgboost_split_nodes(
    booster: Any,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return normalized non-leaf nodes from an XGBoost booster."""
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature.")

    tree_df = booster.trees_to_dataframe()
    if not isinstance(tree_df, pd.DataFrame):
        raise TypeError("booster.trees_to_dataframe() must return a DataFrame.")
    required = {"Feature", "Split", "Gain", "Cover"}
    missing = sorted(required - set(tree_df.columns))
    if missing:
        raise ValueError(
            "XGBoost tree DataFrame is missing required columns: "
            f"{missing}"
        )

    nodes = tree_df.loc[tree_df["Feature"].astype(str).str.lower() != "leaf"].copy()
    if nodes.empty:
        return pd.DataFrame(columns=["feature", "cutoff", "gain", "cover"])

    key_map = {name: name for name in names}
    key_map.update({f"f{index}": name for index, name in enumerate(names)})
    nodes["feature"] = nodes["Feature"].astype(str).map(
        lambda value: key_map.get(value, value)
    )
    unknown = sorted(set(nodes["feature"]) - set(names))
    if unknown:
        raise ValueError(
            "XGBoost tree contains features not present in feature_names: "
            f"{unknown}"
        )

    nodes["cutoff"] = pd.to_numeric(nodes["Split"], errors="coerce")
    nodes["gain"] = pd.to_numeric(nodes["Gain"], errors="coerce")
    nodes["cover"] = pd.to_numeric(nodes["Cover"], errors="coerce")
    if nodes[["cutoff", "gain", "cover"]].isna().any().any():
        raise ValueError("XGBoost split nodes contain non-numeric values.")
    return nodes.loc[:, ["feature", "cutoff", "gain", "cover"]].reset_index(
        drop=True
    )


def extract_split_cutoffs(
    booster: Any,
    feature_names: Sequence[str],
    *,
    max_cutoffs_per_condition: int | None = None,
) -> pd.DataFrame:
    """Aggregate XGBoost split thresholds into calibration candidates."""
    if (
        max_cutoffs_per_condition is not None
        and (
            isinstance(max_cutoffs_per_condition, bool)
            or not isinstance(max_cutoffs_per_condition, int)
            or max_cutoffs_per_condition < 1
        )
    ):
        raise ValueError(
            "max_cutoffs_per_condition must be a positive integer or None."
        )

    nodes = xgboost_split_nodes(booster, feature_names)
    if nodes.empty:
        return pd.DataFrame(columns=list(CUTOFF_COLUMNS))

    grouped = (
        nodes.groupby(["feature", "cutoff"], sort=False)
        .agg(
            count=("cutoff", "size"),
            total_gain=("gain", "sum"),
            mean_gain=("gain", "mean"),
            total_cover=("cover", "sum"),
            mean_cover=("cover", "mean"),
        )
        .reset_index()
    )
    total_splits = int(grouped["count"].sum())
    feature_splits = grouped.groupby("feature")["count"].transform("sum")
    grouped["frequency"] = grouped["count"] / total_splits
    grouped["feature_frequency"] = grouped["count"] / feature_splits
    grouped = grouped.sort_values(
        ["feature", "count", "total_gain", "cutoff"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    grouped["cutoff_rank"] = (
        grouped.groupby("feature", sort=False).cumcount() + 1
    )
    if max_cutoffs_per_condition is not None:
        grouped = grouped.loc[
            grouped["cutoff_rank"] <= max_cutoffs_per_condition
        ]

    grouped = grouped.sort_values(
        ["cutoff_rank", "count", "total_gain", "feature"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return grouped.loc[:, list(CUTOFF_COLUMNS)]


__all__ = [
    "CUTOFF_COLUMNS",
    "extract_split_cutoffs",
    "xgboost_split_nodes",
]
