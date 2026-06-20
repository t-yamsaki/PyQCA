"""Cross-validation and bootstrap stability analysis for mlQCA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.config import MLQCAConfig
from qca.mlqca.validation import ValidatedMLQCAInput
from qca.mlqca.xgboost import XGBoostBackend

StabilityMethod = Literal["cv", "bootstrap"]


@dataclass(frozen=True)
class MLQCAStabilityResult:
    """Aggregated condition and cutoff stability across repeated fits."""

    method: StabilityMethod
    run_summary: pd.DataFrame
    feature_stability: pd.DataFrame
    cutoff_stability: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)
    runs: tuple[PredictorFitResult, ...] = ()

    def __post_init__(self) -> None:
        for name in ("run_summary", "feature_stability", "cutoff_stability"):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "settings", dict(self.settings))
        object.__setattr__(self, "runs", tuple(self.runs))

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def summary(self) -> pd.DataFrame:
        """Return feature-level stability statistics."""
        return self.feature_stability.copy()

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a compact Markdown stability report."""
        lines = [
            "# mlQCA Stability Report",
            "",
            f"- `method`: {self.method}",
            f"- `n_runs`: {self.n_runs}",
            "",
            "## Settings",
            "",
        ]
        lines.extend(f"- `{key}`: {value}" for key, value in self.settings.items())
        for title, table in (
            ("Run Summary", self.run_summary),
            ("Feature Stability", self.feature_stability),
            ("Cutoff Stability", self.cutoff_stability),
        ):
            lines.extend(["", f"## {title}", "", _markdown_table(table)])
        markdown = "\n".join(lines)
        if path is not None:
            Path(path).write_text(markdown, encoding="utf-8")
        return markdown


def cross_validate_mlqca(
    validated: ValidatedMLQCAInput,
    config: MLQCAConfig | None = None,
    *,
    n_splits: int | None = None,
    top_k: int | None = None,
    cutoff_decimals: int = 3,
) -> MLQCAStabilityResult:
    """Evaluate feature and cutoff stability with stratified cross-validation."""
    _validate_stability_input(validated)
    resolved = config or MLQCAConfig(validation="cv")
    folds = resolved.n_splits if n_splits is None else int(n_splits)
    selected_top_k = _validate_aggregation_settings(top_k, cutoff_decimals)
    if folds < 2:
        raise ValueError("n_splits must be at least 2.")
    class_counts = validated.target.value_counts()
    if folds > int(class_counts.min()):
        raise ValueError(
            "n_splits cannot exceed the number of cases in the smallest class."
        )
    sklearn_model_selection = _load_model_selection()
    splitter = sklearn_model_selection.StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=resolved.random_state,
    )
    runs: list[PredictorFitResult] = []
    run_ids: list[str] = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(validated.features, validated.target),
        start=1,
    ):
        run_config = replace(
            resolved,
            random_state=resolved.random_state + fold_index,
        )
        train = _subset_validated(validated, train_index)
        test = _subset_validated(validated, test_index)
        runs.append(XGBoostBackend(run_config).fit(train, evaluation=test))
        run_ids.append(f"fold_{fold_index:02d}")
    return _aggregate_stability(
        "cv",
        runs,
        run_ids,
        top_k=resolved.top_k if selected_top_k is None else selected_top_k,
        cutoff_decimals=cutoff_decimals,
        settings={
            "n_splits": folds,
            "shuffle": True,
            "random_state": resolved.random_state,
        },
    )


def bootstrap_mlqca(
    validated: ValidatedMLQCAInput,
    config: MLQCAConfig | None = None,
    *,
    n_bootstrap: int | None = None,
    top_k: int | None = None,
    cutoff_decimals: int = 3,
) -> MLQCAStabilityResult:
    """Evaluate feature and cutoff stability with stratified bootstrap fits."""
    _validate_stability_input(validated)
    resolved = config or MLQCAConfig(validation="bootstrap")
    n_runs = resolved.n_bootstrap if n_bootstrap is None else int(n_bootstrap)
    selected_top_k = _validate_aggregation_settings(top_k, cutoff_decimals)
    if n_runs < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    rng = np.random.default_rng(resolved.random_state)
    target = validated.target.to_numpy(dtype=int)
    class_indices = {
        value: np.flatnonzero(target == value) for value in (0, 1)
    }
    runs: list[PredictorFitResult] = []
    run_ids: list[str] = []
    all_indices = np.arange(validated.n_cases)
    for run_index in range(1, n_runs + 1):
        sampled_parts = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices.values()
        ]
        sampled = np.concatenate(sampled_parts)
        rng.shuffle(sampled)
        out_of_bag = np.setdiff1d(all_indices, np.unique(sampled))
        train = _subset_validated(validated, sampled)
        evaluation = (
            _subset_validated(validated, out_of_bag)
            if _contains_both_classes(target[out_of_bag])
            else None
        )
        run_config = replace(
            resolved,
            random_state=resolved.random_state + run_index,
        )
        runs.append(
            XGBoostBackend(run_config).fit(train, evaluation=evaluation)
        )
        run_ids.append(f"bootstrap_{run_index:03d}")
    return _aggregate_stability(
        "bootstrap",
        runs,
        run_ids,
        top_k=resolved.top_k if selected_top_k is None else selected_top_k,
        cutoff_decimals=cutoff_decimals,
        settings={
            "n_bootstrap": n_runs,
            "stratified": True,
            "random_state": resolved.random_state,
        },
    )


def _aggregate_stability(
    method: StabilityMethod,
    runs: list[PredictorFitResult],
    run_ids: list[str],
    *,
    top_k: int,
    cutoff_decimals: int,
    settings: dict[str, Any],
) -> MLQCAStabilityResult:
    feature_rows: list[pd.DataFrame] = []
    cutoff_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for run_id, run in zip(run_ids, runs, strict=True):
        importance = run.feature_importance.copy()
        importance["run_id"] = run_id
        importance["selected_top_k"] = False
        importance.loc[importance.index[:top_k], "selected_top_k"] = True
        feature_rows.append(importance)

        cutoffs = run.cutoff_candidates.copy()
        if not cutoffs.empty:
            cutoffs["run_id"] = run_id
            cutoffs["cutoff_rounded"] = cutoffs["cutoff"].round(cutoff_decimals)
            cutoff_rows.append(cutoffs)
        summary_rows.append(
            {
                "run_id": run_id,
                "evaluation_scope": run.settings.get("evaluation_scope"),
                "n_train": run.settings.get("n_cases"),
                "n_evaluation": run.settings.get("n_evaluation_cases"),
                **run.metrics,
            }
        )

    feature_long = pd.concat(feature_rows, ignore_index=True)
    feature_stability = _aggregate_features(feature_long, len(runs))
    cutoff_stability = _aggregate_cutoffs(cutoff_rows, len(runs))
    return MLQCAStabilityResult(
        method=method,
        run_summary=pd.DataFrame(summary_rows),
        feature_stability=feature_stability,
        cutoff_stability=cutoff_stability,
        settings={
            **settings,
            "top_k": top_k,
            "cutoff_decimals": cutoff_decimals,
        },
        runs=tuple(runs),
    )


def _aggregate_features(
    feature_long: pd.DataFrame,
    n_runs: int,
) -> pd.DataFrame:
    optional_defaults = {
        "used": False,
        "shap_rank": np.nan,
        "shap_mean_abs": np.nan,
        "shap_direction_score": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in feature_long:
            feature_long[column] = default
    grouped = feature_long.groupby("feature", sort=False)
    result = grouped.agg(
        top_k_count=("selected_top_k", "sum"),
        used_count=("used", "sum"),
        rank_mean=("shap_rank", "mean"),
        rank_std=("shap_rank", "std"),
        shap_mean_abs_mean=("shap_mean_abs", "mean"),
        shap_mean_abs_std=("shap_mean_abs", "std"),
        direction_score_mean=("shap_direction_score", "mean"),
    ).reset_index()
    result["top_k_rate"] = result["top_k_count"] / n_runs
    result["used_rate"] = result["used_count"] / n_runs
    return result.sort_values(
        ["top_k_rate", "rank_mean", "feature"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _aggregate_cutoffs(
    cutoff_rows: list[pd.DataFrame],
    n_runs: int,
) -> pd.DataFrame:
    if not cutoff_rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "cutoff_rounded",
                "run_count",
                "selection_rate",
                "split_count_mean",
                "total_gain_mean",
            ]
        )
    long = pd.concat(cutoff_rows, ignore_index=True)
    if "total_gain" not in long:
        long["total_gain"] = np.nan
    grouped = long.groupby(["feature", "cutoff_rounded"], sort=False)
    result = grouped.agg(
        run_count=("run_id", "nunique"),
        split_count_mean=("count", "mean"),
        total_gain_mean=("total_gain", "mean"),
    ).reset_index()
    result["selection_rate"] = result["run_count"] / n_runs
    return result.sort_values(
        ["selection_rate", "split_count_mean", "feature", "cutoff_rounded"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _subset_validated(
    validated: ValidatedMLQCAInput,
    indices: np.ndarray,
) -> ValidatedMLQCAInput:
    data = validated.data.iloc[indices].reset_index(drop=True)
    return ValidatedMLQCAInput(
        data=data,
        outcome=validated.outcome,
        candidates=validated.candidates,
        condition_specs=validated.condition_specs,
        case_id=validated.case_id,
    )


def _validate_stability_input(validated: ValidatedMLQCAInput) -> None:
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    if validated.target.nunique() != 2:
        raise ValueError("Stability analysis requires both outcome classes.")


def _validate_aggregation_settings(
    top_k: int | None,
    cutoff_decimals: int,
) -> int | None:
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or None.")
    if (
        isinstance(cutoff_decimals, bool)
        or not isinstance(cutoff_decimals, int)
        or cutoff_decimals < 0
    ):
        raise ValueError("cutoff_decimals must be a non-negative integer.")
    return top_k


def _contains_both_classes(values: np.ndarray) -> bool:
    return len(values) > 0 and len(np.unique(values)) == 2


def _load_model_selection():
    from sklearn import model_selection

    return model_selection


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "No rows."
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


__all__ = [
    "MLQCAStabilityResult",
    "StabilityMethod",
    "bootstrap_mlqca",
    "cross_validate_mlqca",
]
