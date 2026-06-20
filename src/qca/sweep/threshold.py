"""Threshold-sweep QCA utilities.

This module follows the ThSQCA algorithmic pattern:

1. binarize the outcome and non-pre-calibrated conditions with ``x >= threshold``;
2. run a crisp/fuzzy QCA model for each threshold setting;
3. collect expression, consistency, coverage, summary rows, and details.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qca.core.conditions import ConditionSpec
from qca.engines import MixedQCA, QCAEngineBase
from qca.results import QCAFitResult

OUTCOME_COLUMN = "__tsqca_Y__"


def qca_bin(x: pd.Series | Sequence[float], threshold: float) -> pd.Series:
    """Binarize values using the ThSQCA rule ``x >= threshold``."""
    values = pd.to_numeric(pd.Series(x), errors="raise")
    return (values >= threshold).astype(int)


def validate_pre_calibrated(
    pre_calibrated: Sequence[str] | None,
    conditions: Sequence[str],
    data: pd.DataFrame,
) -> None:
    """Validate pass-through calibrated conditions, following ThSQCA."""
    if pre_calibrated is None:
        return

    invalid = sorted(set(pre_calibrated) - set(conditions))
    if invalid:
        raise ValueError(
            f"pre_calibrated variable(s) not found in conditions: {invalid}"
        )

    for name in pre_calibrated:
        values = pd.to_numeric(data[name], errors="raise")
        if values.isna().any():
            warnings.warn(
                f"pre_calibrated variable {name!r} contains NA values.",
                UserWarning,
                stacklevel=2,
            )
        minimum = float(values.min(skipna=True))
        maximum = float(values.max(skipna=True))
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"pre_calibrated variable {name!r} has values outside [0, 1]: "
                f"[{minimum:.4f}, {maximum:.4f}]."
            )


def prepare_threshold_data(
    data: pd.DataFrame,
    *,
    outcome: str,
    conditions: Sequence[str],
    outcome_threshold: float,
    condition_thresholds: Mapping[str, float],
    case_id: str | None = None,
    pre_calibrated: Sequence[str] | None = None,
    negate_outcome: bool = False,
) -> tuple[pd.DataFrame, list[ConditionSpec]]:
    """Prepare one threshold-specific QCA data frame."""
    pre_calibrated_set = set(pre_calibrated or [])
    validate_pre_calibrated(pre_calibrated, conditions, data)

    out = pd.DataFrame(index=data.index)
    if case_id is not None and case_id in data.columns:
        out[case_id] = data[case_id]

    outcome_membership = qca_bin(data[outcome], outcome_threshold)
    out[OUTCOME_COLUMN] = (
        1 - outcome_membership if negate_outcome else outcome_membership
    )

    specs: list[ConditionSpec] = []
    for condition in conditions:
        if condition in pre_calibrated_set:
            values = pd.to_numeric(data[condition], errors="raise").astype(float)
            out[condition] = values
            kind = _infer_calibrated_condition_kind(values)
            specs.append(
                ConditionSpec(
                    condition,
                    kind,
                    domain=(0.0, 1.0),
                    calibrated=True,
                )
            )
            continue

        if condition not in condition_thresholds:
            raise ValueError(
                f"Missing threshold for condition {condition!r}. "
                "Pass condition_thresholds/thrX or list the condition in "
                "pre_calibrated."
            )
        out[condition] = qca_bin(data[condition], condition_thresholds[condition])
        specs.append(
            ConditionSpec(
                condition,
                "crisp",
                domain=(0.0, 1.0),
                calibrated=True,
            )
        )

    return out.reset_index(drop=True), specs


def _infer_calibrated_condition_kind(values: pd.Series) -> str:
    clean = values.dropna().astype(float)
    if not clean.empty and clean.isin([0.0, 1.0]).all():
        return "crisp"
    return "fuzzy"


@dataclass(frozen=True)
class ThresholdSweepResult:
    """Result object returned by ``ThresholdSweep`` methods."""

    sweep_type: str
    summary_df: pd.DataFrame
    settings: dict[str, Any]
    results: list[QCAFitResult | None] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def solutions(self) -> pd.DataFrame:
        """Alias for the cross-threshold summary table."""
        return self.summary_df

    @property
    def truth_tables(self) -> dict[str, pd.DataFrame]:
        """Truth tables keyed by threshold label."""
        tables: dict[str, pd.DataFrame] = {}
        for detail in self.details:
            label = str(detail.get("label", detail.get("combo_id", len(tables) + 1)))
            table = detail.get("truth_table")
            if table is not None:
                tables[label] = table
        return tables

    @property
    def stability(self) -> dict[str, Any]:
        """Simple cross-threshold stability summary."""
        if self.summary_df.empty:
            return {
                "n_runs": 0,
                "n_valid": 0,
                "valid_rate": 0.0,
                "most_common_expression": None,
                "most_common_expression_count": 0,
                "mean_consistency": None,
                "mean_coverage": None,
            }

        valid = self.summary_df["expression"] != "No solution"
        expressions = self.summary_df.loc[valid, "expression"]
        counts = expressions.value_counts()
        most_common = None if counts.empty else str(counts.index[0])
        most_common_count = 0 if counts.empty else int(counts.iloc[0])
        return {
            "n_runs": int(len(self.summary_df)),
            "n_valid": int(valid.sum()),
            "valid_rate": float(valid.mean()),
            "most_common_expression": most_common,
            "most_common_expression_count": most_common_count,
            "mean_consistency": _mean_or_none(self.summary_df["consistency"]),
            "mean_coverage": _mean_or_none(self.summary_df["coverage"]),
        }

    def summary(self) -> pd.DataFrame:
        """Return the cross-threshold summary table."""
        return self.summary_df

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a Markdown summary and optionally write it to disk."""
        lines = [
            "# Threshold Sweep Result",
            "",
            "## Settings",
            "",
        ]
        for key, value in self.settings.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Stability", ""])
        for key, value in self.stability.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Summary", ""])
        lines.append(_dataframe_to_markdown(self.summary_df))

        markdown = "\n".join(lines)
        if path is not None:
            Path(path).write_text(markdown, encoding="utf-8")
        return markdown

    def plot_heatmap(self, metric: str = "consistency"):
        """Plot a threshold heatmap. Requires optional matplotlib."""
        plt = _load_pyplot()
        values, x_labels, y_labels = self._heatmap_matrix(metric)
        fig, ax = plt.subplots()
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Condition threshold")
        ax.set_ylabel("Outcome threshold")
        ax.set_title(f"{self.sweep_type} {metric}")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        return fig

    def plot_trajectory(self, metric: str = "consistency"):
        """Plot metric trajectories across threshold settings."""
        plt = _load_pyplot()
        metric_col = _resolve_metric_column(self.summary_df, metric)
        fig, ax = plt.subplots()

        if "thrY" in self.summary_df.columns:
            x_col = "thrY"
        elif "threshold" in self.summary_df.columns:
            x_col = "threshold"
        else:
            x_col = "combo_id"

        if self.sweep_type == "dual" and "thrX" in self.summary_df.columns:
            for label, group in self.summary_df.groupby("thrX", sort=False):
                ax.plot(group[x_col], group[metric_col], marker="o", label=str(label))
            ax.legend(title="X thresholds")
        else:
            ax.plot(self.summary_df[x_col], self.summary_df[metric_col], marker="o")

        ax.set_xlabel(x_col)
        ax.set_ylabel(metric_col)
        ax.set_title(f"{self.sweep_type} {metric_col} trajectory")
        fig.tight_layout()
        return fig

    def _heatmap_matrix(
        self,
        metric: str,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        metric_col = _resolve_metric_column(self.summary_df, metric)
        df = self.summary_df.copy()

        if "thrY" in df.columns and "thrX" in df.columns:
            pivot = df.pivot_table(
                index="thrY",
                columns="thrX",
                values=metric_col,
                aggfunc="mean",
            )
        elif "threshold" in df.columns:
            pivot = df.set_index("threshold")[[metric_col]].T
        elif "thrY" in df.columns:
            pivot = df.set_index("thrY")[[metric_col]].T
        else:
            pivot = df.set_index("combo_id")[[metric_col]].T

        return (
            pivot.to_numpy(dtype=float),
            [str(c) for c in pivot.columns],
            [str(i) for i in pivot.index],
        )


def _mean_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = [str(c) for c in df.columns]
    rows = [
        [_format_markdown_cell(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_markdown_cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Threshold sweep plotting requires matplotlib. "
            "Install PyQCA with the 'viz' extra to use plot_heatmap() "
            "or plot_trajectory()."
        ) from exc
    return plt


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> str:
    aliases = {
        "inclS": "inclS",
        "consistency": "consistency",
        "covS": "covS",
        "coverage": "coverage",
        "valid": "valid",
    }
    column = aliases.get(metric, metric)
    if column not in df.columns:
        raise ValueError(
            f"Unknown metric {metric!r}. Available columns: {list(df.columns)}"
        )
    return column


class ThresholdSweep:
    """ThSQCA-style threshold sweep facade for PyQCA models or data frames."""

    def __init__(
        self,
        source: QCAEngineBase | pd.DataFrame,
        *,
        outcome: str | None = None,
        conditions: Sequence[str] | None = None,
        case_id: str | None = None,
    ) -> None:
        if isinstance(source, pd.DataFrame):
            if outcome is None or conditions is None:
                raise ValueError(
                    "outcome and conditions are required when source is a DataFrame."
                )
            self.data = source.copy()
            self.outcome_name = outcome
            self.conditions = list(conditions)
            self.case_id = case_id
        else:
            self.data = source.data.copy()
            self.outcome_name = outcome or source.outcome
            self.conditions = list(conditions or source.conditions)
            self.case_id = case_id if case_id is not None else source.case_id

    def outcome(
        self,
        thresholds: Sequence[float],
        *,
        condition_thresholds: Mapping[str, float] | None = None,
        thrX: Mapping[str, float] | None = None,
        pre_calibrated: Sequence[str] | None = None,
        include: str = "",
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None = None,
        incl_cut: float = 0.8,
        n_cut: int = 1,
        pri_cut: float = 0.0,
        minimizer: str = "standard",
        coverage_cutoff: float | None = None,
    ) -> ThresholdSweepResult:
        """OTS-QCA: sweep outcome thresholds while X thresholds are fixed."""
        thresholds_x = self._resolve_condition_thresholds(
            condition_thresholds or thrX,
            default=0.5,
        )
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        results: list[QCAFitResult | None] = []

        for thr_y in thresholds:
            run = self._run_one(
                thr_y=float(thr_y),
                thr_x=thresholds_x,
                include=include,
                dir_exp=dir_exp,
                incl_cut=incl_cut,
                n_cut=n_cut,
                pri_cut=pri_cut,
                minimizer=minimizer,
                coverage_cutoff=coverage_cutoff,
                pre_calibrated=pre_calibrated,
            )
            row = {"thrY": float(thr_y), **run["summary"]}
            rows.append(row)
            details.append({"label": str(thr_y), "thrY": float(thr_y), **run["detail"]})
            results.append(run["result"])

        return self._build_result(
            "outcome",
            rows,
            details,
            results,
            {
                "mode": "otSweep",
                "thresholds": list(thresholds),
                "thrX": dict(thresholds_x),
                "incl.cut": incl_cut,
                "n.cut": n_cut,
                "pri.cut": pri_cut,
                "include": include,
                "dir.exp": dir_exp,
                "minimizer": minimizer,
                "pre_calibrated": list(pre_calibrated or []),
            },
        )

    def condition(
        self,
        condition: str,
        thresholds: Sequence[float],
        *,
        outcome_threshold: float,
        condition_thresholds: Mapping[str, float] | None = None,
        default_condition_threshold: float = 0.5,
        pre_calibrated: Sequence[str] | None = None,
        include: str = "",
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None = None,
        incl_cut: float = 0.8,
        n_cut: int = 1,
        pri_cut: float = 0.0,
        minimizer: str = "standard",
        coverage_cutoff: float | None = None,
    ) -> ThresholdSweepResult:
        """CTS-QCA: sweep one condition threshold."""
        if condition not in self.conditions:
            raise ValueError("condition must be one of the model conditions.")
        base_thresholds = self._resolve_condition_thresholds(
            condition_thresholds,
            default=default_condition_threshold,
        )
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        results: list[QCAFitResult | None] = []

        for threshold in thresholds:
            thr_x = dict(base_thresholds)
            thr_x[condition] = float(threshold)
            run = self._run_one(
                thr_y=float(outcome_threshold),
                thr_x=thr_x,
                include=include,
                dir_exp=dir_exp,
                incl_cut=incl_cut,
                n_cut=n_cut,
                pri_cut=pri_cut,
                minimizer=minimizer,
                coverage_cutoff=coverage_cutoff,
                pre_calibrated=pre_calibrated,
            )
            row = {
                "threshold": float(threshold),
                f"thr_{condition}": float(threshold),
                **run["summary"],
            }
            rows.append(row)
            details.append(
                {
                    "label": str(threshold),
                    "threshold": float(threshold),
                    "thrY": float(outcome_threshold),
                    "thrX_vec": thr_x,
                    **run["detail"],
                }
            )
            results.append(run["result"])

        return self._build_result(
            "condition",
            rows,
            details,
            results,
            {
                "mode": "ctSweepS",
                "sweep_var": condition,
                "thresholds": list(thresholds),
                "thrY": outcome_threshold,
                "thrX_default": default_condition_threshold,
                "incl.cut": incl_cut,
                "n.cut": n_cut,
                "pri.cut": pri_cut,
                "include": include,
                "dir.exp": dir_exp,
                "minimizer": minimizer,
                "pre_calibrated": list(pre_calibrated or []),
            },
        )

    def multi_condition(
        self,
        threshold_grid: Mapping[str, Sequence[float]],
        *,
        outcome_threshold: float,
        pre_calibrated: Sequence[str] | None = None,
        include: str = "",
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None = None,
        incl_cut: float = 0.8,
        n_cut: int = 1,
        pri_cut: float = 0.0,
        minimizer: str = "standard",
        coverage_cutoff: float | None = None,
    ) -> ThresholdSweepResult:
        """MCTS-QCA: sweep a grid of condition thresholds."""
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        results: list[QCAFitResult | None] = []

        for combo_id, thr_x in enumerate(_threshold_product(threshold_grid), start=1):
            label = _format_thresholds(thr_x)
            run = self._run_one(
                thr_y=float(outcome_threshold),
                thr_x=thr_x,
                include=include,
                dir_exp=dir_exp,
                incl_cut=incl_cut,
                n_cut=n_cut,
                pri_cut=pri_cut,
                minimizer=minimizer,
                coverage_cutoff=coverage_cutoff,
                pre_calibrated=pre_calibrated,
            )
            row = {
                "combo_id": combo_id,
                "threshold": label,
                **{f"thr_{key}": value for key, value in thr_x.items()},
                **run["summary"],
            }
            rows.append(row)
            details.append(
                {
                    "label": label,
                    "combo_id": combo_id,
                    "thrY": float(outcome_threshold),
                    "thrX_vec": thr_x,
                    **run["detail"],
                }
            )
            results.append(run["result"])

        return self._build_result(
            "multi_condition",
            rows,
            details,
            results,
            {
                "mode": "ctSweepM",
                "sweep_list": {k: list(v) for k, v in threshold_grid.items()},
                "thrY": outcome_threshold,
                "incl.cut": incl_cut,
                "n.cut": n_cut,
                "pri.cut": pri_cut,
                "include": include,
                "dir.exp": dir_exp,
                "minimizer": minimizer,
                "pre_calibrated": list(pre_calibrated or []),
            },
        )

    def dual(
        self,
        threshold_grid: Mapping[str, Sequence[float]],
        outcome_thresholds: Sequence[float],
        *,
        pre_calibrated: Sequence[str] | None = None,
        include: str = "",
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None = None,
        incl_cut: float = 0.8,
        n_cut: int = 1,
        pri_cut: float = 0.0,
        minimizer: str = "standard",
        coverage_cutoff: float | None = None,
    ) -> ThresholdSweepResult:
        """DTS-QCA: sweep condition and outcome thresholds jointly."""
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        results: list[QCAFitResult | None] = []
        combo_id = 1

        for thr_x in _threshold_product(threshold_grid):
            label = _format_thresholds(thr_x)
            for thr_y in outcome_thresholds:
                run = self._run_one(
                    thr_y=float(thr_y),
                    thr_x=thr_x,
                    include=include,
                    dir_exp=dir_exp,
                    incl_cut=incl_cut,
                    n_cut=n_cut,
                    pri_cut=pri_cut,
                    minimizer=minimizer,
                    coverage_cutoff=coverage_cutoff,
                    pre_calibrated=pre_calibrated,
                )
                row = {
                    "combo_id": combo_id,
                    "thrY": float(thr_y),
                    "thrX": label,
                    **{f"thr_{key}": value for key, value in thr_x.items()},
                    **run["summary"],
                }
                rows.append(row)
                details.append(
                    {
                        "label": f"{label}; Y={thr_y}",
                        "combo_id": combo_id,
                        "thrY": float(thr_y),
                        "thrX_vec": thr_x,
                        **run["detail"],
                    }
                )
                results.append(run["result"])
            combo_id += 1

        return self._build_result(
            "dual",
            rows,
            details,
            results,
            {
                "mode": "dtSweep",
                "sweep_list_X": {k: list(v) for k, v in threshold_grid.items()},
                "sweep_range_Y": list(outcome_thresholds),
                "incl.cut": incl_cut,
                "n.cut": n_cut,
                "pri.cut": pri_cut,
                "include": include,
                "dir.exp": dir_exp,
                "minimizer": minimizer,
                "pre_calibrated": list(pre_calibrated or []),
            },
        )

    def fuzzy_anchors(
        self,
        condition: str,
        anchor_grid: Mapping[str, Sequence[float]],
        **kwargs: Any,
    ):
        """Run fsQCA anchor-sensitivity analysis for one raw condition."""
        from qca.sweep.anchors import AnchorSensitivity

        return AnchorSensitivity(
            self.data,
            outcome=self.outcome_name,
            conditions=self.conditions,
            case_id=self.case_id,
        ).grid(condition, anchor_grid, **kwargs)

    def _run_one(
        self,
        *,
        thr_y: float,
        thr_x: Mapping[str, float],
        include: str,
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None,
        incl_cut: float,
        n_cut: int,
        pri_cut: float,
        minimizer: str,
        coverage_cutoff: float | None,
        pre_calibrated: Sequence[str] | None,
    ) -> dict[str, Any]:
        del pri_cut
        outcome_clean, negate_outcome = self._outcome_parts()
        try:
            data_bin, specs = prepare_threshold_data(
                self.data,
                outcome=outcome_clean,
                conditions=self.conditions,
                outcome_threshold=thr_y,
                condition_thresholds=thr_x,
                case_id=self.case_id,
                pre_calibrated=pre_calibrated,
                negate_outcome=negate_outcome,
            )
            model = MixedQCA(
                data=data_bin,
                case_id=self.case_id if self.case_id in data_bin.columns else None,
                outcome=OUTCOME_COLUMN,
                condition_specs=specs,
            )
            directional_expectations = self._resolve_directional_expectations(dir_exp)
            fit_result = model.fit(
                minimizer=minimizer,
                consistency_cutoff=incl_cut,
                outcome_threshold=incl_cut,
                frequency_cutoff=n_cut,
                coverage_cutoff=coverage_cutoff,
                directional_expectations=directional_expectations,
                include_remainders_in_parsimonious=(include == "?"),
            )
            solution_type = _solution_type(include, directional_expectations)
            expression = fit_result.to_formula(solution_type)
            metric = fit_result.solution_metrics.get(solution_type)
            consistency = None if metric is None else metric.consistency
            coverage = None if metric is None else metric.raw_coverage
            n_solutions = 1 if expression else 0
            summary = {
                "expression": expression or "No solution",
                "inclS": consistency,
                "covS": coverage,
                "consistency": consistency,
                "coverage": coverage,
                "n_solutions": n_solutions,
                "solution_type": solution_type,
                "valid": bool(expression),
            }
            detail = {
                "thrX_vec": dict(thr_x),
                "truth_table": fit_result.truth_table,
                "solution": fit_result,
                "dat_bin": data_bin,
            }
            return {"summary": summary, "detail": detail, "result": fit_result}
        except Exception as exc:
            summary = {
                "expression": "No solution",
                "inclS": None,
                "covS": None,
                "consistency": None,
                "coverage": None,
                "n_solutions": 0,
                "solution_type": _solution_type(include, None),
                "valid": False,
                "error": str(exc),
            }
            detail = {
                "thrX_vec": dict(thr_x),
                "truth_table": None,
                "solution": None,
                "dat_bin": None,
                "error": exc,
            }
            return {"summary": summary, "detail": detail, "result": None}

    def _resolve_condition_thresholds(
        self,
        thresholds: Mapping[str, float] | None,
        *,
        default: float,
    ) -> dict[str, float]:
        resolved = {condition: float(default) for condition in self.conditions}
        if thresholds is not None:
            for key, value in thresholds.items():
                if key not in self.conditions:
                    raise ValueError(f"Unknown condition threshold key: {key!r}")
                resolved[key] = float(value)
        return resolved

    def _resolve_directional_expectations(
        self,
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None,
    ) -> dict[str, Any] | None:
        if dir_exp is None:
            return None
        if isinstance(dir_exp, Mapping):
            return dict(dir_exp)
        if isinstance(dir_exp, Sequence) and not isinstance(dir_exp, (str, bytes)):
            values = list(dir_exp)
            if len(values) == 1:
                values = values * len(self.conditions)
            if len(values) != len(self.conditions):
                raise ValueError(
                    "dir_exp sequence length must be 1 or match conditions."
                )
            return dict(zip(self.conditions, values, strict=False))
        return {condition: dir_exp for condition in self.conditions}

    def _outcome_parts(self) -> tuple[str, bool]:
        negate = str(self.outcome_name).startswith("~")
        clean = str(self.outcome_name)[1:] if negate else str(self.outcome_name)
        if clean not in self.data.columns:
            raise ValueError(f"Outcome variable {clean!r} not found in data.")
        missing = [
            condition for condition in self.conditions if condition not in self.data
        ]
        if missing:
            raise ValueError(f"Condition variable(s) not found in data: {missing}")
        return clean, negate

    @staticmethod
    def _build_result(
        sweep_type: str,
        rows: list[dict[str, Any]],
        details: list[dict[str, Any]],
        results: list[QCAFitResult | None],
        settings: dict[str, Any],
    ) -> ThresholdSweepResult:
        summary_df = pd.DataFrame(rows)
        return ThresholdSweepResult(
            sweep_type=sweep_type,
            summary_df=summary_df,
            settings=settings,
            results=results,
            details=details,
        )


def _solution_type(
    include: str,
    directional_expectations: Mapping[str, Any] | None,
) -> str:
    if include != "?":
        return "complex"
    if directional_expectations is None:
        return "parsimonious"
    return "intermediate"


def _threshold_product(
    threshold_grid: Mapping[str, Sequence[float]],
) -> list[dict[str, float]]:
    keys = list(threshold_grid)
    values = [list(threshold_grid[key]) for key in keys]
    return [
        {key: float(value) for key, value in zip(keys, combo, strict=False)}
        for combo in product(*values)
    ]


def _format_thresholds(thresholds: Mapping[str, float]) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in thresholds.items())


__all__ = [
    "ThresholdSweep",
    "ThresholdSweepResult",
    "prepare_threshold_data",
    "qca_bin",
    "validate_pre_calibrated",
]
