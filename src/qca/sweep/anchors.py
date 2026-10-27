"""fsQCA anchor-sensitivity analysis utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from qca.calibration._validators import validate_anchor_ordering
from qca.calibration.piecewise import piecewise_fuzzy_series
from qca.core.conditions import ConditionSpec
from qca.engines import FSQCA
from qca.results import QCAFitResult
from qca.sweep.threshold import (
    _dataframe_to_markdown,
    _load_pyplot,
    _mean_or_none,
    _resolve_metric_column,
    _solution_type,
)


@dataclass(frozen=True)
class AnchorSensitivityResult:
    """Result object returned by fsQCA anchor-sensitivity sweeps."""

    sweep_type: str
    summary_df: pd.DataFrame
    settings: dict[str, Any]
    results: list[QCAFitResult | None] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def solutions(self) -> pd.DataFrame:
        """Alias for the cross-anchor summary table."""
        return self.summary_df

    @property
    def calibrated_frames(self) -> dict[str, pd.DataFrame]:
        """Calibrated data frames keyed by run label."""
        frames: dict[str, pd.DataFrame] = {}
        for detail in self.details:
            label = str(detail.get("label", detail.get("combo_id", len(frames) + 1)))
            frame = detail.get("calibrated_data")
            if frame is not None:
                frames[label] = frame
        return frames

    @property
    def stability(self) -> dict[str, Any]:
        """Cross-anchor calibration and solution stability summary."""
        if self.summary_df.empty:
            return {
                "n_runs": 0,
                "n_valid": 0,
                "valid_rate": 0.0,
                "most_common_expression": None,
                "most_common_expression_count": 0,
                "stable_expression_rate": 0.0,
                "n_unique_expressions": 0,
                "mean_consistency": None,
                "mean_coverage": None,
            }

        valid = self.summary_df["expression"] != "No solution"
        expressions = self.summary_df.loc[valid, "expression"]
        counts = expressions.value_counts()
        most_common = None if counts.empty else str(counts.index[0])
        most_common_count = 0 if counts.empty else int(counts.iloc[0])
        stable_rate = (
            0.0
            if len(self.summary_df) == 0
            else float(most_common_count / len(self.summary_df))
        )
        summary = {
            "n_runs": int(len(self.summary_df)),
            "n_valid": int(valid.sum()),
            "valid_rate": float(valid.mean()),
            "most_common_expression": most_common,
            "most_common_expression_count": most_common_count,
            "stable_expression_rate": stable_rate,
            "n_unique_expressions": int(expressions.nunique()),
            "mean_consistency": _mean_or_none(self.summary_df["consistency"]),
            "mean_coverage": _mean_or_none(self.summary_df["coverage"]),
        }
        summary.update(_anchor_ranges(self.summary_df))
        return summary

    def summary(self) -> pd.DataFrame:
        """Return the cross-anchor summary table."""
        return self.summary_df

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a Markdown stability report and optionally write it to disk."""
        lines = [
            "# fsQCA Anchor Sensitivity Result",
            "",
            "## Settings",
            "",
        ]
        for key, value in self.settings.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Calibration Stability", ""])
        for key, value in self.stability.items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Summary", ""])
        lines.append(_dataframe_to_markdown(self.summary_df))

        markdown = "\n".join(lines)
        if path is not None:
            Path(path).write_text(markdown, encoding="utf-8")
        return markdown

    def plot_heatmap(
        self,
        metric: str = "consistency",
        *,
        x: str = "crossover",
        y: str = "full_in",
    ):
        """Plot a two-anchor heatmap. Requires optional matplotlib."""
        plt = _load_pyplot()
        metric_col = _resolve_metric_column(self.summary_df, metric)
        if x not in self.summary_df.columns or y not in self.summary_df.columns:
            raise ValueError(
                f"Heatmap axes must be summary columns. Got x={x!r}, y={y!r}."
            )

        pivot = self.summary_df.pivot_table(
            index=y,
            columns=x,
            values=metric_col,
            aggfunc="mean",
        )
        fig, ax = plt.subplots()
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{self.sweep_type} {metric_col}")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        return fig

    def plot_trajectory(self, metric: str = "consistency", *, x: str | None = None):
        """Plot metric trajectories across anchor settings."""
        plt = _load_pyplot()
        metric_col = _resolve_metric_column(self.summary_df, metric)
        x_col = x or _default_trajectory_x(self.summary_df)

        fig, ax = plt.subplots()
        if self.sweep_type == "multi_anchor" and "full_out" in self.summary_df:
            for label, group in self.summary_df.groupby("full_out", sort=False):
                ax.plot(group[x_col], group[metric_col], marker="o", label=str(label))
            ax.legend(title="full_out")
        else:
            ax.plot(self.summary_df[x_col], self.summary_df[metric_col], marker="o")

        ax.set_xlabel(x_col)
        ax.set_ylabel(metric_col)
        ax.set_title(f"{self.sweep_type} {metric_col} trajectory")
        fig.tight_layout()
        return fig


class AnchorSensitivity:
    """fsQCA anchor-sensitivity facade for calibrated-set workflows."""

    def __init__(
        self,
        source: pd.DataFrame | Any,
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
            source_outcome = getattr(source, "outcome_name", None)
            if source_outcome is None:
                source_outcome = source.outcome
            self.outcome_name = outcome or source_outcome
            self.conditions = list(conditions or source.conditions)
            self.case_id = case_id if case_id is not None else source.case_id

    def full_membership(
        self,
        condition: str,
        values: Sequence[float],
        *,
        full_out: float,
        crossover: float,
        **kwargs: Any,
    ) -> AnchorSensitivityResult:
        """Sweep full-membership anchors while other anchors are fixed."""
        return self.grid(
            condition,
            {
                "full_out": [full_out],
                "crossover": [crossover],
                "full_in": values,
            },
            sweep_type="full_membership",
            **kwargs,
        )

    def crossover(
        self,
        condition: str,
        values: Sequence[float],
        *,
        full_out: float,
        full_in: float,
        **kwargs: Any,
    ) -> AnchorSensitivityResult:
        """Sweep crossover anchors while full-out/full-in are fixed."""
        return self.grid(
            condition,
            {
                "full_out": [full_out],
                "crossover": values,
                "full_in": [full_in],
            },
            sweep_type="crossover",
            **kwargs,
        )

    def full_nonmembership(
        self,
        condition: str,
        values: Sequence[float],
        *,
        crossover: float,
        full_in: float,
        **kwargs: Any,
    ) -> AnchorSensitivityResult:
        """Sweep full non-membership anchors while other anchors are fixed."""
        return self.grid(
            condition,
            {
                "full_out": values,
                "crossover": [crossover],
                "full_in": [full_in],
            },
            sweep_type="full_nonmembership",
            **kwargs,
        )

    def grid(
        self,
        condition: str,
        anchor_grid: Mapping[str, Sequence[float]],
        *,
        sweep_type: str = "multi_anchor",
        out_col: str | None = None,
        include: str = "",
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None = None,
        incl_cut: float = 0.75,
        n_cut: int = 1,
        pri_cut: float = 0.0,
        minimizer: str = "standard",
        outcome_threshold: float = 0.75,
        coverage_cutoff: float | None = None,
    ) -> AnchorSensitivityResult:
        """Run a full three-anchor grid analysis for one raw condition."""
        self._validate_variable_names(condition)
        calibrated_col = out_col or f"{condition}_fuzzy"
        self._validate_output_column(condition, calibrated_col)
        anchor_combos = _anchor_product(anchor_grid)
        model_conditions = [
            calibrated_col if name == condition else name for name in self.conditions
        ]
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        results: list[QCAFitResult | None] = []

        for combo_id, anchors in enumerate(anchor_combos, start=1):
            run = self._run_one(
                condition=condition,
                out_col=calibrated_col,
                model_conditions=model_conditions,
                anchors=anchors,
                include=include,
                dir_exp=dir_exp,
                incl_cut=incl_cut,
                n_cut=n_cut,
                pri_cut=pri_cut,
                minimizer=minimizer,
                outcome_threshold=outcome_threshold,
                coverage_cutoff=coverage_cutoff,
            )
            row = {
                "combo_id": combo_id,
                "anchor": _format_anchors(anchors),
                **anchors,
                **run["summary"],
            }
            rows.append(row)
            details.append(
                {
                    "label": _format_anchors(anchors),
                    "combo_id": combo_id,
                    "anchors": anchors,
                    **run["detail"],
                }
            )
            results.append(run["result"])

        return AnchorSensitivityResult(
            sweep_type=sweep_type,
            summary_df=pd.DataFrame(rows),
            settings={
                "mode": "anchorSensitivity",
                "sweep_type": sweep_type,
                "condition": condition,
                "out_col": calibrated_col,
                "anchor_grid": {k: list(v) for k, v in anchor_grid.items()},
                "incl.cut": incl_cut,
                "n.cut": n_cut,
                "pri.cut": pri_cut,
                "include": include,
                "dir.exp": dir_exp,
                "minimizer": minimizer,
                "outcome_threshold": outcome_threshold,
                "coverage_cutoff": coverage_cutoff,
            },
            results=results,
            details=details,
        )

    def _run_one(
        self,
        *,
        condition: str,
        out_col: str,
        model_conditions: Sequence[str],
        anchors: dict[str, float],
        include: str,
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None,
        incl_cut: float,
        n_cut: int,
        pri_cut: float,
        minimizer: str,
        outcome_threshold: float,
        coverage_cutoff: float | None,
    ) -> dict[str, Any]:
        try:
            work = self.data.copy()
            work[out_col] = piecewise_fuzzy_series(
                work[condition],
                anchors["full_out"],
                anchors["crossover"],
                anchors["full_in"],
            )
            specs = self._condition_specs(model_conditions, fuzzy_col=out_col)
            model = FSQCA(
                data=work,
                outcome=self.outcome_name,
                conditions=model_conditions,
                condition_types={spec.name: spec.kind for spec in specs},
                case_id=self.case_id if self.case_id in work.columns else None,
            )
            directional_expectations = self._resolve_directional_expectations(
                dir_exp,
                model_conditions,
            )
            fit_result = model.fit(
                minimizer=minimizer,
                consistency_cutoff=incl_cut,
                outcome_threshold=outcome_threshold,
                pri_cut=pri_cut,
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
            summary = {
                "expression": expression or "No solution",
                "inclS": consistency,
                "covS": coverage,
                "consistency": consistency,
                "coverage": coverage,
                "n_solutions": 1 if expression else 0,
                "solution_type": solution_type,
                "valid": bool(expression),
            }
            detail = {
                "truth_table": fit_result.truth_table,
                "solution": fit_result,
                "calibrated_data": work,
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
                "truth_table": None,
                "solution": None,
                "calibrated_data": None,
                "error": exc,
            }
            return {"summary": summary, "detail": detail, "result": None}

    def _condition_specs(
        self,
        model_conditions: Sequence[str],
        *,
        fuzzy_col: str,
    ) -> list[ConditionSpec]:
        specs: list[ConditionSpec] = []
        for name in model_conditions:
            kind = (
                "fuzzy"
                if name == fuzzy_col
                else _infer_set_condition_kind(self.data[name])
            )
            specs.append(
                ConditionSpec(
                    name=name,
                    kind=kind,
                    domain=(0.0, 1.0),
                    calibrated=True,
                )
            )
        return specs

    def _resolve_directional_expectations(
        self,
        dir_exp: Mapping[str, Any] | Sequence[Any] | Any | None,
        model_conditions: Sequence[str],
    ) -> dict[str, Any] | None:
        if dir_exp is None:
            return None
        if isinstance(dir_exp, Mapping):
            renamed = dict(zip(self.conditions, model_conditions, strict=True))
            resolved: dict[str, Any] = {}
            for name, value in dir_exp.items():
                target = renamed.get(name, name)
                if target in resolved and resolved[target] != value:
                    raise ValueError(
                        f"Conflicting dir_exp entries for condition {target!r}: "
                        "original and calibrated names must specify the same value."
                    )
                resolved[target] = value
            return resolved
        if isinstance(dir_exp, Sequence) and not isinstance(dir_exp, (str, bytes)):
            values = list(dir_exp)
            if len(values) == 1:
                values = values * len(model_conditions)
            if len(values) != len(model_conditions):
                raise ValueError(
                    "dir_exp sequence length must be 1 or match conditions."
                )
            return dict(zip(model_conditions, values, strict=False))
        return {condition: dir_exp for condition in model_conditions}

    def _validate_output_column(self, condition: str, output: str) -> None:
        """Protect analytical columns while allowing target recalibration."""
        if output == self.outcome_name:
            role = "the outcome column"
        elif self.case_id is not None and output == self.case_id:
            role = "the case ID column"
        elif output in self.conditions and output != condition:
            role = "a non-swept condition column"
        else:
            # The swept condition itself and existing auxiliary output columns
            # may be recalibrated, including mlQCA's previous fuzzy output.
            return
        raise ValueError(
            f"Calibrated output column {output!r} conflicts with {role}. "
            "Choose a different out_col."
        )

    def _validate_variable_names(self, condition: str) -> None:
        if self.outcome_name not in self.data.columns:
            raise ValueError(f"Outcome variable {self.outcome_name!r} not found.")
        missing = [name for name in self.conditions if name not in self.data.columns]
        if missing:
            raise ValueError(f"Condition variable(s) not found: {missing}")
        if condition not in self.conditions:
            raise ValueError("condition must be one of the model conditions.")


def _infer_set_condition_kind(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="raise").dropna().astype(float)
    if not numeric.empty and numeric.isin([0.0, 1.0]).all():
        return "crisp"
    if numeric.min(skipna=True) < 0.0 or numeric.max(skipna=True) > 1.0:
        raise ValueError(
            "Non-swept fsQCA conditions must already be calibrated to [0, 1]."
        )
    return "fuzzy"


def _anchor_product(
    anchor_grid: Mapping[str, Sequence[float]],
) -> list[dict[str, float]]:
    required = ("full_out", "crossover", "full_in")
    missing = [key for key in required if key not in anchor_grid]
    if missing:
        raise ValueError(f"anchor_grid is missing required key(s): {missing}")

    keys = list(required)
    values = [list(anchor_grid[key]) for key in keys]
    combos: list[dict[str, float]] = []
    for combo in product(*values):
        anchors = {key: float(value) for key, value in zip(keys, combo, strict=False)}
        validate_anchor_ordering(
            anchors["full_out"],
            anchors["crossover"],
            anchors["full_in"],
            param_name="anchor_grid",
        )
        combos.append(anchors)
    return combos


def _format_anchors(anchors: Mapping[str, float]) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in anchors.items())


def _anchor_ranges(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for column in ("full_out", "crossover", "full_in"):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        ranges[f"{column}_range"] = (float(values.min()), float(values.max()))
    return ranges


def _default_trajectory_x(df: pd.DataFrame) -> str:
    for column in ("full_in", "crossover", "full_out", "combo_id"):
        if column in df.columns:
            return column
    raise ValueError("summary_df does not contain a plottable x column.")


__all__ = [
    "AnchorSensitivity",
    "AnchorSensitivityResult",
]
