"""Text report generation for PyQCA analyses."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from qca.core.results import truth_table_to_df
from qca.reporting._formatters import (
    consistency_label,
    coverage_label,
    format_key_value,
    header,
    section,
    strength_bar,
    subheader,
)


def _as_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_dataframe"):
        return value.to_dataframe().copy()
    return pd.DataFrame(value)


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table_lines(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> list[str]:
    if df.empty:
        return ["  No rows."]

    present = [col for col in columns if col in df.columns]
    if not present:
        present = list(df.columns[: min(5, len(df.columns))])
    view = df.loc[:, present].head(max_rows)
    widths = [
        max(len(str(col)), *(len(_format_metric(v)) for v in view[col]))
        for col in present
    ]

    lines = []
    header_line = "  " + "  ".join(
        str(col).ljust(width) for col, width in zip(present, widths, strict=False)
    )
    lines.append(header_line)
    lines.append("  " + "  ".join("-" * width for width in widths))
    for _, row in view.iterrows():
        lines.append(
            "  "
            + "  ".join(
                _format_metric(row[col]).ljust(width)
                for col, width in zip(present, widths, strict=False)
            )
        )
    if len(df) > max_rows:
        lines.append(f"  ... {len(df) - max_rows} more rows")
    return lines


def _resolve_fit_result(candidate: Any) -> Any | None:
    if candidate is None:
        return None
    if hasattr(candidate, "truth_table") and hasattr(candidate, "solutions"):
        return candidate
    return None


def _search_sufficiency(
    model: Any, consistency_threshold: float, coverage_threshold: float
) -> pd.DataFrame:
    if model is None or not hasattr(model, "search_sufficient_configurations"):
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.search_sufficient_configurations(
            max_depth=min(3, max(1, getattr(model, "n_conditions", 1))),
            min_consistency=consistency_threshold,
            min_coverage=coverage_threshold,
            min_cases=1,
        )


def _search_necessity(model: Any, consistency_threshold: float) -> pd.DataFrame:
    if model is None or not hasattr(model, "search_necessary_conditions"):
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.search_necessary_conditions(
            min_consistency=min(consistency_threshold, 0.9),
            min_coverage=0.0,
        )


def generate_report(
    model: Any | None = None,
    sufficiency_results: pd.DataFrame | None = None,
    necessity_results: pd.DataFrame | None = None,
    solution: Any | None = None,
    fit_result: Any | None = None,
    path: str | Path | None = None,
    consistency_threshold: float = 0.75,
    coverage_threshold: float = 0.10,
    max_rows: int = 10,
    include_minimization: bool = True,
) -> str:
    """Generate a compact text report for a PyQCA model or fit result.

    ``fit_result`` can be a ``QCAFitResult``. For convenience,
    ``generate_report(fit_result)`` is also accepted.
    """
    inferred_fit = _resolve_fit_result(model)
    if inferred_fit is not None and fit_result is None:
        fit_result = inferred_fit
        model = None
    if fit_result is None:
        fit_result = _resolve_fit_result(solution)

    if fit_result is not None and solution is None:
        solution = getattr(fit_result, "minimization", None)

    if sufficiency_results is None:
        sufficiency_df = _search_sufficiency(
            model, consistency_threshold, coverage_threshold
        )
    else:
        sufficiency_df = _as_dataframe(sufficiency_results)

    if necessity_results is None:
        necessity_df = _search_necessity(model, consistency_threshold)
    else:
        necessity_df = _as_dataframe(necessity_results)

    lines: list[str] = []
    lines.append(header("PyQCA Analysis Report"))
    lines.append(
        format_key_value(
            "engine", type(model).__name__ if model is not None else "fit result"
        )
    )
    if model is not None:
        lines.append(format_key_value("outcome", getattr(model, "outcome", "N/A")))
        lines.append(format_key_value("cases", getattr(model, "n_cases", "N/A")))
        lines.append(
            format_key_value(
                "conditions", ", ".join(getattr(model, "all_condition_names", []))
            )
        )
    lines.append(format_key_value("consistency min", f"{consistency_threshold:.2f}"))
    lines.append(format_key_value("coverage min", f"{coverage_threshold:.2f}"))

    if fit_result is not None:
        lines.append(subheader("Fit Result"))
        lines.append(
            format_key_value(
                "selected solution",
                getattr(fit_result, "selected_solution_type", "N/A"),
            )
        )
        lines.append(
            format_key_value(
                "consistency", _format_metric(getattr(fit_result, "consistency", None))
            )
        )
        lines.append(
            format_key_value(
                "coverage", _format_metric(getattr(fit_result, "coverage", None))
            )
        )
        if getattr(fit_result, "consistency", None) is not None:
            lines.append(
                "  consistency bar : " + strength_bar(float(fit_result.consistency))
            )
        solution_table = fit_result.solutions
        lines.append(section("Solution Terms"))
        lines.extend(
            _table_lines(
                solution_table,
                ["solution_type", "term", "consistency", "coverage"],
                max_rows,
            )
        )

    lines.append(subheader("Sufficiency Analysis"))
    if sufficiency_df.empty:
        lines.append("  No sufficient configurations supplied or found.")
    else:
        best = sufficiency_df.iloc[0]
        lines.append(
            format_key_value("best", best.get("antecedent", best.get("term", "N/A")))
        )
        if "consistency" in sufficiency_df.columns:
            value = float(best["consistency"])
            lines.append(
                format_key_value(
                    "consistency", f"{value:.3f} ({consistency_label(value)})"
                )
            )
        coverage_col = (
            "raw_coverage" if "raw_coverage" in sufficiency_df.columns else "coverage"
        )
        if coverage_col in sufficiency_df.columns:
            value = float(best[coverage_col])
            lines.append(
                format_key_value("coverage", f"{value:.3f} ({coverage_label(value)})")
            )
        lines.append(section("Top Configurations"))
        lines.extend(
            _table_lines(
                sufficiency_df,
                [
                    "antecedent",
                    "term",
                    "consistency",
                    "raw_coverage",
                    "coverage",
                    "n_cases_in",
                ],
                max_rows,
            )
        )

    lines.append(subheader("Necessity Analysis"))
    if necessity_df.empty:
        lines.append("  No necessary conditions supplied or found.")
    else:
        lines.extend(
            _table_lines(
                necessity_df,
                ["condition", "consistency", "coverage", "n_cases_in"],
                max_rows,
            )
        )

    if include_minimization and solution is not None and hasattr(solution, "summary"):
        lines.append(subheader("Logical Minimization"))
        summary = solution.summary()
        lines.extend(
            _table_lines(
                summary,
                ["solution_type", "term", "complexity", "n_minterms_covered"],
                max_rows,
            )
        )

    if fit_result is not None:
        truth_df = fit_result.truth_table
    elif model is not None and hasattr(model, "build_truth_table"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth_df = truth_table_to_df(model.build_truth_table())
    else:
        truth_df = pd.DataFrame()

    lines.append(subheader("Truth Table"))
    lines.extend(_table_lines(truth_df, list(truth_df.columns), max_rows))

    report = "\n".join(lines).strip() + "\n"
    if path is not None:
        Path(path).write_text(report, encoding="utf-8")
    return report
