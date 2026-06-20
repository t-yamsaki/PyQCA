"""Markdown and LaTeX export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qca.reporting.reproducibility import collect_reproducibility_metadata


def generate_markdown_report(
    obj: Any,
    *,
    path: str | Path | None = None,
    title: str | None = None,
    include_metadata: bool = True,
    max_rows: int = 20,
) -> str:
    """Generate an automated Markdown report for a model, fit result, or sweep."""
    report_title = title or _default_title(obj)
    lines = [f"# {report_title}", ""]

    if include_metadata:
        lines.extend(["## Reproducibility Metadata", ""])
        metadata = collect_reproducibility_metadata(obj)
        for key, value in metadata.items():
            lines.append(f"- `{key}`: {_format_markdown_value(value)}")
        lines.append("")

    if hasattr(obj, "settings"):
        lines.extend(["## Settings", ""])
        for key, value in obj.settings.items():
            lines.append(f"- `{key}`: {_format_markdown_value(value)}")
        lines.append("")

    if hasattr(obj, "summary") and not isinstance(obj.summary, pd.DataFrame):
        try:
            summary_value = obj.summary()
        except TypeError:
            summary_value = None
        if isinstance(summary_value, str):
            lines.extend(["## Summary", "", summary_value, ""])

    if hasattr(obj, "stability"):
        lines.extend(["## Stability", ""])
        for key, value in obj.stability.items():
            lines.append(f"- `{key}`: {_format_markdown_value(value)}")
        lines.append("")

    for section, table in _report_tables(obj):
        if table.empty:
            continue
        lines.extend([f"## {section}", ""])
        lines.append(_dataframe_to_markdown(table.head(max_rows)))
        if len(table) > max_rows:
            lines.append(f"\n... {len(table) - max_rows} more rows")
        lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    if path is not None:
        Path(path).write_text(markdown, encoding="utf-8")
    return markdown


def to_latex_table(
    obj: Any,
    *,
    table: str = "solutions",
    path: str | Path | None = None,
    caption: str | None = None,
    label: str | None = None,
    max_rows: int | None = None,
    index: bool = False,
) -> str:
    """Export one object table as a simple LaTeX tabular environment."""
    df = _resolve_table(obj, table)
    if max_rows is not None:
        df = df.head(max_rows)
    latex = _dataframe_to_latex(df, caption=caption, label=label, index=index)
    if path is not None:
        Path(path).write_text(latex, encoding="utf-8")
    return latex


def _report_tables(obj: Any) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    for title, attr in (
        ("Solutions", "solutions"),
        ("Truth Table", "truth_table"),
        ("Case Coverage", "case_coverage"),
        ("Condition Schema", "condition_schema"),
        ("Sweep Summary", "summary_df"),
        ("Condition Ranking", "condition_ranking"),
        ("Feature Importance", "feature_importance"),
        ("Calibration Proposals", "calibration_proposals"),
        ("Candidate Models", "candidate_models"),
        ("Pareto Frontier", "pareto_models"),
    ):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, pd.DataFrame):
                tables.append((title, value.copy()))
    if hasattr(obj, "data") and not tables:
        value = obj.data
        if isinstance(value, pd.DataFrame):
            tables.append(("Data Preview", value.copy()))
    return tables


def _resolve_table(obj: Any, table: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, table):
        value = getattr(obj, table)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    if table == "summary" and hasattr(obj, "summary"):
        value = obj.summary()
        if isinstance(value, pd.DataFrame):
            return value.copy()
    available = [
        name
        for name in (
            "solutions",
            "truth_table",
            "case_coverage",
            "condition_schema",
            "summary_df",
            "condition_ranking",
            "feature_importance",
            "calibration_proposals",
            "candidate_models",
            "pareto_models",
        )
        if hasattr(obj, name)
    ]
    raise ValueError(f"Unknown table {table!r}. Available tables: {available}")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = [str(c) for c in df.columns]
    rows = [
        [_format_markdown_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _dataframe_to_latex(
    df: pd.DataFrame,
    *,
    caption: str | None,
    label: str | None,
    index: bool,
) -> str:
    view = df.reset_index() if index else df.copy()
    columns = [str(column) for column in view.columns]
    align = "l" * max(len(columns), 1)
    lines: list[str] = []
    if caption is not None or label is not None:
        lines.append("\\begin{table}")
        lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_escape_latex(column) for column in columns) + r" \\")
    lines.append("\\midrule")
    for row in view.itertuples(index=False, name=None):
        lines.append(
            " & ".join(_escape_latex(_format_latex_value(value)) for value in row)
            + r" \\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption is not None:
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    if label is not None:
        lines.append(f"\\label{{{_escape_latex(label)}}}")
    if caption is not None or label is not None:
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _format_markdown_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        return str(value).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def _format_latex_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _default_title(obj: Any) -> str:
    if hasattr(obj, "qca_type"):
        return f"{obj.qca_type} Report"
    return f"{type(obj).__name__} Report"


__all__ = [
    "generate_markdown_report",
    "to_latex_table",
]
