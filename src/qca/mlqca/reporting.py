"""Markdown reporting for mlQCA results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_mlqca_markdown_report(
    result: Any,
    *,
    path: str | Path | None = None,
    title: str = "mlQCA Analysis Report",
    max_rows: int = 20,
) -> str:
    """Generate a research-oriented Markdown report for an mlQCA result."""
    lines = [f"# {title}", "", "## Overview", ""]
    lines.extend(
        [
            f"- `models_evaluated`: {getattr(result, 'n_models', 0)}",
            f"- `valid_models`: {getattr(result, 'n_valid_models', 0)}",
            f"- `best_model_id`: {getattr(result, 'best_model_id', None)}",
        ]
    )
    best = getattr(result, "best_qca_result", None)
    if best is not None:
        lines.extend(
            [
                f"- `best_consistency`: {best.consistency}",
                f"- `best_coverage`: {best.coverage}",
                f"- `best_formula`: {best.formula}",
            ]
        )

    _append_mapping(lines, "Settings", getattr(result, "settings", {}))
    _append_mapping(
        lines,
        "Reproducibility Metadata",
        getattr(result, "metadata", {}),
    )
    for title_text, attr in (
        ("Condition Ranking", "condition_ranking"),
        ("Feature Importance", "feature_importance"),
        ("Calibration Proposals", "calibration_proposals"),
        ("Candidate Models", "candidate_models"),
        ("Pareto Frontier", "pareto_models"),
    ):
        table = getattr(result, attr, None)
        if isinstance(table, pd.DataFrame):
            _append_table(lines, title_text, table, max_rows=max_rows)

    markdown = "\n".join(lines).rstrip() + "\n"
    if path is not None:
        Path(path).write_text(markdown, encoding="utf-8")
    return markdown


def _append_mapping(
    lines: list[str],
    title: str,
    values: dict[str, Any],
) -> None:
    lines.extend(["", f"## {title}", ""])
    if not values:
        lines.append("- No values recorded")
        return
    for key, value in values.items():
        lines.append(f"- `{key}`: {_format(value)}")


def _append_table(
    lines: list[str],
    title: str,
    table: pd.DataFrame,
    *,
    max_rows: int,
) -> None:
    lines.extend(["", f"## {title}", ""])
    if table.empty:
        lines.append("No rows.")
        return
    view = table.head(max_rows)
    columns = [str(column) for column in view.columns]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    if len(table) > max_rows:
        lines.append(f"\n... {len(table) - max_rows} more rows")


def _format(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


__all__ = ["generate_mlqca_markdown_report"]
