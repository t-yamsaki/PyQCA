"""Jupyter-friendly summary objects."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import pandas as pd

from qca.reporting.reproducibility import collect_reproducibility_metadata


@dataclass(frozen=True)
class JupyterSummary:
    """Small display object with plain text and HTML representations."""

    title: str
    text: str
    html: str

    def _repr_html_(self) -> str:
        return self.html

    def _repr_markdown_(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


def jupyter_summary(
    obj: Any,
    *,
    title: str | None = None,
    max_rows: int = 8,
) -> JupyterSummary:
    """Build a compact summary suitable for notebooks."""
    summary_title = title or _default_title(obj)
    metadata = collect_reproducibility_metadata(obj)
    metrics = _summary_metrics(obj)
    text_lines = [f"# {summary_title}", ""]
    for key, value in metrics.items():
        text_lines.append(f"- {key}: {value}")
    text_lines.append(f"- qca_version: {metadata.get('qca_version')}")

    html_parts = [
        '<div class="pyqca-summary" style="font-family: system-ui, sans-serif;">',
        f"<h3>{escape(summary_title)}</h3>",
        '<table style="border-collapse: collapse;">',
    ]
    for key, value in metrics.items():
        html_parts.append(
            "<tr>"
            f'<th style="text-align:left;padding:2px 8px;">{escape(str(key))}</th>'
            f'<td style="padding:2px 8px;">{escape(str(value))}</td>'
            "</tr>"
        )
    html_parts.append(
        "<tr>"
        '<th style="text-align:left;padding:2px 8px;">qca_version</th>'
        f'<td style="padding:2px 8px;">{escape(str(metadata.get("qca_version")))}</td>'
        "</tr>"
    )
    html_parts.append("</table>")

    table = _primary_table(obj)
    if table is not None and not table.empty:
        html_parts.append("<h4>Preview</h4>")
        html_parts.append(table.head(max_rows).to_html(index=False, escape=True))
    html_parts.append("</div>")

    return JupyterSummary(
        title=summary_title,
        text="\n".join(text_lines),
        html="\n".join(html_parts),
    )


def _summary_metrics(obj: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "object_type": type(obj).__name__,
    }
    for attr in (
        "qca_type",
        "outcome",
        "workflow",
        "selected_solution_type",
        "consistency",
        "coverage",
    ):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, float):
                value = f"{value:.4f}"
            metrics[attr] = value
    if hasattr(obj, "solutions"):
        solutions = obj.solutions
        if isinstance(solutions, pd.DataFrame):
            metrics["n_solution_rows"] = len(solutions)
    if hasattr(obj, "truth_table"):
        truth_table = obj.truth_table
        if isinstance(truth_table, pd.DataFrame):
            metrics["n_truth_table_rows"] = len(truth_table)
    if hasattr(obj, "summary_df"):
        summary_df = obj.summary_df
        if isinstance(summary_df, pd.DataFrame):
            metrics["n_summary_rows"] = len(summary_df)
    return metrics


def _primary_table(obj: Any) -> pd.DataFrame | None:
    for attr in ("solutions", "summary_df", "truth_table", "condition_schema"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, pd.DataFrame):
                return value
    return None


def _default_title(obj: Any) -> str:
    if hasattr(obj, "qca_type"):
        return f"{obj.qca_type} Summary"
    return f"{type(obj).__name__} Summary"


__all__ = [
    "JupyterSummary",
    "jupyter_summary",
]
