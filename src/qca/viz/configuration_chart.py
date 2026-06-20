"""Configuration chart visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qca.viz._helpers import _require_matplotlib, _save_if_path


def configuration_matrix(
    results: Any, *, top_n: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a literal-presence matrix and the source rows used for plotting."""
    df = _normalize_results(results)
    if df.empty:
        return pd.DataFrame(), df

    term_col = _term_column(df)
    view = df.head(top_n).copy()
    parsed_terms = [_parse_term(value) for value in view[term_col]]
    literals = sorted(
        {literal for term in parsed_terms for literal in term}, key=_literal_sort_key
    )
    matrix = pd.DataFrame(0, index=range(len(view)), columns=literals, dtype=int)
    for row_index, term in enumerate(parsed_terms):
        for literal, value in term.items():
            matrix.loc[row_index, literal] = value

    label_col = term_col
    view = view.reset_index(drop=True)
    view["configuration_label"] = view[label_col].astype(str)
    return matrix, view


def plot_configuration_chart(
    results: Any,
    *,
    top_n: int = 12,
    metric: str = "consistency",
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    """Plot solution/configuration literal patterns. Requires Matplotlib."""
    plt = _require_matplotlib()
    matrix, rows = configuration_matrix(results, top_n=top_n)
    fig, ax = plt.subplots(
        figsize=(
            max(7, max(len(matrix.columns), 1) * 0.7 + 3),
            max(3.5, len(rows) * 0.45 + 2),
        )
    )
    if matrix.empty:
        ax.text(
            0.5,
            0.5,
            "No configurations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        _save_if_path(fig, path)
        return fig

    image = ax.imshow(
        matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu", vmin=-1, vmax=1
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    labels = rows["configuration_label"].astype(str).tolist()
    if metric in rows.columns:
        labels = [
            f"{label} | {metric}={_format_metric(value)}"
            for label, value in zip(labels, rows[metric], strict=False)
        ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title or "Configuration Chart")
    ax.set_xlabel("Literal")
    ax.set_ylabel("Configuration")
    cbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["negated", "absent", "present"])
    fig.tight_layout()
    _save_if_path(fig, path)
    return fig


def _normalize_results(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if hasattr(results, "solutions"):
        value = results.solutions
        if isinstance(value, pd.DataFrame):
            return value.copy()
    if hasattr(results, "summary_df"):
        value = results.summary_df
        if isinstance(value, pd.DataFrame):
            return value.copy()
    return pd.DataFrame(results)


def _term_column(df: pd.DataFrame) -> str:
    for candidate in ("term", "antecedent", "expression", "configuration"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "results must include one of: term, antecedent, expression, configuration"
    )


def _parse_term(value: Any) -> dict[str, int]:
    text = str(value).strip()
    if not text or text == "No solution":
        return {}
    text = text.replace("(", "").replace(")", "")
    parts = [part.strip() for part in text.split("*") if part.strip()]
    parsed: dict[str, int] = {}
    for part in parts:
        if part.startswith("~"):
            parsed[part[1:]] = -1
        else:
            parsed[part] = 1
    return parsed


def _literal_sort_key(value: str) -> tuple[str, int]:
    clean = value[1:] if value.startswith("~") else value
    return clean, 1 if value.startswith("~") else 0


def _format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "N/A"
    return f"{numeric:.3f}"


__all__ = [
    "configuration_matrix",
    "plot_configuration_chart",
]
