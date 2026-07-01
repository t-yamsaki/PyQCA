"""Dashboard visualization composed from PyQCA plot helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from qca.core.literals import parse_conjunction
from qca.viz._helpers import _require_matplotlib, _save_if_path, apply_grid_style
from qca.viz.xy_plot import plot_xy


def _parse_best_literals(
    results: pd.DataFrame, model: Any | None = None
) -> list[Any] | None:
    """Parse the first sufficient configuration into literals when possible."""
    if results is None or results.empty:
        return None
    row = results.iloc[0]
    label = row.get("antecedent", row.get("term", None))
    if not isinstance(label, str) or not label or label == "(tautology)":
        return None
    try:
        literals = parse_conjunction(label.replace(" + ", " * "))
    except Exception:
        return None
    if model is not None:
        model_cols = set(model.data.columns)
        if any(literal.name not in model_cols for literal in literals):
            return None
    return literals


def _search_results(model: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sufficiency = model.search_sufficient_configurations(
            max_depth=min(3, max(1, model.n_conditions)),
            min_consistency=0.5,
            min_coverage=0.0,
            min_cases=1,
        )
        necessity = model.search_necessary_conditions(
            min_consistency=0.5, min_coverage=0.0
        )
    return sufficiency, necessity


def _bar_summary(
    ax: Any, df: pd.DataFrame, label_col: str, value_col: str, title: str
) -> None:
    if df.empty or value_col not in df.columns:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_axis_off()
        return
    view = df.head(8).iloc[::-1]
    ax.barh(
        range(len(view)),
        view[value_col].astype(float),
        color="#3498db",
        edgecolor="white",
    )
    ax.set_yticks(range(len(view)))
    ax.set_yticklabels(view[label_col].astype(str), fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    apply_grid_style(ax, axis="x")


def plot_dashboard(
    model: Any,
    suf_results: pd.DataFrame | None = None,
    nec_results: pd.DataFrame | None = None,
    solution_terms: Any | None = None,
    outcome_threshold: float = 0.5,
    best_literals: Any | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    """Create a compact dashboard figure for a GSQCA model."""
    plt = _require_matplotlib()
    if suf_results is None or nec_results is None:
        found_suf, found_nec = _search_results(model)
        suf_results = found_suf if suf_results is None else suf_results
        nec_results = found_nec if nec_results is None else nec_results

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_meta, ax_suff, ax_nec, ax_xy = axes.flatten()

    ax_meta.axis("off")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(
            [
                f"Cases: {model.n_cases}",
                f"Outcome: {model.outcome}",
                f"Set conditions: {', '.join(model.set_conditions) or 'none'}",
                f"Multi-value: {', '.join(model.multivalue_conditions) or 'none'}",
                f"Outcome threshold: {outcome_threshold:.2f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        transform=ax_meta.transAxes,
    )
    ax_meta.set_title("Model")

    suff_label = (
        "antecedent"
        if "antecedent" in suf_results.columns
        else "term"
        if "term" in suf_results.columns
        else None
    )
    if suff_label is None:
        ax_suff.text(
            0.5,
            0.5,
            "No sufficiency labels",
            ha="center",
            va="center",
            transform=ax_suff.transAxes,
        )
        ax_suff.set_axis_off()
    else:
        _bar_summary(
            ax_suff, suf_results, suff_label, "consistency", "Sufficiency Consistency"
        )

    if "condition" in nec_results.columns:
        _bar_summary(
            ax_nec, nec_results, "condition", "consistency", "Necessity Consistency"
        )
    else:
        ax_nec.text(
            0.5,
            0.5,
            "No necessity labels",
            ha="center",
            va="center",
            transform=ax_nec.transAxes,
        )
        ax_nec.set_axis_off()

    if best_literals is None:
        best_literals = _parse_best_literals(suf_results, model)
    if best_literals is not None:
        plot_xy(
            model,
            best_literals,
            ax=ax_xy,
            annotate=False,
            title="Best Configuration XY",
        )
    else:
        ax_xy.text(
            0.5,
            0.5,
            "No XY configuration",
            ha="center",
            va="center",
            transform=ax_xy.transAxes,
            color="gray",
        )
        ax_xy.set_axis_off()

    fig.suptitle(title or "PyQCA Dashboard")
    fig.tight_layout()
    _save_if_path(fig, path)
    return fig
