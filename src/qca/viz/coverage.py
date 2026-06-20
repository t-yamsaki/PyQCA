"""Coverage visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qca._constants import COLOR_ACCENT, COLOR_HIGH, COLOR_MEDIUM
from qca.viz._helpers import _require_matplotlib, _save_if_path, apply_grid_style


def draw_unique_coverage(model: Any, terms: Sequence[Sequence[Any]], ax: Any) -> None:
    """Draw unique/shared coverage for solution terms."""
    if not terms:
        ax.text(
            0.5,
            0.5,
            "No terms provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        return

    term_results = model.compute_solution_coverages(terms)
    solution_result = model.evaluate_solution(terms)
    labels = [result.antecedent for result in term_results]
    raw = [float(result.raw_coverage) for result in term_results]
    unique = [
        0.0 if result.unique_coverage is None else float(result.unique_coverage)
        for result in term_results
    ]
    shared = [max(r - u, 0.0) for r, u in zip(raw, unique, strict=False)]
    positions = np.arange(len(labels))

    ax.bar(positions, unique, color=COLOR_ACCENT, edgecolor="white", label="Unique")
    ax.bar(
        positions,
        shared,
        bottom=unique,
        color=COLOR_MEDIUM,
        edgecolor="white",
        label="Shared",
    )
    ax.axhline(
        solution_result.raw_coverage,
        color=COLOR_HIGH,
        linestyle="--",
        linewidth=1.5,
        label=f"Solution {solution_result.raw_coverage:.3f}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    apply_grid_style(ax, axis="y")


def plot_unique_coverage(
    model: Any,
    terms: Sequence[Sequence[Any]],
    ax: Any | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    """Plot unique/shared coverage for solution terms."""
    plt = _require_matplotlib()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, max(len(terms), 1) * 1.8), 5))
    else:
        fig = ax.figure
    draw_unique_coverage(model, terms, ax)
    ax.set_title(title or "Unique and Shared Coverage")
    if standalone:
        fig.tight_layout()
    _save_if_path(fig, path)
    return ax
