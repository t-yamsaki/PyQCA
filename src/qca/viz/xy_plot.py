"""XY membership scatter plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qca._constants import COLOR_ACCENT, COLOR_HIGH, COLOR_LOW, MEMBERSHIP_THRESHOLD
from qca.core.literals import MultiValueLiteral, SetLiteral, parse_conjunction
from qca.viz._helpers import (
    _require_matplotlib,
    _save_if_path,
    add_crossover_lines,
    add_stat_box,
    annotate_cases,
    apply_grid_style,
    scatter_point_color,
)


def _coerce_literals(
    literals: str | Sequence[SetLiteral | MultiValueLiteral],
) -> list[SetLiteral | MultiValueLiteral]:
    if isinstance(literals, str):
        return parse_conjunction(literals)
    return list(literals)


def plot_xy(
    model: Any,
    literals: str | Sequence[SetLiteral | MultiValueLiteral],
    ax: Any | None = None,
    title: str | None = None,
    annotate: bool = True,
    path: str | Path | None = None,
) -> Any:
    """Plot condition membership against outcome membership."""
    plt = _require_matplotlib()
    term = _coerce_literals(literals)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    x_vals = model.conjunction_membership(term).astype(float).to_numpy()
    y_vals = model.outcome_series.astype(float).to_numpy()
    case_ids = model.data[model.case_id].astype(str).tolist()
    label = " * ".join(literal.label() for literal in term)
    result = model.evaluate_sufficiency(term)

    ax.fill_between(
        [0, MEMBERSHIP_THRESHOLD],
        [MEMBERSHIP_THRESHOLD, MEMBERSHIP_THRESHOLD],
        [1.0, 1.0],
        color=COLOR_ACCENT,
        alpha=0.05,
    )
    ax.fill_between(
        [MEMBERSHIP_THRESHOLD, 1.0],
        [MEMBERSHIP_THRESHOLD, MEMBERSHIP_THRESHOLD],
        [1.0, 1.0],
        color=COLOR_HIGH,
        alpha=0.07,
    )
    ax.fill_between(
        [MEMBERSHIP_THRESHOLD, 1.0],
        [0.0, 0.0],
        [MEMBERSHIP_THRESHOLD, MEMBERSHIP_THRESHOLD],
        color=COLOR_LOW,
        alpha=0.07,
    )
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1.2, linestyle="--", label="Y = X")
    add_crossover_lines(ax)

    colors = [
        scatter_point_color(x_val, y_val)
        for x_val, y_val in zip(x_vals, y_vals, strict=False)
    ]
    ax.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=70,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.9,
    )
    if annotate:
        annotate_cases(ax, x_vals, y_vals, case_ids)

    unique = (
        "N/A" if result.unique_coverage is None else f"{result.unique_coverage:.3f}"
    )
    add_stat_box(
        ax,
        "\n".join(
            [
                f"Consistency: {result.consistency:.3f}",
                f"Coverage: {result.raw_coverage:.3f}",
                f"Unique: {unique}",
                f"N in: {result.n_cases_in}",
            ]
        ),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"Membership in [{label}]")
    ax.set_ylabel(f"Membership in [{model.outcome}]")
    ax.set_title(title or f"XY Plot: {label}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    apply_grid_style(ax)

    if standalone:
        fig.tight_layout()
    _save_if_path(fig, path)
    return ax
