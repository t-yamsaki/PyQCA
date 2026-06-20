"""Truth-table heatmap utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qca._constants import COLOR_HIGH, CONSISTENCY_MODERATE
from qca.viz._helpers import (
    _require_matplotlib,
    _save_if_path,
    apply_grid_style,
    consistency_color,
)


def _build_condition_matrix(
    rows: list[Any],
    all_conditions: list[str],
    set_conditions: list[str],
    multivalue_conditions: list[str],
) -> tuple[np.ndarray, dict[str, dict[Any, int]]]:
    """Build a numeric condition matrix and maps for multi-value categories."""
    matrix = np.full((len(rows), len(all_conditions)), np.nan)
    value_maps: dict[str, dict[Any, int]] = {}
    for condition in multivalue_conditions:
        values = sorted(
            {
                row.config.get(condition)
                for row in rows
                if row.config.get(condition) is not None
            },
            key=str,
        )
        value_maps[condition] = {value: index for index, value in enumerate(values)}

    for row_index, row in enumerate(rows):
        for col_index, condition in enumerate(all_conditions):
            value = row.config.get(condition)
            if value is None:
                continue
            if condition in set_conditions:
                matrix[row_index, col_index] = float(value)
            else:
                denom = max(len(value_maps.get(condition, {})) - 1, 1)
                matrix[row_index, col_index] = (
                    float(value_maps[condition].get(value, 0)) / denom
                )
    return matrix, value_maps


def draw_truth_table(
    model: Any,
    ax_cond: Any,
    ax_outcome: Any,
    ax_consist: Any,
    outcome_threshold: float = 0.5,
    fontsize_cell: float = 9,
    fontsize_label: float = 8,
) -> None:
    """Draw a truth-table heatmap across three axes."""
    plt = _require_matplotlib()
    rows = model.build_truth_table(outcome_threshold=outcome_threshold)
    if not rows:
        ax_cond.text(
            0.5,
            0.5,
            "Truth table is empty",
            ha="center",
            va="center",
            transform=ax_cond.transAxes,
        )
        return

    conditions = model.all_condition_names
    matrix, _ = _build_condition_matrix(
        rows, conditions, model.set_conditions, model.multivalue_conditions
    )
    ax_cond.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    for row_index, row in enumerate(rows):
        for col_index, condition in enumerate(conditions):
            value = row.config.get(condition, "?")
            text = (
                str(int(value))
                if condition in model.set_conditions and value != "?"
                else str(value)
            )
            cell_value = matrix[row_index, col_index]
            ax_cond.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=fontsize_cell,
                color="white"
                if np.isfinite(cell_value) and cell_value > 0.55
                else "black",
                fontweight="bold",
            )
        if row.include:
            ax_cond.add_patch(
                plt.Rectangle(
                    (-0.5, row_index - 0.5),
                    len(conditions),
                    1,
                    fill=False,
                    edgecolor=COLOR_HIGH,
                    linewidth=2.0,
                )
            )

    labels = [
        f"R{i + 1:02d} | n={row.n_cases} | {'in' if row.include else 'out'}"
        for i, row in enumerate(rows)
    ]
    ax_cond.set_xticks(range(len(conditions)))
    ax_cond.set_xticklabels(conditions, fontsize=fontsize_cell, rotation=30, ha="right")
    ax_cond.set_yticks(range(len(rows)))
    ax_cond.set_yticklabels(labels, fontsize=fontsize_label)
    ax_cond.set_title("Condition Pattern")

    outcome_values = np.array([row.outcome_mean for row in rows], dtype=float)
    consistency_values = np.array(
        [row.outcome_raw_consist for row in rows], dtype=float
    )
    ax_outcome.barh(
        range(len(rows)),
        outcome_values,
        color=[consistency_color(v) for v in outcome_values],
        edgecolor="white",
    )
    ax_outcome.axvline(outcome_threshold, color="gray", linestyle="--", linewidth=1.1)
    ax_outcome.set_xlim(0, 1.05)
    ax_outcome.set_yticks([])
    ax_outcome.set_title("Outcome Mean")
    apply_grid_style(ax_outcome, axis="x")

    ax_consist.barh(
        range(len(rows)),
        consistency_values,
        color=[consistency_color(v) for v in consistency_values],
        edgecolor="white",
    )
    ax_consist.axvline(
        CONSISTENCY_MODERATE, color="gray", linestyle="--", linewidth=1.1
    )
    ax_consist.set_xlim(0, 1.05)
    ax_consist.set_yticks([])
    ax_consist.set_title("Row Consistency")
    apply_grid_style(ax_consist, axis="x")


def plot_truth_table(
    model: Any,
    outcome_threshold: float = 0.5,
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    """Plot a truth-table heatmap and return the figure."""
    plt = _require_matplotlib()
    rows = model.build_truth_table(outcome_threshold=outcome_threshold)
    n_rows = max(len(rows), 1)
    n_conditions = max(len(getattr(model, "all_condition_names", [])), 1)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(max(10, n_conditions * 1.2 + 5), max(4.5, n_rows * 0.45 + 2)),
        gridspec_kw={"width_ratios": [max(n_conditions, 1), 1.2, 1.2]},
    )
    draw_truth_table(
        model, axes[0], axes[1], axes[2], outcome_threshold=outcome_threshold
    )
    fig.suptitle(title or "Truth Table")
    fig.tight_layout()
    _save_if_path(fig, path)
    return fig
