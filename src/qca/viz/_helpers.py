"""Shared helpers for optional Matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qca._constants import (
    COLOR_ACCENT,
    COLOR_GRID,
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_MEDIUM,
    CONSISTENCY_MODERATE,
    CONSISTENCY_STRONG,
    MEMBERSHIP_THRESHOLD,
)


def _require_matplotlib() -> Any:
    """Import matplotlib lazily and raise a helpful optional-extra error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "qca.viz plotting requires Matplotlib. Install it with "
            "`pip install pyqca[viz]` or `pip install matplotlib`."
        ) from exc
    return plt


def consistency_color(value: float) -> str:
    """Return a color for a consistency-like score."""
    if value >= CONSISTENCY_STRONG:
        return COLOR_HIGH
    if value >= CONSISTENCY_MODERATE:
        return COLOR_MEDIUM
    return COLOR_LOW


def scatter_point_color(
    x_val: float,
    y_val: float,
    threshold: float = MEMBERSHIP_THRESHOLD,
) -> str:
    """Return a quadrant color for XY membership plots."""
    if x_val > threshold and y_val > threshold:
        return COLOR_HIGH
    if x_val > threshold and y_val <= threshold:
        return COLOR_LOW
    return COLOR_ACCENT


def save_or_show(fig: Any, path: str | Path | None) -> None:
    """Save a figure when a path is supplied, otherwise show it."""
    plt = _require_matplotlib()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def _save_if_path(fig: Any, path: str | Path | None) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")


def apply_grid_style(ax: Any, axis: str = "both") -> None:
    """Apply a quiet grid style to an axes object."""
    ax.grid(axis=axis, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def add_crossover_lines(
    ax: Any,
    threshold: float = MEMBERSHIP_THRESHOLD,
    linestyle: str = ":",
    alpha: float = 0.7,
) -> None:
    """Add horizontal and vertical crossover lines."""
    ax.axhline(threshold, color="gray", linewidth=0.8, linestyle=linestyle, alpha=alpha)
    ax.axvline(threshold, color="gray", linewidth=0.8, linestyle=linestyle, alpha=alpha)


def annotate_cases(
    ax: Any,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    case_ids: list[str],
    fontsize: float = 7,
    color: str = "dimgray",
    offset: tuple[float, float] = (4, 4),
) -> None:
    """Add case-id annotations to a scatter plot."""
    for x_val, y_val, case_id in zip(x_vals, y_vals, case_ids, strict=False):
        ax.annotate(
            case_id,
            (x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=fontsize,
            color=color,
            zorder=4,
        )


def add_stat_box(
    ax: Any, text: str, loc: str = "upper left", fontsize: float = 9
) -> None:
    """Add a small statistics text box to an axes object."""
    loc_map = {
        "upper left": (0.03, 0.97, "top", "left"),
        "upper right": (0.97, 0.97, "top", "right"),
        "lower left": (0.03, 0.03, "bottom", "left"),
        "lower right": (0.97, 0.03, "bottom", "right"),
    }
    x_pos, y_pos, va, ha = loc_map.get(loc, loc_map["upper left"])
    ax.text(
        x_pos,
        y_pos,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
    )


def consistency_legend_patches() -> list[Any]:
    """Return Matplotlib legend patches for consistency strength."""
    try:
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "qca.viz plotting requires Matplotlib. Install it with "
            "`pip install pyqca[viz]` or `pip install matplotlib`."
        ) from exc
    return [
        mpatches.Patch(color=COLOR_HIGH, label=f"Strong (>= {CONSISTENCY_STRONG})"),
        mpatches.Patch(
            color=COLOR_MEDIUM, label=f"Moderate (>= {CONSISTENCY_MODERATE})"
        ),
        mpatches.Patch(color=COLOR_LOW, label=f"Weak (< {CONSISTENCY_MODERATE})"),
    ]
