"""Membership distribution visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qca._constants import COLOR_ACCENT, COLOR_HIGH, COLOR_LOW, MEMBERSHIP_THRESHOLD
from qca.viz._helpers import _require_matplotlib, _save_if_path, apply_grid_style


def draw_membership_distribution(model: Any, axes: list[Any]) -> None:
    """Draw membership distributions for set conditions and the outcome."""
    cols = [*model.set_conditions, model.outcome]
    rng = np.random.default_rng(42)
    for ax, col in zip(axes, cols, strict=False):
        values = model.data[col].astype(float).to_numpy()
        parts = ax.violinplot(values, positions=[0], showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(COLOR_ACCENT)
            body.set_alpha(0.6)
        parts["cmedians"].set_color(COLOR_HIGH)
        jitter = rng.uniform(-0.06, 0.06, size=len(values))
        ax.scatter(
            jitter,
            values,
            color=COLOR_ACCENT,
            alpha=0.55,
            s=25,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.axhline(MEMBERSHIP_THRESHOLD, color=COLOR_LOW, linestyle="--", linewidth=1.1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_title(col)
        apply_grid_style(ax, axis="y")


def plot_membership_distribution(
    model: Any,
    title: str | None = None,
    path: str | Path | None = None,
) -> list[Any]:
    """Plot distributions for set conditions and outcome."""
    plt = _require_matplotlib()
    n_cols = max(len(model.set_conditions) + 1, 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.8, 5), sharey=True)
    axes_list = [axes] if n_cols == 1 else list(axes)
    draw_membership_distribution(model, axes_list)
    axes_list[0].set_ylabel("Membership")
    fig.suptitle(title or "Membership Distributions")
    fig.tight_layout()
    _save_if_path(fig, path)
    return axes_list
