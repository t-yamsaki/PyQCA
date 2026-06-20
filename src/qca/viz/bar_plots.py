"""Bar plots for sufficiency and necessity results."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import (
    CONSISTENCY_MODERATE,
    CONSISTENCY_NECESSITY,
)
from qca.viz._helpers import (
    _require_matplotlib,
    _save_if_path,
    apply_grid_style,
    consistency_color,
    consistency_legend_patches,
)


def _normalize_results(results: Any) -> pd.DataFrame:
    if hasattr(results, "solutions"):
        return results.solutions.copy()
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def _label_col(df: pd.DataFrame, fallback: str = "item") -> str:
    for candidate in ("antecedent", "term", "condition", "solution_type"):
        if candidate in df.columns:
            return candidate
    df[fallback] = [str(i + 1) for i in range(len(df))]
    return fallback


def _coverage_col(df: pd.DataFrame) -> str | None:
    for candidate in ("raw_coverage", "coverage"):
        if candidate in df.columns:
            return candidate
    return None


def _make_axes_pair(
    ax_consistency: Any | None, ax_coverage: Any | None, figsize: tuple[float, float]
) -> tuple[Any, Any, Any]:
    plt = _require_matplotlib()
    if ax_consistency is None or ax_coverage is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        return fig, axes[0], axes[1]
    return ax_consistency.figure, ax_consistency, ax_coverage


def _draw_barh(
    ax: Any,
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    xlabel: str,
    threshold: float,
    color_by: str | None = None,
) -> None:
    positions = np.arange(len(df))
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    color_values = (
        pd.to_numeric(df[color_by], errors="coerce").fillna(0.0) if color_by else values
    )
    colors = [consistency_color(float(value)) for value in color_values]
    bars = ax.barh(positions, values, color=colors, edgecolor="white", alpha=0.88)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            min(float(value) + 0.015, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}",
            va="center",
            fontsize=8,
        )
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_yticks(positions)
    ax.set_yticklabels(df[label_col].astype(str), fontsize=8)
    ax.set_xlabel(xlabel)
    apply_grid_style(ax, axis="x")


def plot_sufficiency_bar(
    results: Any,
    top_n: int = 15,
    consistency_threshold: float = CONSISTENCY_MODERATE,
    ax_consistency: Any | None = None,
    ax_coverage: Any | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot sufficiency consistency and coverage bars."""
    df = _normalize_results(results)
    fig, ax_consistency, ax_coverage = _make_axes_pair(
        ax_consistency,
        ax_coverage,
        (14, max(4, min(max(len(df), 1), top_n) * 0.45 + 1.5)),
    )
    if df.empty:
        for ax in (ax_consistency, ax_coverage):
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
        _save_if_path(fig, path)
        return ax_consistency, ax_coverage

    if "consistency" not in df.columns:
        raise ValueError("results must include a 'consistency' column")
    coverage_col = _coverage_col(df)
    if coverage_col is None:
        raise ValueError("results must include 'raw_coverage' or 'coverage'")

    label_col = _label_col(df)
    df = df.head(top_n).iloc[::-1].reset_index(drop=True)
    _draw_barh(
        ax_consistency,
        df,
        "consistency",
        label_col,
        "Consistency",
        consistency_threshold,
    )
    _draw_barh(
        ax_coverage,
        df,
        coverage_col,
        label_col,
        "Coverage",
        consistency_threshold,
        color_by="consistency",
    )
    ax_consistency.set_ylabel("Configuration")
    fig.suptitle(title or f"Sufficiency Analysis (Top {top_n})")
    with contextlib.suppress(ImportError):
        fig.legend(
            handles=consistency_legend_patches(), loc="upper center", ncol=3, fontsize=8
        )
    fig.tight_layout()
    _save_if_path(fig, path)
    return ax_consistency, ax_coverage


def plot_necessity_bar(
    results: Any,
    consistency_threshold: float = CONSISTENCY_NECESSITY,
    ax_consistency: Any | None = None,
    ax_coverage: Any | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot necessity consistency and coverage bars."""
    df = _normalize_results(results)
    fig, ax_consistency, ax_coverage = _make_axes_pair(
        ax_consistency, ax_coverage, (13, max(3, len(df) * 0.5 + 1.5))
    )
    if df.empty:
        for ax in (ax_consistency, ax_coverage):
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
        _save_if_path(fig, path)
        return ax_consistency, ax_coverage

    if "consistency" not in df.columns or "coverage" not in df.columns:
        raise ValueError("necessity results must include 'consistency' and 'coverage'")
    label_col = _label_col(df)
    df = df.iloc[::-1].reset_index(drop=True)
    _draw_barh(
        ax_consistency,
        df,
        "consistency",
        label_col,
        "Necessity Consistency",
        consistency_threshold,
    )
    _draw_barh(
        ax_coverage,
        df,
        "coverage",
        label_col,
        "Necessity Coverage",
        consistency_threshold,
        color_by="consistency",
    )
    ax_consistency.set_ylabel("Condition")
    fig.suptitle(title or "Necessity Analysis")
    fig.tight_layout()
    _save_if_path(fig, path)
    return ax_consistency, ax_coverage
