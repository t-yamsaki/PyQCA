"""Optional Matplotlib visualizations for mlQCA results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def plot_mlqca_importance(
    result: Any,
    *,
    top_n: int = 15,
    metric: str = "shap_mean_abs",
    path: str | Path | None = None,
):
    """Plot ranked mlQCA feature importance."""
    plt = _require_matplotlib()
    table = _table(result, "feature_importance")
    if metric not in table or "feature" not in table:
        raise ValueError(f"feature_importance must include feature and {metric}.")
    view = table.dropna(subset=[metric]).nlargest(top_n, metric).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(view) * 0.4 + 1)))
    ax.barh(view["feature"].astype(str), view[metric].astype(float), color="#2f6f8f")
    ax.set_xlabel(metric)
    ax.set_ylabel("Condition")
    ax.set_title("mlQCA Condition Importance")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_mlqca_cutoffs(
    result: Any,
    feature: str,
    *,
    path: str | Path | None = None,
):
    """Plot XGBoost cutoff frequency for one feature."""
    plt = _require_matplotlib()
    table = _table(result, "cutoff_candidates")
    required = {"feature", "cutoff", "count"}
    if not required.issubset(table.columns):
        raise ValueError(f"cutoff_candidates must include {sorted(required)}.")
    view = table.loc[table["feature"] == feature].sort_values("cutoff")
    if view.empty:
        raise ValueError(f"No cutoff candidates found for {feature!r}.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(view["cutoff"].astype(str), view["count"].astype(float), color="#5b8c5a")
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Split count")
    ax.set_title(f"mlQCA Cutoff Candidates: {feature}")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_mlqca_frontier(
    result: Any,
    *,
    path: str | Path | None = None,
):
    """Plot candidate models and highlight the Pareto frontier."""
    plt = _require_matplotlib()
    models = _table(result, "candidate_models")
    required = {"coverage", "consistency", "complexity", "model_id"}
    if not required.issubset(models.columns):
        raise ValueError(f"candidate_models must include {sorted(required)}.")
    valid = models.dropna(subset=["coverage", "consistency", "complexity"])
    frontier = _table(result, "pareto_models")
    fig, ax = plt.subplots(figsize=(7, 5))
    if not valid.empty:
        scatter = ax.scatter(
            valid["coverage"],
            valid["consistency"],
            c=valid["complexity"],
            cmap="viridis_r",
            alpha=0.65,
            label="Candidate models",
        )
        fig.colorbar(scatter, ax=ax, label="Complexity")
    if not frontier.empty:
        ax.scatter(
            frontier["coverage"],
            frontier["consistency"],
            facecolors="none",
            edgecolors="#b22222",
            linewidths=1.8,
            s=90,
            label="Pareto frontier",
        )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Consistency")
    ax.set_title("mlQCA Model Frontier")
    ax.grid(color="#d9d9d9", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


def _table(result: Any, name: str) -> pd.DataFrame:
    value = getattr(result, name, None)
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"result.{name} must be a pandas DataFrame.")
    return value.copy()


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "mlQCA visualization requires Matplotlib. "
            "Install it with `pip install pyqca[viz]`."
        ) from exc
    return plt


def _save(fig: Any, path: str | Path | None) -> None:
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")


__all__ = [
    "plot_mlqca_cutoffs",
    "plot_mlqca_frontier",
    "plot_mlqca_importance",
]
