"""Result objects for machine-learning-enhanced QCA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qca.results import QCAFitResult


@dataclass(frozen=True)
class MLQCAResult:
    """Unified result container populated across mlQCA workflow phases."""

    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cutoff_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_proposals: pd.DataFrame = field(default_factory=pd.DataFrame)
    condition_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_models: pd.DataFrame = field(default_factory=pd.DataFrame)
    qca_results: tuple[QCAFitResult | None, ...] = ()
    best_model_id: int | str | None = None
    pareto_models: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        table_fields = (
            "feature_importance",
            "cutoff_candidates",
            "calibration_proposals",
            "condition_ranking",
            "candidate_models",
            "pareto_models",
        )
        for name in table_fields:
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "qca_results", tuple(self.qca_results))
        object.__setattr__(self, "settings", dict(self.settings))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.best_model_id is not None and self.candidate_models.empty:
            raise ValueError(
                "best_model_id cannot be set when candidate_models is empty."
            )

    @property
    def best_qca_result(self) -> QCAFitResult | None:
        """Return the QCA result selected by ``best_model_id``."""
        if self.best_model_id is None:
            return None
        return self.get_qca_result(self.best_model_id)

    @property
    def n_models(self) -> int:
        return len(self.candidate_models)

    @property
    def n_valid_models(self) -> int:
        if "valid" not in self.candidate_models:
            return sum(result is not None for result in self.qca_results)
        return int(self.candidate_models["valid"].fillna(False).astype(bool).sum())

    def summary(self) -> pd.DataFrame:
        """Return the candidate-model summary table."""
        return self.candidate_models.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary()

    def get_qca_result(self, model_id: int | str) -> QCAFitResult | None:
        """Return one QCA result by model identifier."""
        if self.candidate_models.empty:
            raise KeyError(f"Unknown mlQCA model_id {model_id!r}.")
        if "model_id" in self.candidate_models:
            matches = self.candidate_models.index[
                self.candidate_models["model_id"] == model_id
            ].tolist()
            if not matches:
                raise KeyError(f"Unknown mlQCA model_id {model_id!r}.")
            position = self.candidate_models.index.get_loc(matches[0])
        elif isinstance(model_id, int) and 0 <= model_id < len(self.candidate_models):
            position = model_id
        else:
            raise KeyError(f"Unknown mlQCA model_id {model_id!r}.")

        if position >= len(self.qca_results):
            return None
        return self.qca_results[position]

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a compact Markdown report and optionally write it."""
        lines = ["# mlQCA Result", "", "## Settings", ""]
        if self.settings:
            lines.extend(f"- `{key}`: {value}" for key, value in self.settings.items())
        else:
            lines.append("- No settings recorded")

        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- `n_models`: {self.n_models}",
                f"- `n_valid_models`: {self.n_valid_models}",
                f"- `best_model_id`: {self.best_model_id}",
            ]
        )
        for title, table in (
            ("Condition Ranking", self.condition_ranking),
            ("Cutoff Candidates", self.cutoff_candidates),
            ("Calibration Proposals", self.calibration_proposals),
            ("Candidate Models", self.candidate_models),
            ("Pareto Models", self.pareto_models),
        ):
            lines.extend(["", f"## {title}", "", _dataframe_to_markdown(table)])

        markdown = "\n".join(lines)
        if path is not None:
            Path(path).write_text(markdown, encoding="utf-8")
        return markdown


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(_format_cell(value) for value in row)
        + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _format_cell(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


__all__ = ["MLQCAResult"]
