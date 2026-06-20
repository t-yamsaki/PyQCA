"""Backend contracts and result objects for mlQCA predictors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from qca.mlqca.validation import ValidatedMLQCAInput


@dataclass(frozen=True)
class PredictorFitResult:
    """Result returned by an mlQCA predictive backend."""

    backend: str
    model: Any
    feature_importance: pd.DataFrame
    cutoff_candidates: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = str(self.backend).strip()
        if not backend:
            raise ValueError("backend must be a non-empty string.")
        object.__setattr__(self, "backend", backend)
        for name in ("feature_importance", "cutoff_candidates", "predictions"):
            value = getattr(self, name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            object.__setattr__(self, name, value.copy())
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in self.metrics.items()},
        )
        object.__setattr__(self, "settings", dict(self.settings))

    @property
    def condition_ranking(self) -> pd.DataFrame:
        """Return feature importance ordered for condition selection."""
        return self.feature_importance.copy()

    @property
    def used_features(self) -> tuple[str, ...]:
        """Return features used by at least one tree split."""
        if "used" not in self.feature_importance:
            return ()
        return tuple(
            self.feature_importance.loc[
                self.feature_importance["used"].astype(bool),
                "feature",
            ].astype(str)
        )


@runtime_checkable
class PredictorBackend(Protocol):
    """Protocol implemented by mlQCA predictive backends."""

    name: str

    def fit(self, validated: ValidatedMLQCAInput) -> PredictorFitResult:
        """Fit a predictor and return analysis-ready backend output."""


__all__ = ["PredictorBackend", "PredictorFitResult"]
