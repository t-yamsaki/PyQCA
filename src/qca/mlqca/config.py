"""Configuration objects for machine-learning-enhanced QCA workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

MLQCAMode = Literal["conservative", "radical"]
MLQCAEngine = Literal["csqca", "fsqca", "gsqca"]
MLQCAValidation = Literal["paper", "cv", "bootstrap"]


@dataclass(frozen=True)
class MLQCAConfig:
    """Validated settings shared by future mlQCA workflow stages."""

    mode: MLQCAMode = "conservative"
    engine: MLQCAEngine = "csqca"
    top_k: int = 10
    model_size: int = 4
    max_cutoffs_per_condition: int = 3
    consistency_cutoff: float = 0.8
    frequency_cutoff: int = 1
    minimizer: str = "standard"
    random_state: int = 201
    validation: MLQCAValidation = "paper"
    n_splits: int = 5
    n_bootstrap: int = 100
    max_models: int = 10_000
    n_jobs: int = 1
    required_conditions: tuple[str, ...] = ()
    excluded_conditions: tuple[str, ...] = ()
    model_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_choice("mode", self.mode, {"conservative", "radical"})
        _validate_choice("engine", self.engine, {"csqca", "fsqca", "gsqca"})
        _validate_choice("validation", self.validation, {"paper", "cv", "bootstrap"})
        _validate_positive_int("top_k", self.top_k)
        _validate_positive_int("model_size", self.model_size)
        _validate_positive_int(
            "max_cutoffs_per_condition", self.max_cutoffs_per_condition
        )
        _validate_positive_int("frequency_cutoff", self.frequency_cutoff)
        _validate_positive_int("n_splits", self.n_splits, minimum=2)
        _validate_positive_int("n_bootstrap", self.n_bootstrap)
        _validate_positive_int("max_models", self.max_models)
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if self.model_size > self.top_k:
            raise ValueError("model_size cannot be greater than top_k.")
        if not 0.0 <= float(self.consistency_cutoff) <= 1.0:
            raise ValueError("consistency_cutoff must be in [0, 1].")
        if not str(self.minimizer).strip():
            raise ValueError("minimizer must be a non-empty string.")

        required = _normalize_names("required_conditions", self.required_conditions)
        excluded = _normalize_names("excluded_conditions", self.excluded_conditions)
        overlap = sorted(set(required) & set(excluded))
        if overlap:
            raise ValueError(
                "required_conditions and excluded_conditions overlap: "
                f"{overlap}"
            )

        object.__setattr__(self, "mode", str(self.mode).lower())
        object.__setattr__(self, "engine", str(self.engine).lower())
        object.__setattr__(self, "validation", str(self.validation).lower())
        object.__setattr__(self, "minimizer", str(self.minimizer).strip())
        object.__setattr__(self, "required_conditions", required)
        object.__setattr__(self, "excluded_conditions", excluded)
        object.__setattr__(
            self,
            "model_params",
            MappingProxyType(dict(self.model_params)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration dictionary."""
        return {
            "mode": self.mode,
            "engine": self.engine,
            "top_k": self.top_k,
            "model_size": self.model_size,
            "max_cutoffs_per_condition": self.max_cutoffs_per_condition,
            "consistency_cutoff": self.consistency_cutoff,
            "frequency_cutoff": self.frequency_cutoff,
            "minimizer": self.minimizer,
            "random_state": self.random_state,
            "validation": self.validation,
            "n_splits": self.n_splits,
            "n_bootstrap": self.n_bootstrap,
            "max_models": self.max_models,
            "n_jobs": self.n_jobs,
            "required_conditions": list(self.required_conditions),
            "excluded_conditions": list(self.excluded_conditions),
            "model_params": dict(self.model_params),
        }


def _validate_choice(name: str, value: str, available: set[str]) -> None:
    normalized = str(value).strip().lower()
    if normalized not in available:
        choices = ", ".join(sorted(available))
        raise ValueError(f"{name} must be one of: {choices}. Got {value!r}.")


def _validate_positive_int(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )


def _normalize_names(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{name} cannot contain empty names.")
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(f"{name} contains duplicate names: {duplicates}")
    return normalized


__all__ = [
    "MLQCAConfig",
    "MLQCAEngine",
    "MLQCAMode",
    "MLQCAValidation",
]
