"""Validation and normalization for mlQCA input data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from qca.mlqca.schema import MLConditionSpec


@dataclass(frozen=True)
class ValidatedMLQCAInput:
    """Normalized input returned by :func:`validate_mlqca_input`."""

    data: pd.DataFrame
    outcome: str
    candidates: tuple[str, ...]
    condition_specs: tuple[MLConditionSpec, ...]
    case_id: str | None = None

    @property
    def n_cases(self) -> int:
        return len(self.data)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def features(self) -> pd.DataFrame:
        """Return a copy of the validated predictor matrix."""
        return self.data.loc[:, list(self.candidates)].copy()

    @property
    def target(self) -> pd.Series:
        """Return the binary outcome as an integer Series."""
        return self.data[self.outcome].astype(int).copy()


def validate_mlqca_input(
    data: pd.DataFrame,
    outcome: str,
    candidates: Sequence[str] | None = None,
    *,
    case_id: str | None = None,
    condition_specs: Sequence[MLConditionSpec] | None = None,
    allow_missing_features: bool = True,
) -> ValidatedMLQCAInput:
    """Validate and normalize input for future mlQCA workflow stages.

    P0 accepts numeric or boolean candidate columns. Nominal variables should
    be encoded before use; native categorical support belongs to the ML
    backend phase.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one case.")
    if not data.columns.is_unique:
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"data contains duplicate column names: {duplicates}")

    outcome_name = _normalize_column_name("outcome", outcome)
    case_id_name = (
        None if case_id is None else _normalize_column_name("case_id", case_id)
    )
    _require_columns(data, [outcome_name])
    if case_id_name is not None:
        _require_columns(data, [case_id_name])
        if case_id_name == outcome_name:
            raise ValueError("case_id cannot be the outcome column.")
        if data[case_id_name].isna().any():
            raise ValueError("case_id cannot contain missing values.")
        if data[case_id_name].duplicated().any():
            raise ValueError("case_id values must be unique.")

    specs = _normalize_specs(condition_specs)
    candidate_names = _resolve_candidates(
        data,
        outcome=outcome_name,
        case_id=case_id_name,
        candidates=candidates,
        condition_specs=specs,
    )
    _validate_outcome(data[outcome_name], outcome_name)
    _validate_candidate_columns(
        data,
        candidate_names,
        allow_missing_features=allow_missing_features,
    )

    if specs:
        enabled_specs = tuple(spec for spec in specs if spec.enabled)
    else:
        enabled_specs = tuple(
            MLConditionSpec(
                name=name,
                data_type=_infer_data_type(data[name]),
            )
            for name in candidate_names
        )

    work = data.copy()
    work[outcome_name] = work[outcome_name].astype(int)
    return ValidatedMLQCAInput(
        data=work,
        outcome=outcome_name,
        candidates=candidate_names,
        condition_specs=enabled_specs,
        case_id=case_id_name,
    )


def _normalize_column_name(name: str, value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


def _normalize_specs(
    condition_specs: Sequence[MLConditionSpec] | None,
) -> tuple[MLConditionSpec, ...]:
    if condition_specs is None:
        return ()
    specs = tuple(condition_specs)
    if any(not isinstance(spec, MLConditionSpec) for spec in specs):
        raise TypeError("condition_specs must contain MLConditionSpec objects.")
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"condition_specs contains duplicate names: {duplicates}")
    return specs


def _resolve_candidates(
    data: pd.DataFrame,
    *,
    outcome: str,
    case_id: str | None,
    candidates: Sequence[str] | None,
    condition_specs: tuple[MLConditionSpec, ...],
) -> tuple[str, ...]:
    if candidates is not None and condition_specs:
        raise ValueError("candidates and condition_specs cannot be provided together.")

    if condition_specs:
        resolved = tuple(spec.name for spec in condition_specs if spec.enabled)
    elif candidates is None:
        excluded = {outcome}
        if case_id is not None:
            excluded.add(case_id)
        resolved = tuple(
            str(column) for column in data.columns if column not in excluded
        )
    else:
        resolved = tuple(
            _normalize_column_name("candidate", value) for value in candidates
        )

    if not resolved:
        raise ValueError("At least one candidate condition is required.")
    duplicates = sorted({name for name in resolved if resolved.count(name) > 1})
    if duplicates:
        raise ValueError(f"candidates contains duplicate names: {duplicates}")
    forbidden = [name for name in resolved if name in {outcome, case_id}]
    if forbidden:
        raise ValueError(
            "Candidate conditions cannot contain outcome or case_id columns: "
            f"{forbidden}"
        )
    _require_columns(data, resolved)
    return resolved


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")


def _validate_outcome(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"Outcome column {name!r} cannot contain missing values.")
    values = set(pd.unique(series))
    if not values or not values.issubset({0, 1}):
        raise ValueError(
            f"Outcome column {name!r} must be binary with values 0 and 1."
        )
    if len(values) != 2:
        raise ValueError(
            f"Outcome column {name!r} must contain both outcome classes 0 and 1."
        )


def _validate_candidate_columns(
    data: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    allow_missing_features: bool,
) -> None:
    non_numeric = [
        name
        for name in candidates
        if not (is_numeric_dtype(data[name]) or is_bool_dtype(data[name]))
    ]
    if non_numeric:
        raise TypeError(
            "mlQCA candidate columns must be numeric or boolean in P0. "
            f"Encode nominal columns before use: {non_numeric}"
        )

    all_missing = [name for name in candidates if data[name].isna().all()]
    if all_missing:
        raise ValueError(f"Candidate columns cannot be entirely missing: {all_missing}")
    if not allow_missing_features:
        with_missing = [name for name in candidates if data[name].isna().any()]
        if with_missing:
            raise ValueError(
                "Candidate columns contain missing values while "
                f"allow_missing_features=False: {with_missing}"
            )

    constant = [
        name
        for name in candidates
        if data[name].replace([np.inf, -np.inf], np.nan).nunique(dropna=True) < 2
    ]
    if constant:
        raise ValueError(f"Candidate columns must not be constant: {constant}")

    infinite = [
        name
        for name in candidates
        if np.isinf(
            pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)
        ).any()
    ]
    if infinite:
        raise ValueError(f"Candidate columns contain infinite values: {infinite}")


def _infer_data_type(series: pd.Series) -> str:
    clean = series.dropna()
    unique = clean.nunique()
    if is_bool_dtype(series) or unique == 2:
        return "binary"
    if is_numeric_dtype(series) and unique <= 10:
        return "ordinal"
    return "continuous"


__all__ = [
    "ValidatedMLQCAInput",
    "validate_mlqca_input",
]
