"""Shared helpers for first-class single-family QCA engines."""

from __future__ import annotations

from collections.abc import Sequence

from qca.engines.base import QCAEngineBase


def resolve_engine_conditions(
    *,
    engine_name: str,
    conditions: Sequence[str] | None,
    legacy_conditions: Sequence[str] | None,
    legacy_name: str,
) -> list[str]:
    """Resolve canonical and legacy condition arguments."""
    if conditions is not None and legacy_conditions is not None:
        raise ValueError(
            f"{engine_name} accepts either conditions or {legacy_name}, not both."
        )

    resolved = conditions if conditions is not None else legacy_conditions
    if resolved is None:
        raise ValueError(f"{engine_name} requires conditions.")

    resolved_list = list(resolved)
    if not resolved_list:
        raise ValueError(f"{engine_name} requires at least one condition.")

    return resolved_list


def normalize_engine_condition_types(
    *,
    engine_name: str,
    condition_types: dict[str, str],
    allowed_types: set[str],
) -> dict[str, str]:
    """Normalize and constrain condition types before core engine setup."""
    normalized: dict[str, str] = {}
    for condition, condition_type in condition_types.items():
        normalized_type = QCAEngineBase._normalize_condition_type(
            condition, condition_type
        )
        if normalized_type not in allowed_types:
            allowed = ", ".join(sorted(allowed_types))
            raise ValueError(
                f"{engine_name} only supports {allowed} condition types; "
                f"{condition!r} was {normalized_type!r}."
            )
        normalized[condition] = normalized_type

    return normalized
