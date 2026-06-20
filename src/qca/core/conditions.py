"""Condition schema objects shared by QCA engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ConditionKind = Literal["crisp", "fuzzy", "multi"]


@dataclass(frozen=True)
class ConditionSpec:
    """Normalized description of one QCA condition.

    Public engines can accept friendly APIs such as ``conditions`` plus
    ``condition_types`` or legacy ``set_conditions``. Internally they should
    converge on this small schema object so csQCA, fsQCA, mvQCA, and MixedQCA
    share the same condition contract.
    """

    name: str
    kind: ConditionKind
    domain: tuple[Any, ...] | None = None
    calibrated: bool | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("ConditionSpec.name must be a non-empty string.")
        normalized_kind = normalize_condition_kind(self.name, self.kind)
        object.__setattr__(self, "kind", normalized_kind)
        if self.domain is not None and not isinstance(self.domain, tuple):
            object.__setattr__(self, "domain", tuple(self.domain))

    @property
    def is_crisp(self) -> bool:
        return self.kind == "crisp"

    @property
    def is_fuzzy(self) -> bool:
        return self.kind == "fuzzy"

    @property
    def is_multi(self) -> bool:
        return self.kind == "multi"

    @property
    def is_set(self) -> bool:
        return self.kind in {"crisp", "fuzzy"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "domain": self.domain,
            "calibrated": self.calibrated,
        }


def normalize_condition_kind(condition: str, condition_type: str) -> ConditionKind:
    """Normalize user-facing condition type aliases."""
    value = str(condition_type).strip().lower().replace("_", "-")
    if value in {"crisp", "crisp-set", "cs", "csqca"}:
        return "crisp"
    if value in {"fuzzy", "fuzzy-set", "fs", "fsqca"}:
        return "fuzzy"
    if value in {"multi", "multi-value", "multivalue", "mv", "mvqca"}:
        return "multi"
    raise ValueError(
        f"condition_types[{condition!r}] has unknown type {condition_type!r}. "
        "Available types: crisp, fuzzy, multi"
    )


def check_duplicate_condition_names(conditions: list[str]) -> None:
    """Raise when a condition list contains duplicate names."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for condition in conditions:
        if condition in seen and condition not in duplicates:
            duplicates.append(condition)
        seen.add(condition)
    if duplicates:
        raise ValueError(f"conditions contains duplicate names: {duplicates}")


def specs_from_condition_types(
    conditions: list[str],
    condition_types: dict[str, str],
) -> list[ConditionSpec]:
    """Build ``ConditionSpec`` objects from PyQCA's schema-style API."""
    check_duplicate_condition_names(conditions)

    missing_types = [c for c in conditions if c not in condition_types]
    if missing_types:
        raise ValueError(f"condition_types is missing entries for: {missing_types}")

    extra_types = [c for c in condition_types if c not in conditions]
    if extra_types:
        raise ValueError(
            f"condition_types contains keys not present in conditions: {extra_types}"
        )

    return [
        ConditionSpec(
            name=condition,
            kind=normalize_condition_kind(condition, condition_types[condition]),
        )
        for condition in conditions
    ]


def condition_specs_from_schema(schema: Any) -> list[ConditionSpec]:
    """Normalize a user-facing condition schema into ``ConditionSpec`` objects.

    Accepted inputs are:

    - a sequence of ``ConditionSpec`` objects;
    - a sequence of mapping objects with ``name`` and ``kind``/``type`` fields;
    - a pandas-like DataFrame with matching columns.

    Optional mapping/DataFrame fields are ``domain`` and ``calibrated``.
    """
    if _is_dataframe_like(schema):
        records = schema.to_dict(orient="records")
    elif isinstance(schema, ConditionSpec):
        records = [schema]
    elif isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        records = list(schema)
    else:
        raise TypeError(
            "condition schema must be a ConditionSpec, a sequence of "
            "ConditionSpec/mapping objects, or a pandas DataFrame."
        )

    specs = [_coerce_condition_spec(record) for record in records]
    check_duplicate_condition_names([spec.name for spec in specs])
    return specs


def _is_dataframe_like(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "to_dict")


def _coerce_condition_spec(value: Any) -> ConditionSpec:
    if isinstance(value, ConditionSpec):
        return value
    if isinstance(value, Mapping):
        return _condition_spec_from_mapping(value)
    raise TypeError(
        "condition schema entries must be ConditionSpec objects or mappings."
    )


def _condition_spec_from_mapping(row: Mapping[str, Any]) -> ConditionSpec:
    name = _first_present(row, ("name", "condition"))
    kind = _first_present(row, ("kind", "type", "condition_type"))
    if _is_missing(name):
        raise ValueError("condition schema entry is missing a 'name' field.")
    if _is_missing(kind):
        raise ValueError(
            f"condition schema entry for {name!r} is missing a 'kind'/'type' field."
        )

    domain = row.get("domain")
    calibrated = row.get("calibrated")
    return ConditionSpec(
        name=str(name),
        kind=str(kind),
        domain=None if _is_missing(domain) else domain,
        calibrated=None if _is_missing(calibrated) else bool(calibrated),
    )


def _first_present(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        missing = pd.isna(value)
        try:
            return bool(missing)
        except (TypeError, ValueError):
            pass
    except (TypeError, ValueError):
        pass
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def split_condition_specs(
    specs: list[ConditionSpec],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Return engine-facing condition lists from normalized condition specs."""
    set_conditions = [spec.name for spec in specs if spec.is_set]
    multivalue_conditions = [spec.name for spec in specs if spec.is_multi]
    conditions = [spec.name for spec in specs]
    condition_types = {spec.name: spec.kind for spec in specs}
    return set_conditions, multivalue_conditions, conditions, condition_types
