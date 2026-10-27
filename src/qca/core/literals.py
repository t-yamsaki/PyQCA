"""Literals module."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca._constants import (
    CROSSOVER_EPSILON,
    FULL_MEMBERSHIP,
    MEMBERSHIP_THRESHOLD,
    NO_MEMBERSHIP,
)
from qca._validation import validate_category_value

# ---------------------------------------------------------------------------
# SetLiteral
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetLiteral:
    """Literal for a crisp-set or fuzzy-set condition."""

    name: str
    negated: bool = False

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Return a human-readable label."""
        return f"~{self.name}" if self.negated else self.name

    def __repr__(self) -> str:
        return f"SetLiteral({self.label()!r})"

    # ------------------------------------------------------------------
    # Membership calculation
    # ------------------------------------------------------------------

    def membership(self, df: pd.DataFrame) -> pd.Series:
        """Return membership scores for this literal."""
        if self.name not in df.columns:
            raise KeyError(
                f"SetLiteral '{self.name}' is not present in the DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        s = pd.to_numeric(df[self.name], errors="raise").astype(float)

        # Validate the [0.0, 1.0] membership range.
        if not ((s >= NO_MEMBERSHIP).all() and (s <= FULL_MEMBERSHIP).all()):
            raise ValueError(
                f"SetLiteral '{self.name}' values must be within [0, 1]. "
                f"Observed range: [{s.min():.4f}, {s.max():.4f}]"
            )

        # Ragin (2008, p. 30): cases at 0.5 are analytically ambiguous.
        at_crossover = (s - MEMBERSHIP_THRESHOLD).abs() < CROSSOVER_EPSILON
        if at_crossover.any():
            n_at = int(at_crossover.sum())
            warnings.warn(
                f"SetLiteral '{self.name}' has {n_at} case(s) at the crossover "
                "point (0.5). Ragin (2008, p. 30) recommends avoiding exact "
                "0.5 memberships because of ambiguity. Review the calibration.",
                UserWarning,
                stacklevel=2,
            )

        return (FULL_MEMBERSHIP - s) if self.negated else s

    # ------------------------------------------------------------------
    # Set-operation utilities
    # ------------------------------------------------------------------

    def negate(self) -> SetLiteral:
        """Return the negated literal."""
        return SetLiteral(self.name, not self.negated)

    def is_negation_of(self, other: object) -> bool:
        """Return whether another literal is the logical negation of this one."""
        if not isinstance(other, SetLiteral):
            return False
        return self.name == other.name and self.negated != other.negated


# ---------------------------------------------------------------------------
# MultiValueLiteral
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiValueLiteral:
    """Literal for a multi-value QCA condition."""

    name: str
    value: Any

    def __post_init__(self) -> None:
        validate_category_value(self.value)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Return a human-readable label."""
        if _is_value_collection(self.value):
            values = _sorted_values(self.value)
            return f"{self.name}={{{','.join(str(v) for v in values)}}}"
        return f"{self.name}={self.value}"

    def __repr__(self) -> str:
        return f"MultiValueLiteral({self.name!r}, {self.value!r})"

    # ------------------------------------------------------------------
    # Membership calculation
    # ------------------------------------------------------------------

    def membership(self, df: pd.DataFrame) -> pd.Series:
        """Return membership scores for this literal."""
        values = _value_set(self.value)
        fuzzy_membership = _multivalue_membership_from_columns(df, self.name, values)
        if fuzzy_membership is not None:
            return fuzzy_membership

        if self.name not in df.columns:
            raise KeyError(
                f"MultiValueLiteral '{self.name}' is not present in the DataFrame. "
                "No value-specific membership columns were found. "
                f"Available columns: {list(df.columns)}"
            )

        col = df[self.name]

        if _is_value_collection(self.value):
            missing_values = values - frozenset(col.dropna().unique().tolist())
            if missing_values:
                warnings.warn(
                    f"Values {sorted(missing_values, key=str)!r} for "
                    f"MultiValueLiteral '{self.label()}' are not present in "
                    "the DataFrame. Available values: "
                    f"{sorted(col.dropna().unique().tolist(), key=str)}",
                    UserWarning,
                    stacklevel=2,
                )
            return col.isin(values).astype(float)

        # Check that the category value is present.
        if self.value not in col.values:
            warnings.warn(
                f"Value {self.value!r} for MultiValueLiteral "
                f"'{self.name}={self.value}' is not present in the DataFrame. "
                "Available values: "
                f"{sorted(col.dropna().unique().tolist(), key=str)}\n"
                "Check for a type mismatch, such as int versus str.",
                UserWarning,
                stacklevel=2,
            )

        return (col == self.value).astype(float)

    # ------------------------------------------------------------------
    # Set-operation utilities
    # ------------------------------------------------------------------

    def conflicts_with(self, other: object) -> bool:
        """Conflicts with."""
        if not isinstance(other, MultiValueLiteral):
            return False
        if self.name != other.name:
            return False
        return _value_set(self.value).isdisjoint(_value_set(other.value))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def parse_literal(text: str) -> SetLiteral | MultiValueLiteral:
    """Parse a literal label into a literal object."""
    text = text.strip()
    if not text:
        raise ValueError("Literal string cannot be empty.")

    if "=" in text:
        # MultiValueLiteral: "name=value"
        name, val_str = text.split("=", 1)
        value: Any = _parse_value(val_str)
        return MultiValueLiteral(name.strip(), value)

    # SetLiteral: "name" or "~name"
    if text.startswith("~"):
        return SetLiteral(text[1:].strip(), negated=True)
    return SetLiteral(text)


def _parse_value(val_str: str) -> Any:
    """Parse a string token into a Python value."""
    val_str = val_str.strip()
    if val_str.startswith("{") and val_str.endswith("}"):
        inner = val_str[1:-1].strip()
        if not inner:
            raise ValueError("Multi-value set literal cannot be empty.")
        return frozenset(_parse_value(part) for part in inner.split(","))
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def _multivalue_membership_from_columns(
    df: pd.DataFrame,
    name: str,
    values: frozenset[Any],
) -> pd.Series | None:
    available = {
        value: _find_multivalue_membership_column(df, name, value) for value in values
    }
    present = {value: column for value, column in available.items() if column}
    if not present:
        return None

    missing = [value for value, column in available.items() if column is None]
    if missing:
        raise KeyError(
            f"Missing value-specific membership columns for MultiValueLiteral "
            f"'{name}' values {sorted(missing, key=str)!r}. Available columns: "
            f"{list(df.columns)}"
        )

    memberships = [
        _coerce_membership_column(df[column], f"{name}={value}")
        for value, column in present.items()
    ]
    if len(memberships) == 1:
        return memberships[0]
    return pd.concat(memberships, axis=1).max(axis=1)


def _find_multivalue_membership_column(
    df: pd.DataFrame,
    name: str,
    value: Any,
) -> str | None:
    for candidate in _multivalue_membership_column_candidates(name, value):
        if candidate in df.columns:
            return candidate
    return None


def _multivalue_membership_column_candidates(name: str, value: Any) -> list[str]:
    tokens = _value_tokens(value)
    candidates: list[str] = []
    for token in tokens:
        candidates.extend(
            [
                f"{name}={token}",
                f"{name}[{token}]",
                f"{name}__{token}",
            ]
        )
    return list(dict.fromkeys(candidates))


def _coerce_membership_column(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    if not ((values >= NO_MEMBERSHIP).all() and (values <= FULL_MEMBERSHIP).all()):
        raise ValueError(
            f"MultiValueLiteral '{label}' values must be within [0, 1]. "
            f"Observed range: [{values.min():.4f}, {values.max():.4f}]"
        )
    return values


def _is_value_collection(value: Any) -> bool:
    return isinstance(value, (frozenset, tuple)) and not isinstance(value, str)


def _value_set(value: Any) -> frozenset[Any]:
    if _is_value_collection(value):
        return frozenset(value)
    return frozenset([value])


def _sorted_values(values: Any) -> list[Any]:
    def _sort_key(value: Any) -> tuple:
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    return sorted(list(values), key=_sort_key)


def _value_tokens(value: Any) -> list[str]:
    tokens = [str(value)]
    if isinstance(value, float) and value.is_integer():
        tokens.append(str(int(value)))
    return list(dict.fromkeys(tokens))


def parse_conjunction(text: str) -> list[SetLiteral | MultiValueLiteral]:
    """Parse a conjunction label into literal objects."""
    if not text.strip():
        raise ValueError("Conjunction string cannot be empty.")
    return [parse_literal(part) for part in text.split("*")]
