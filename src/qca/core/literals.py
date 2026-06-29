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

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Return a human-readable label."""
        return f"{self.name}={self.value}"

    def __repr__(self) -> str:
        return f"MultiValueLiteral({self.name!r}, {self.value!r})"

    # ------------------------------------------------------------------
    # Membership calculation
    # ------------------------------------------------------------------

    def membership(self, df: pd.DataFrame) -> pd.Series:
        """Return membership scores for this literal."""
        if self.name not in df.columns:
            raise KeyError(
                f"MultiValueLiteral '{self.name}' is not present in the DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        col = df[self.name]

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
        return self.name == other.name and self.value != other.value


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
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def parse_conjunction(text: str) -> list[SetLiteral | MultiValueLiteral]:
    """Parse a conjunction label into literal objects."""
    if not text.strip():
        raise ValueError("Conjunction string cannot be empty.")
    return [parse_literal(part) for part in text.split("*")]
