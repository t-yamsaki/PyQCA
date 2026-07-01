"""Implicant module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from qca._constants import DONT_CARE

# ---------------------------------------------------------------------------
# Implicant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Implicant:
    """One Quine-McCluskey implicant."""

    pattern: tuple[Any, ...]
    covered: frozenset[int]
    is_prime: bool = True

    # ------------------------------------------------------------------
    # Analytical properties
    # ------------------------------------------------------------------

    def complexity(self) -> int:
        """Return the number of non-wildcard conditions."""
        return sum(1 for v in self.pattern if v != DONT_CARE)

    def covers(self, minterm_idx: int) -> bool:
        """Return whether this implicant covers a minterm index."""
        return minterm_idx in self.covered

    def subsumes(self, other: Implicant) -> bool:
        """Return whether this implicant logically subsumes another."""
        if len(self.pattern) != len(other.pattern):
            return False

        for sv, ov in zip(self.pattern, other.pattern, strict=False):
            # A concrete value in self that differs from other is not subsumed.
            if sv == DONT_CARE:
                continue
            if _is_value_collection(sv):
                if not _value_set(ov).issubset(_value_set(sv)):
                    return False
                continue
            if sv != ov:
                return False

        # Coverage must also be contained.
        return other.covered.issubset(self.covered)

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def label(self, condition_names: list[str]) -> str:
        """Return a human-readable label."""
        if len(condition_names) != len(self.pattern):
            raise ValueError(
                f"condition_names length ({len(condition_names)}) does not match "
                f"pattern length ({len(self.pattern)})."
            )

        parts: list[str] = []
        for name, val in zip(condition_names, self.pattern, strict=False):
            if val == DONT_CARE:
                continue
            if _is_value_collection(val):
                values = _sorted_values(val)
                parts.append(f"{name}={{{','.join(str(v) for v in values)}}}")
                continue
            if isinstance(val, int):
                parts.append(f"~{name}" if val == 0 else name)
            else:
                parts.append(f"{name}={val}")

        return " * ".join(parts) if parts else "(tautology)"

    def __repr__(self) -> str:
        return (
            f"Implicant("
            f"pattern={self.pattern}, "
            f"covered={set(self.covered)}, "
            f"complexity={self.complexity()}"
            f")"
        )


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


# ---------------------------------------------------------------------------
# QMSolution
# ---------------------------------------------------------------------------


@dataclass
class QMSolution:
    """Complete Quine-McCluskey solution bundle."""

    complex_solution: list[Implicant] = field(default_factory=list)
    parsimonious_solution: list[Implicant] = field(default_factory=list)
    intermediate_solution: list[Implicant] = field(default_factory=list)
    prime_implicants_complex: list[Implicant] = field(default_factory=list)
    prime_implicants_parsimonious: list[Implicant] = field(default_factory=list)
    prime_implicants_intermediate: list[Implicant] = field(default_factory=list)
    coverage_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    backend: str = "qmc"
    condition_names: list[str] = field(default_factory=list)
    n_positive: int = 0
    n_remainder: int = 0
    n_contradiction: int = 0

    # ------------------------------------------------------------------
    # Aggregation and output
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame."""
        rows: list[dict] = []
        for sol_type, implicants in [
            ("complex", self.complex_solution),
            ("parsimonious", self.parsimonious_solution),
            ("intermediate", self.intermediate_solution),
        ]:
            for imp in implicants:
                rows.append(
                    {
                        "solution_type": sol_type,
                        "term": imp.label(self.condition_names),
                        "complexity": imp.complexity(),
                        "n_minterms_covered": len(imp.covered),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "solution_type",
                    "term",
                    "complexity",
                    "n_minterms_covered",
                ]
            )

        order = {"complex": 0, "parsimonious": 1, "intermediate": 2}
        df = pd.DataFrame(rows)
        df["_order"] = df["solution_type"].map(order)
        return (
            df.sort_values(["_order", "complexity"])
            .drop(columns=["_order"])
            .reset_index(drop=True)
        )

    def complexity_reduction(self) -> dict[str, int]:
        """Return each solution's complexity reduction from the complex solution."""
        complex_total = sum(imp.complexity() for imp in self.complex_solution)
        parsimonious_total = sum(imp.complexity() for imp in self.parsimonious_solution)
        intermediate_total = sum(imp.complexity() for imp in self.intermediate_solution)
        return {
            "parsimonious": complex_total - parsimonious_total,
            "intermediate": complex_total - intermediate_total,
        }

    def __repr__(self) -> str:
        return (
            f"QMSolution("
            f"n_positive={self.n_positive}, "
            f"n_remainder={self.n_remainder}, "
            f"complex={len(self.complex_solution)} terms, "
            f"parsimonious={len(self.parsimonious_solution)} terms, "
            f"intermediate={len(self.intermediate_solution)} terms"
            f")"
        )
