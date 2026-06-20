"""Dataclasses for QCA analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca._constants import (
    CONSISTENCY_MODERATE,
    CONSISTENCY_NECESSITY,
    CONSISTENCY_STRONG,
    MEMBERSHIP_THRESHOLD,
)

# ---------------------------------------------------------------------------
# SufficiencyResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SufficiencyResult:
    """Result of a sufficiency analysis."""

    antecedent: str
    consistency: float
    raw_coverage: float
    unique_coverage: float | None
    n_cases_in: int

    # ------------------------------------------------------------------
    # 評価プロパティ
    # ------------------------------------------------------------------

    @property
    def is_sufficient(self) -> bool:
        """Return whether the result meets the sufficiency criteria."""
        return self.consistency >= CONSISTENCY_MODERATE and self.n_cases_in > 0

    @property
    def strength(self) -> str:
        """Return a qualitative strength label."""
        if self.n_cases_in == 0:
            return "empty"
        if self.consistency >= CONSISTENCY_STRONG:
            return "strong"
        if self.consistency >= CONSISTENCY_MODERATE:
            return "moderate"
        return "weak"

    # ------------------------------------------------------------------
    # シリアライズ
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        return {
            "antecedent": self.antecedent,
            "consistency": self.consistency,
            "raw_coverage": self.raw_coverage,
            "unique_coverage": self.unique_coverage,
            "n_cases_in": self.n_cases_in,
            "is_sufficient": self.is_sufficient,
            "strength": self.strength,
        }

    def to_series(self) -> pd.Series:
        """Return a pandas Series representation."""
        return pd.Series(self.to_dict())

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        uc_str = (
            f"{self.unique_coverage:.3f}" if self.unique_coverage is not None else "N/A"
        )
        return (
            f"SufficiencyResult(\n"
            f"  antecedent    = {self.antecedent}\n"
            f"  consistency   = {self.consistency:.3f}  [{self.strength}]\n"
            f"  raw_coverage  = {self.raw_coverage:.3f}\n"
            f"  unique_cov    = {uc_str}\n"
            f"  n_cases_in    = {self.n_cases_in}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# NecessityResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NecessityResult:
    """Result of a necessity analysis."""

    condition: str
    consistency: float
    coverage: float
    n_cases_in: int

    # ------------------------------------------------------------------
    # 評価プロパティ
    # ------------------------------------------------------------------

    @property
    def is_necessary(self) -> bool:
        """Is necessary."""
        return self.consistency >= CONSISTENCY_NECESSITY

    @property
    def is_trivial(self) -> bool:
        """Is trivial."""
        return self.coverage >= 0.90

    @property
    def strength(self) -> str:
        """Return a qualitative strength label."""
        if self.consistency >= 0.95:
            return "strong"
        if self.consistency >= CONSISTENCY_NECESSITY:
            return "necessary"
        if self.consistency >= CONSISTENCY_MODERATE:
            return "weak"
        return "insufficient"

    # ------------------------------------------------------------------
    # シリアライズ
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        return {
            "condition": self.condition,
            "consistency": self.consistency,
            "coverage": self.coverage,
            "n_cases_in": self.n_cases_in,
            "is_necessary": self.is_necessary,
            "is_trivial": self.is_trivial,
            "strength": self.strength,
        }

    def to_series(self) -> pd.Series:
        """Return a pandas Series representation."""
        return pd.Series(self.to_dict())

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        trivial_note = (
            "  ⚠ トリビアルな必要条件の可能性あり\n" if self.is_trivial else ""
        )
        return (
            f"NecessityResult(\n"
            f"  condition   = {self.condition}\n"
            f"  consistency = {self.consistency:.3f}  [{self.strength}]\n"
            f"  coverage    = {self.coverage:.3f}\n"
            f"  n_cases_in  = {self.n_cases_in}\n"
            f"{trivial_note}"
            f")"
        )


# ---------------------------------------------------------------------------
# TruthTableRow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthTableRow:
    """One row in a QCA truth table."""

    config: dict[str, Any]
    n_cases: int
    outcome_mean: float
    outcome_raw_consist: float
    case_ids: list[str]
    include: bool | None = None

    # ------------------------------------------------------------------
    # TruthTableRowProtocol との互換性確認
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the dataclass after initialization."""
        if not (0.0 <= self.outcome_mean <= 1.0):
            raise ValueError(
                f"outcome_mean は [0, 1] の範囲内である必要があります。"
                f"得られた値: {self.outcome_mean}"
            )
        if not (0.0 <= self.outcome_raw_consist <= 1.0):
            raise ValueError(
                f"outcome_raw_consist は [0, 1] の範囲内である必要があります。"
                f"得られた値: {self.outcome_raw_consist}"
            )
        if self.n_cases < 0:
            raise ValueError(
                f"n_cases は 0 以上である必要があります。得られた値: {self.n_cases}"
            )

    # ------------------------------------------------------------------
    # 評価プロパティ
    # ------------------------------------------------------------------

    @property
    def is_positive(self) -> bool:
        """Return whether this row is included in the outcome."""
        return self.include is True

    @property
    def is_negative(self) -> bool:
        """Return whether this row is excluded from the outcome."""
        return self.include is False

    @property
    def is_undecided(self) -> bool:
        """Is undecided."""
        return self.include is None

    @property
    def n_cases_in_outcome(self) -> int:
        """N cases in outcome."""
        return self.n_cases if self.outcome_mean > MEMBERSHIP_THRESHOLD else 0

    # ------------------------------------------------------------------
    # シリアライズ
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        record: dict[str, Any] = {**self.config}
        record["n_cases"] = self.n_cases
        record["outcome_mean"] = round(self.outcome_mean, 4)
        record["outcome_raw_consist"] = round(self.outcome_raw_consist, 4)
        record["include"] = self.include
        record["cases"] = ", ".join(self.case_ids)
        return record

    def to_series(self) -> pd.Series:
        """Return a pandas Series representation."""
        return pd.Series(self.to_dict())

    # ------------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        cases_preview = ", ".join(self.case_ids[:5]) + (
            " ..." if len(self.case_ids) > 5 else ""
        )
        include_str = {True: "✓", False: "✗", None: "?"}[self.include]
        return (
            f"TruthTableRow(\n"
            f"  config       = {{{config_str}}}\n"
            f"  n_cases      = {self.n_cases}\n"
            f"  outcome_mean = {self.outcome_mean:.3f}\n"
            f"  row_consist  = {self.outcome_raw_consist:.3f}\n"
            f"  include      = {include_str}\n"
            f"  cases        = [{cases_preview}]\n"
            f")"
        )


# ---------------------------------------------------------------------------
# 結果集約ユーティリティ
# ---------------------------------------------------------------------------


def sufficiency_results_to_df(
    results: list[SufficiencyResult],
    depth: int | None = None,
) -> pd.DataFrame:
    """Convert sufficiency results to a DataFrame."""
    if not results:
        cols = [
            "antecedent",
            "consistency",
            "raw_coverage",
            "unique_coverage",
            "n_cases_in",
            "is_sufficient",
            "strength",
        ]
        if depth is not None:
            cols.append("depth")
        return pd.DataFrame(columns=cols)

    records = [r.to_dict() for r in results]
    if depth is not None:
        for rec in records:
            rec["depth"] = depth

    return pd.DataFrame(records)


def necessity_results_to_df(
    results: list[NecessityResult],
) -> pd.DataFrame:
    """Convert necessity results to a DataFrame."""
    if not results:
        return pd.DataFrame(
            columns=[
                "condition",
                "consistency",
                "coverage",
                "n_cases_in",
                "is_necessary",
                "is_trivial",
                "strength",
            ]
        )
    return pd.DataFrame([r.to_dict() for r in results])


def truth_table_to_df(
    rows: list[TruthTableRow],
) -> pd.DataFrame:
    """Return the truth table as a DataFrame."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in rows])
