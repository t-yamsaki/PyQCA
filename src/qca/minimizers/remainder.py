"""Logical-remainder utilities for QCA minimization."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import product
from typing import Any

from qca._constants import (
    DONT_CARE,
)
from qca._types import (
    ConditionDomains,
    DirectionalExpectations,
    MintermList,
    TruthTableRowProtocol,
)
from qca._validation import validate_category_value

# ---------------------------------------------------------------------------
# Full logical-space generation
# ---------------------------------------------------------------------------


def generate_all_minterms(
    condition_names: list[str],
    condition_domains: ConditionDomains,
) -> list[tuple[Any, ...]]:
    """Enumerate the full logical space."""
    if not condition_names:
        return [()]

    domains = [condition_domains[c] for c in condition_names]
    for domain in domains:
        for value in domain:
            validate_category_value(value)
    return list(product(*domains))


# ---------------------------------------------------------------------------
# Truth-table row conversion
# ---------------------------------------------------------------------------


def row_to_pattern(
    config: dict[str, Any],
    condition_names: list[str],
    is_multivalue_flags: list[bool],
) -> tuple[Any, ...] | None:
    """Row to pattern."""
    pattern: list[Any] = []
    for c, is_mv in zip(condition_names, is_multivalue_flags, strict=False):
        if c not in config:
            return None
        val = config[c]
        if not is_mv:
            val = int(val)
        pattern.append(val)
    return tuple(pattern)


# ---------------------------------------------------------------------------
# Truth-table row classification
# ---------------------------------------------------------------------------


def classify_truth_table_rows(
    truth_table_rows: list[TruthTableRowProtocol],
    all_patterns: list[tuple[Any, ...]],
    pattern_to_idx: dict[tuple[Any, ...], int],
    condition_names: list[str],
    is_multivalue_flags: list[bool],
    outcome_threshold: float,
    consistency_threshold: float | None,
    pri_cut: float = 0.0,
    frequency_cutoff: int = 1,
) -> tuple[MintermList, MintermList, MintermList, MintermList]:
    """Classify truth-table rows as positive, negative, remainder, or contradiction."""
    resolved_consistency_threshold = (
        outcome_threshold if consistency_threshold is None else consistency_threshold
    )
    observed_patterns: set[tuple[Any, ...]] = set()
    positive_set: set[int] = set()
    negative_set: set[int] = set()
    contradiction_set: set[int] = set()

    for row in truth_table_rows:
        if getattr(row, "n_cases", 1) < frequency_cutoff:
            continue

        pat = row_to_pattern(row.config, condition_names, is_multivalue_flags)
        if pat is None:
            warnings.warn(
                f"Could not convert truth-table configuration {row.config} to a "
                "known pattern; skipping it. Check condition_names and "
                "is_multivalue_flags.",
                UserWarning,
                stacklevel=3,
            )
            continue

        idx = pattern_to_idx.get(pat)
        if idx is None:
            warnings.warn(
                f"Pattern {pat} is not present in the full logical space. "
                "Check condition_domains.",
                UserWarning,
                stacklevel=3,
            )
            continue

        observed_patterns.add(pat)

        row_pri = getattr(row, "pri_consistency", None)
        passes_pri = row_pri is None or row_pri >= pri_cut
        is_positive = (
            row.outcome_raw_consist >= resolved_consistency_threshold
            and passes_pri
        )

        # Detect indices that have already received the opposite classification.
        if (
            is_positive
            and idx in negative_set
            or not is_positive
            and idx in positive_set
        ):
            contradiction_set.add(idx)

        if is_positive:
            positive_set.add(idx)
        else:
            negative_set.add(idx)

    # Remove contradictory rows from the positive and negative sets.
    positive_indices = sorted(positive_set - contradiction_set)
    negative_indices = sorted(negative_set - contradiction_set)
    contradiction_indices = sorted(contradiction_set)

    # Logical remainders are unobserved rows in the full logical space.
    remainder_indices = [
        i for i, pat in enumerate(all_patterns) if pat not in observed_patterns
    ]

    if contradiction_indices:
        warnings.warn(
            f"Detected {len(contradiction_indices)} contradictory row(s); "
            "they were excluded from logical minimization. "
            "Consider reviewing the calibration."
            f"(Ragin 2008, p.128)",
            UserWarning,
            stacklevel=2,
        )

    return (
        positive_indices,
        negative_indices,
        remainder_indices,
        contradiction_indices,
    )


# ---------------------------------------------------------------------------
# Directional filtering of logical remainders
# ---------------------------------------------------------------------------


def validate_directional_expectations(
    directional_expectations: DirectionalExpectations | None,
    condition_names: list[str],
    condition_domains: ConditionDomains,
) -> None:
    """Validate explicit theory against the model's complete condition domains."""
    if directional_expectations is None:
        return
    if not isinstance(directional_expectations, Mapping):
        raise TypeError("directional_expectations must be a mapping or None.")
    for name, expected in directional_expectations.items():
        if name not in condition_names:
            raise ValueError(
                f"directional_expectations contains unknown condition {name!r}. "
                f"Available conditions: {condition_names!r}."
            )
        domain = condition_domains[name]
        try:
            valid = expected in domain
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"directional_expectations[{name!r}] has value {expected!r} "
                f"outside its condition domain {domain!r}. "
                "Omit the condition to leave it unconstrained."
            )


def is_theory_consistent(
    pattern: tuple[Any, ...],
    condition_names: list[str],
    directional_expectations: DirectionalExpectations,
) -> bool:
    """Is theory consistent."""
    for name, val in zip(condition_names, pattern, strict=False):
        if val == DONT_CARE:
            continue
        if name not in directional_expectations:
            continue
        expected = directional_expectations[name]
        if val != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# Safe logical-remainder extraction
# ---------------------------------------------------------------------------


def filter_safe_remainders(
    remainder_indices: MintermList,
    negative_indices: MintermList,
    contradiction_indices: MintermList,
) -> MintermList:
    """Filter safe remainders."""
    excluded = set(negative_indices) | set(contradiction_indices)
    return [i for i in remainder_indices if i not in excluded]


def filter_theory_consistent_remainders(
    remainder_indices: MintermList,
    negative_indices: MintermList,
    contradiction_indices: MintermList,
    all_patterns: list[tuple[Any, ...]],
    condition_names: list[str],
    directional_expectations: DirectionalExpectations,
) -> MintermList:
    """Filter theory consistent remainders."""
    safe = filter_safe_remainders(
        remainder_indices, negative_indices, contradiction_indices
    )
    return [
        i
        for i in safe
        if is_theory_consistent(
            all_patterns[i], condition_names, directional_expectations
        )
    ]


def filter_easy_remainders(
    remainder_indices: MintermList,
    negative_indices: MintermList,
    contradiction_indices: MintermList,
    all_patterns: list[tuple[Any, ...]],
    condition_names: list[str],
    directional_expectations: DirectionalExpectations,
    positive_indices: MintermList,
    parsimonious_patterns: list[tuple[Any, ...]],
) -> MintermList:
    """Select easy counterfactuals for the chosen conservative/parsimonious pair.

    A changed coordinate must move toward its expected category, or be
    unspecified. Unchanged coordinates need not equal the expectation.
    Restrict candidates to the selected parsimonious solution, not the union
    of all alternative prime implicants (Dusa, QCA with R, section 8.7).
    """
    safe = filter_safe_remainders(
        remainder_indices, negative_indices, contradiction_indices
    )
    easy = []
    for index in safe:
        candidate = all_patterns[index]
        in_parsimonious = any(
            all(
                value == DONT_CARE
                or (
                    actual in value if isinstance(value, frozenset) else actual == value
                )
                for actual, value in zip(candidate, term, strict=True)
            )
            for term in parsimonious_patterns
        )
        if not in_parsimonious:
            continue
        if any(
            all(
                actual == observed
                or name not in directional_expectations
                or actual == directional_expectations[name]
                for name, actual, observed in zip(
                    condition_names, candidate, all_patterns[positive], strict=True
                )
            )
            for positive in positive_indices
        ):
            easy.append(index)
    return easy
