"""Algorithms module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qca._constants import DONT_CARE
from qca.minimizers._cover import exact_set_cover
from qca.minimizers.implicant import Implicant

# ---------------------------------------------------------------------------
# Merge eligibility
# ---------------------------------------------------------------------------


def can_merge(
    p1: tuple[Any, ...],
    p2: tuple[Any, ...],
    is_multivalue: list[bool],
    condition_domains: list[list[Any]] | None = None,
) -> tuple[bool, tuple[Any, ...] | None]:
    """Return whether two patterns can be merged by the QM algorithm."""
    if len(p1) != len(p2):
        return False, None

    diff_positions: list[int] = []

    for i, (v1, v2) in enumerate(zip(p1, p2, strict=False)):
        # Different wildcard states cannot be merged.
        if (v1 == DONT_CARE) != (v2 == DONT_CARE):
            return False, None
        if v1 != v2:
            diff_positions.append(i)

    # Exactly one difference, located at a binary or multi-value condition.
    if len(diff_positions) == 1:
        pos = diff_positions[0]
        if not is_multivalue[pos]:
            merged = list(p1)
            merged[pos] = DONT_CARE
            return True, tuple(merged)
        if condition_domains is not None:
            merged_value = _merge_multivalue_position(
                p1[pos],
                p2[pos],
                condition_domains[pos],
            )
            if merged_value is not None:
                merged = list(p1)
                merged[pos] = merged_value
                return True, tuple(merged)

    return False, None


def _merge_multivalue_position(
    left: Any,
    right: Any,
    domain: list[Any],
) -> Any | None:
    """Merge one multi-value position without overgeneralizing the domain."""
    if left == DONT_CARE or right == DONT_CARE:
        return None

    left_values = _pattern_value_set(left)
    right_values = _pattern_value_set(right)
    merged_values = left_values | right_values
    if merged_values in (left_values, right_values):
        return None

    domain_values = frozenset(domain)
    if not merged_values.issubset(domain_values):
        return None
    if merged_values == domain_values:
        return DONT_CARE
    return frozenset(_sort_values(merged_values))


def _pattern_value_set(value: Any) -> frozenset[Any]:
    if isinstance(value, frozenset):
        return value
    if isinstance(value, tuple):
        return frozenset(value)
    return frozenset([value])


def _sort_values(values: frozenset[Any]) -> list[Any]:
    def _sort_key(value: Any) -> tuple:
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    return sorted(values, key=_sort_key)


# ---------------------------------------------------------------------------
# Quine-McCluskey core
# ---------------------------------------------------------------------------


def quine_mccluskey(
    minterms: list[int],
    all_patterns: list[tuple[Any, ...]],
    is_multivalue: list[bool],
    dont_care_indices: list[int] | None = None,
) -> list[Implicant]:
    """Quine mccluskey."""
    if not minterms:
        return []

    dont_care_set: frozenset[int] = frozenset(dont_care_indices or [])
    target_set: frozenset[int] = frozenset(minterms)
    condition_domains = _domains_by_position(all_patterns)

    # Generate initial implicants.
    # Key: frozenset of covered minterm indices.
    # Value: pattern tuple.
    initial_indices = sorted(target_set | dont_care_set)
    current: dict[frozenset[int], tuple[Any, ...]] = {
        frozenset([i]): all_patterns[i] for i in initial_indices
    }

    all_prime_candidates: list[Implicant] = []

    while current:
        next_level: dict[frozenset[int], tuple[Any, ...]] = {}
        merged_keys: set[frozenset[int]] = set()

        items = list(current.items())
        n = len(items)

        for i in range(n):
            for j in range(i + 1, n):
                cov1, pat1 = items[i]
                cov2, pat2 = items[j]

                ok, merged_pat = can_merge(
                    pat1,
                    pat2,
                    is_multivalue,
                    condition_domains=condition_domains,
                )
                if ok and merged_pat is not None:
                    new_cov = cov1 | cov2
                    # Replace an existing entry with the same coverage.
                    next_level[new_cov] = merged_pat
                    merged_keys.add(cov1)
                    merged_keys.add(cov2)

        # Unmerged implicants are prime-implicant candidates.
        for cov, pat in items:
            if cov not in merged_keys:
                # Retain candidates that cover at least one positive row.
                positive_covered = cov & target_set
                if positive_covered:
                    all_prime_candidates.append(
                        Implicant(
                            pattern=pat,
                            covered=frozenset(positive_covered),
                            is_prime=True,
                        )
                    )

        current = next_level

    return deduplicate_implicants(all_prime_candidates)


def _domains_by_position(all_patterns: list[tuple[Any, ...]]) -> list[list[Any]]:
    """Return concrete condition domains inferred from minterm patterns."""
    if not all_patterns:
        return []
    n_positions = len(all_patterns[0])
    domains: list[list[Any]] = []
    for pos in range(n_positions):
        values = frozenset(pattern[pos] for pattern in all_patterns)
        domains.append(_sort_values(values))
    return domains


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def deduplicate_implicants(
    implicants: list[Implicant],
) -> list[Implicant]:
    """Merge implicants with identical patterns."""
    seen: dict[tuple[Any, ...], frozenset[int]] = {}
    for imp in implicants:
        seen[imp.pattern] = seen.get(imp.pattern, frozenset()) | imp.covered

    return [
        Implicant(pattern=pat, covered=cov, is_prime=True) for pat, cov in seen.items()
    ]


# ---------------------------------------------------------------------------
# Essential prime-implicant selection
# ---------------------------------------------------------------------------


def select_essential_prime_implicants(
    prime_implicants: list[Implicant],
    target_minterms: list[int],
) -> list[Implicant]:
    """Select one exact minimum cover, including all essential implicants.

    Minimize term count, then total non-wildcard condition count, then
    candidate-index order. Raise ValueError if any target cannot be covered.
    """
    return exact_set_cover(prime_implicants, target_minterms)


# ---------------------------------------------------------------------------
# Coverage-table construction
# ---------------------------------------------------------------------------


def build_coverage_table(
    prime_implicants: list[Implicant],
    target_minterms: list[int],
    condition_names: list[str],
) -> pd.DataFrame:
    """Build the implicant-by-minterm coverage table."""
    if not prime_implicants or not target_minterms:
        return pd.DataFrame()

    labels = [imp.label(condition_names) for imp in prime_implicants]
    columns = [f"m{i}" for i in target_minterms]

    data = np.zeros((len(prime_implicants), len(target_minterms)), dtype=int)
    for i, imp in enumerate(prime_implicants):
        for j, m in enumerate(target_minterms):
            if m in imp.covered:
                data[i, j] = 1

    df = pd.DataFrame(data, index=labels, columns=columns)
    df.index.name = "implicant"
    df["complexity"] = [imp.complexity() for imp in prime_implicants]
    df["n_covered"] = [len(imp.covered) for imp in prime_implicants]

    return df
