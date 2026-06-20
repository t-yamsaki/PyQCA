"""Algorithms module."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from qca._constants import DONT_CARE
from qca.minimizers.implicant import Implicant

# ---------------------------------------------------------------------------
# Merge eligibility
# ---------------------------------------------------------------------------


def can_merge(
    p1: tuple[Any, ...],
    p2: tuple[Any, ...],
    is_multivalue: list[bool],
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

    # Exactly one difference, located at a binary condition.
    if len(diff_positions) == 1:
        pos = diff_positions[0]
        if not is_multivalue[pos]:
            merged = list(p1)
            merged[pos] = DONT_CARE
            return True, tuple(merged)

    return False, None


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

                ok, merged_pat = can_merge(pat1, pat2, is_multivalue)
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
    """Select essential prime implicants for a minimum cover."""
    if not prime_implicants or not target_minterms:
        return []

    target_set = set(target_minterms)
    selected: list[Implicant] = []
    selected_set: set[int] = set()  # Selected indices, used to prevent duplicates.
    covered: set[int] = set()

    # Step 1: Identify essential prime implicants.
    for m in target_minterms:
        covering = [
            (i, imp) for i, imp in enumerate(prime_implicants) if m in imp.covered
        ]
        if len(covering) == 1:
            idx, imp = covering[0]
            if idx not in selected_set:
                selected.append(imp)
                selected_set.add(idx)
                covered |= imp.covered

    # Step 2: Complete coverage greedily.
    remaining = target_set - covered
    while remaining:
        # Prefer maximum uncovered coverage, then minimum complexity.
        best_idx, best_imp = max(
            enumerate(prime_implicants),
            key=lambda t: (
                len(t[1].covered & remaining),
                -t[1].complexity(),  # Lower complexity is preferred.
            ),
        )

        if not (best_imp.covered & remaining):
            # Uncoverable minterms remain, for example after contradictory rows.
            warnings.warn(
                f"Unable to cover minterms: {sorted(remaining)}. "
                "Check the truth table for contradictory rows.",
                UserWarning,
                stacklevel=3,
            )
            break

        if best_idx not in selected_set:
            selected.append(best_imp)
            selected_set.add(best_idx)
        covered |= best_imp.covered
        remaining = target_set - covered

    # Preserve the ordering of the original list.
    order = {id(imp): i for i, imp in enumerate(prime_implicants)}
    selected.sort(key=lambda imp: order.get(id(imp), 0))

    return selected


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
