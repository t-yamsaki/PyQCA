"""Set-covering minimization backends for PyQCA.

The Quine-McCluskey pass still generates prime implicant candidates. This
module changes the covering step: choose a subset of candidates that covers all
positive minterms with either a greedy heuristic or an exact branch-and-bound
search.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from qca._types import ConditionDomains, IsMultiValueMap, MintermList
from qca.minimizers._cover import exact_set_cover
from qca.minimizers.algorithms import (
    build_coverage_table,
    quine_mccluskey,
)
from qca.minimizers.engine import QCAMinimizer
from qca.minimizers.implicant import Implicant, QMSolution
from qca.minimizers.remainder import (
    filter_theory_consistent_remainders,
)

SetCoverMethod = Literal["greedy", "exact"]


def greedy_set_cover(
    candidates: list[Implicant],
    target_minterms: list[int],
) -> list[Implicant]:
    """Return a deterministic greedy set cover over implicant candidates."""
    target = set(target_minterms)
    remaining = set(target)
    selected_indices: list[int] = []
    available = set(range(len(candidates)))

    while remaining:
        ranked: list[tuple[int, int, int, int]] = []
        for idx in available:
            imp = candidates[idx]
            newly_covered = len(imp.covered & remaining)
            if newly_covered <= 0:
                continue
            ranked.append(
                (
                    newly_covered,
                    -imp.complexity(),
                    len(imp.covered & target),
                    -idx,
                )
            )

        if not ranked:
            warnings.warn(
                f"Could not cover minterms: {sorted(remaining)}.",
                UserWarning,
                stacklevel=2,
            )
            break

        best_score = max(ranked)
        best_idx = -best_score[3]
        selected_indices.append(best_idx)
        available.remove(best_idx)
        remaining -= set(candidates[best_idx].covered)

    selected_indices.sort()
    return [candidates[idx] for idx in selected_indices]


def select_set_cover(
    candidates: list[Implicant],
    target_minterms: list[int],
    method: SetCoverMethod = "exact",
) -> list[Implicant]:
    """Select implicants with a set-cover backend."""
    if method == "greedy":
        return greedy_set_cover(candidates, target_minterms)
    if method == "exact":
        return exact_set_cover(candidates, target_minterms)
    raise ValueError(
        f"Unknown set-cover method {method!r}. Available methods: greedy, exact."
    )


class SetCoverMinimizer(QCAMinimizer):
    """QCA minimizer using set-covering for implicant selection."""

    def __init__(
        self,
        condition_names: list[str],
        is_multivalue: IsMultiValueMap,
        condition_domains: ConditionDomains,
        cover_method: SetCoverMethod = "exact",
        backend_name: str | None = None,
    ) -> None:
        if cover_method not in {"greedy", "exact"}:
            raise ValueError(
                "cover_method must be either 'greedy' or 'exact'. "
                f"Got {cover_method!r}."
            )
        self.cover_method = cover_method
        resolved_backend = (
            backend_name if backend_name is not None else f"{cover_method}_set_cover"
        )
        super().__init__(
            condition_names=condition_names,
            is_multivalue=is_multivalue,
            condition_domains=condition_domains,
            backend_name=resolved_backend,
        )

    def _select_prime_implicants(
        self,
        prime_implicants: list[Implicant],
        positive_indices: MintermList,
    ) -> list[Implicant]:
        return select_set_cover(
            prime_implicants,
            positive_indices,
            method=self.cover_method,
        )

    def _build_parsimonious_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
        remainder_indices: MintermList,
        negative_indices: MintermList,
        contradiction_indices: MintermList,
        include_remainders: bool,
    ) -> QMSolution:
        solution = super()._build_parsimonious_solution(
            solution=solution,
            positive_indices=positive_indices,
            remainder_indices=remainder_indices,
            negative_indices=negative_indices,
            contradiction_indices=contradiction_indices,
            include_remainders=include_remainders,
        )
        if include_remainders:
            solution.coverage_table = build_coverage_table(
                solution.prime_implicants_parsimonious,
                positive_indices,
                self.condition_names,
            )
        return solution

    def _build_solution_from_dont_cares(
        self,
        positive_indices: MintermList,
        dont_care_indices: MintermList | None,
    ) -> tuple[list[Implicant], list[Implicant]]:
        prime_implicants = quine_mccluskey(
            minterms=positive_indices,
            all_patterns=self._all_patterns,
            is_multivalue=self.is_multivalue_flags,
            dont_care_indices=dont_care_indices,
        )
        selected = self._select_prime_implicants(
            prime_implicants,
            positive_indices,
        )
        return prime_implicants, selected

    def _build_complex_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
    ) -> QMSolution:
        prime_implicants, selected = self._build_solution_from_dont_cares(
            positive_indices=positive_indices,
            dont_care_indices=None,
        )
        solution.prime_implicants_complex = prime_implicants
        solution.complex_solution = selected
        return solution

    def _build_intermediate_solution(
        self,
        solution: QMSolution,
        positive_indices: MintermList,
        remainder_indices: MintermList,
        negative_indices: MintermList,
        contradiction_indices: MintermList,
        directional_expectations: dict[str, Any] | None,
    ) -> QMSolution:
        if directional_expectations is None:
            return super()._build_intermediate_solution(
                solution=solution,
                positive_indices=positive_indices,
                remainder_indices=remainder_indices,
                negative_indices=negative_indices,
                contradiction_indices=contradiction_indices,
                directional_expectations=directional_expectations,
            )

        theory_remainders = filter_theory_consistent_remainders(
            remainder_indices=remainder_indices,
            negative_indices=negative_indices,
            contradiction_indices=contradiction_indices,
            all_patterns=self._all_patterns,
            condition_names=self.condition_names,
            directional_expectations=directional_expectations,
        )
        prime_implicants, selected = self._build_solution_from_dont_cares(
            positive_indices=positive_indices,
            dont_care_indices=theory_remainders,
        )
        solution.prime_implicants_intermediate = prime_implicants
        solution.intermediate_solution = selected
        return solution


__all__ = [
    "SetCoverMethod",
    "SetCoverMinimizer",
    "greedy_set_cover",
    "exact_set_cover",
    "select_set_cover",
]
