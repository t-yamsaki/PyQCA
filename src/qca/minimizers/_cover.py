"""Shared exact cover selection, independent of minimizer engines."""

from __future__ import annotations

from qca.minimizers.implicant import Implicant


def exact_set_cover(
    candidates: list[Implicant],
    target_minterms: list[int],
) -> list[Implicant]:
    """Return an exact minimum set cover using branch-and-bound search.

    The optimization criterion is lexicographic:

    1. fewest implicants;
    2. lowest total logical complexity;
    3. deterministic original candidate order.
    """
    target = frozenset(target_minterms)
    if not target:
        return []

    useful = [(idx, imp) for idx, imp in enumerate(candidates) if imp.covered & target]
    coverers: dict[int, list[int]] = {m: [] for m in target}
    for idx, imp in useful:
        for minterm in imp.covered & target:
            coverers[minterm].append(idx)

    if any(not indices for indices in coverers.values()):
        missing = [m for m, indices in coverers.items() if not indices]
        raise ValueError(f"Could not cover minterms: {sorted(missing)}.")

    for indices in coverers.values():
        indices.sort(
            key=lambda i: (
                -len(candidates[i].covered & target),
                candidates[i].complexity(),
                i,
            )
        )

    best_indices: tuple[int, ...] | None = None
    best_score: tuple[int, int, tuple[int, ...]] | None = None
    seen: dict[frozenset[int], tuple[int, int]] = {}

    def score(indices: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        return (
            len(indices),
            sum(candidates[i].complexity() for i in indices),
            indices,
        )

    def search(selected: tuple[int, ...], uncovered: frozenset[int]) -> None:
        nonlocal best_indices, best_score

        if not uncovered:
            candidate_score = score(tuple(sorted(selected)))
            if best_score is None or candidate_score < best_score:
                best_indices = candidate_score[2]
                best_score = candidate_score
            return

        if best_score is not None and len(selected) >= best_score[0]:
            return

        selected_complexity = sum(candidates[i].complexity() for i in selected)
        current_prefix_score = (len(selected), selected_complexity)
        previous_best = seen.get(uncovered)
        if previous_best is not None and previous_best < current_prefix_score:
            return
        # Equal-cost prefixes may produce a better candidate-order tie break.
        seen[uncovered] = current_prefix_score

        chosen_minterm = min(
            uncovered,
            key=lambda m: len([i for i in coverers[m] if i not in selected]),
        )
        for idx in coverers[chosen_minterm]:
            if idx in selected:
                continue
            new_uncovered = uncovered - set(candidates[idx].covered)
            if new_uncovered == uncovered:
                continue
            search((*selected, idx), frozenset(new_uncovered))

    # Every cover must contain candidates uniquely covering a target.
    essential = tuple(
        sorted({indices[0] for indices in coverers.values() if len(indices) == 1})
    )
    essential_coverage = frozenset().union(*(candidates[i].covered for i in essential))
    search(essential, target - essential_coverage)

    if best_indices is None:
        raise RuntimeError("Exact cover search failed for a coverable target.")
    return [candidates[idx] for idx in best_indices]
