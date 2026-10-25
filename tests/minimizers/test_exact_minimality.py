"""Check exact minimization against independent exhaustive cover enumeration."""

from itertools import combinations, product

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, MVQCA
from qca.minimizers import Implicant, exact_set_cover, get_minimizer_backend
from qca.minimizers.algorithms import quine_mccluskey, select_essential_prime_implicants


@pytest.mark.parametrize("engine", [CSQCA, MVQCA, FSQCA, GSQCA])
@pytest.mark.parametrize("backend", [None, "qmc", "set_cover", "exact_set_cover"])
def test_issue_8_fit_returns_five_terms(engine, backend):
    patterns = list(product([0, 1], repeat=4))
    positive = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15}
    data = pd.DataFrame(patterns, columns=list("ABCD"))
    data["Y"] = [int(i in positive) for i in range(16)]
    kwargs = (
        {"condition_types": dict.fromkeys("ABCD", "crisp")} if engine is GSQCA else {}
    )
    model = engine(data=data, outcome="Y", conditions=list("ABCD"), **kwargs)
    options = {} if backend is None else {"minimizer": backend}
    result = model.fit(directional_expectations=dict.fromkeys("ABCD", 1), **options)
    for kind in ("complex", "parsimonious", "intermediate"):
        terms = getattr(result.minimization, f"{kind}_solution")
        assert len(terms) == 5
        # Check actual patterns against every row, not the stored coverage sets.
        for index, row in enumerate(patterns):
            matches = any(
                all(
                    value == "*"
                    or (
                        observed in value
                        if isinstance(value, frozenset)
                        else observed == value
                    )
                    for observed, value in zip(row, term.pattern, strict=True)
                )
                for term in terms
            )
            assert matches == (index in positive)


def brute_force_cover(candidates, target):
    """Enumerate all candidate subsets; no production solver is used."""
    for count in range(len(candidates) + 1):
        valid = []
        for indices in combinations(range(len(candidates)), count):
            covered = set().union(*(candidates[i].covered for i in indices))
            if set(target) <= covered:
                complexity = sum(candidates[i].complexity() for i in indices)
                valid.append((complexity, indices))
        if valid:
            return min(valid)
    raise AssertionError("Test fixture has no cover")


def test_all_three_condition_truth_tables_match_exhaustive_search():
    patterns = list(product([0, 1], repeat=3))
    for mask in range(256):
        target = [i for i in range(8) if mask & (1 << i)]
        candidates = quine_mccluskey(target, patterns, [False] * 3)
        complexity, indices = brute_force_cover(candidates, target)
        selected = select_essential_prime_implicants(candidates, target)
        assert selected == [candidates[i] for i in indices]
        assert sum(item.complexity() for item in selected) == complexity


def test_cover_objective_with_overlaps_and_equal_cost_paths():
    # Enumerate small coverage charts independently of prime generation.
    subsets = [
        frozenset(i for i in range(3) if mask & (1 << i)) for mask in range(1, 8)
    ]
    for chart in product(subsets, repeat=3):
        candidates = [Implicant((1, "*"), covered) for covered in chart]
        if set().union(*chart) != {0, 1, 2}:
            continue
        _, indices = brute_force_cover(candidates, [0, 1, 2])
        assert exact_set_cover(candidates, [0, 1, 2]) == [
            candidates[i] for i in indices
        ]


@pytest.mark.parametrize(
    "selector", [exact_set_cover, select_essential_prime_implicants]
)
def test_exact_selection_rejects_uncoverable_targets(selector):
    with pytest.raises(ValueError, match="cover"):
        selector([Implicant((1,), frozenset({0}))], [0, 1])
    with pytest.raises(ValueError, match="cover"):
        selector([], [0])
    assert selector([], []) == []


@pytest.mark.parametrize("backend", ["standard", "qmc", "set_cover", "exact_set_cover"])
def test_exact_backend_metadata(backend):
    assert get_minimizer_backend(backend).exact
    assert not get_minimizer_backend("greedy_set_cover").exact
