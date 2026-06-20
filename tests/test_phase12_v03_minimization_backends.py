"""Phase 12: README Roadmap v0.3 minimization backend acceptance tests."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from qca import MixedQCA
from qca.minimizers import (
    Implicant,
    MinimizerBenchmarkResult,
    QMSolution,
    available_minimizers,
    benchmark_minimizers,
    exact_set_cover,
    get_minimizer_backend,
    greedy_set_cover,
    normalize_minimizer_backend,
)


@pytest.fixture
def backend_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 9)],
            "A": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "C": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "region": [
                "north",
                "north",
                "south",
                "south",
                "north",
                "south",
                "north",
                "south",
            ],
            "Y": [1.0, 0.9, 0.8, 0.2, 0.7, 0.1, 0.2, 0.0],
        }
    )


@pytest.fixture
def backend_model(backend_df) -> MixedQCA:
    return MixedQCA(
        data=backend_df,
        case_id="case",
        outcome="Y",
        conditions=["A", "B", "C", "region"],
        condition_types={
            "A": "crisp",
            "B": "crisp",
            "C": "crisp",
            "region": "multi",
        },
    )


def test_v03_minimizer_registry_contains_roadmap_backends() -> None:
    assert available_minimizers() == (
        "standard",
        "qmc",
        "set_cover",
        "greedy_set_cover",
        "exact_set_cover",
    )
    assert normalize_minimizer_backend("set-cover") == "set_cover"
    assert normalize_minimizer_backend("greedy") == "greedy_set_cover"
    assert normalize_minimizer_backend("exact") == "exact_set_cover"
    assert get_minimizer_backend("exact").exact is True


def test_v03_set_cover_helpers_choose_minimum_cover() -> None:
    candidates = [
        Implicant(pattern=(1, 0), covered=frozenset({0})),
        Implicant(pattern=(0, 1), covered=frozenset({1})),
        Implicant(pattern=(1, "*"), covered=frozenset({0, 1})),
    ]

    assert greedy_set_cover(candidates, [0, 1]) == [candidates[2]]
    assert exact_set_cover(candidates, [0, 1]) == [candidates[2]]


def test_v03_exact_set_cover_prefers_lower_complexity_on_tie() -> None:
    candidates = [
        Implicant(pattern=(1, 0), covered=frozenset({0, 1})),
        Implicant(pattern=(1, "*"), covered=frozenset({0, 1})),
    ]

    assert exact_set_cover(candidates, [0, 1]) == [candidates[1]]


@pytest.mark.parametrize(
    "minimizer",
    ["standard", "qmc", "set_cover", "greedy_set_cover", "exact_set_cover"],
)
def test_v03_fit_accepts_all_minimizer_backends(
    backend_model,
    minimizer: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = backend_model.fit(
            minimizer=minimizer,
            outcome_threshold=0.7,
            consistency_cutoff=0.6,
            coverage_cutoff=0.0,
        )

    assert isinstance(result.qm_solution, QMSolution)
    assert result.settings["minimizer"] == minimizer
    assert result.qm_solution.backend == minimizer
    assert isinstance(result.solutions, pd.DataFrame)
    assert isinstance(result.truth_table, pd.DataFrame)


def test_v03_set_cover_aliases_run_through_fit(backend_model) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = backend_model.fit(
            minimizer="set-covering",
            outcome_threshold=0.7,
            consistency_cutoff=0.6,
            coverage_cutoff=0.0,
        )

    assert result.settings["minimizer"] == "set_cover"
    assert result.qm_solution.backend == "set_cover"


def test_v03_unknown_minimizer_still_rejected(backend_model) -> None:
    with pytest.raises(ValueError, match="Unsupported minimizer"):
        backend_model.fit(minimizer="simulated_annealing")


def test_v03_benchmark_minimizers_returns_records_and_summary(
    backend_model,
) -> None:
    result = benchmark_minimizers(
        backend_model,
        minimizers=["standard", "greedy_set_cover"],
        repeats=1,
        outcome_threshold=0.7,
        consistency_cutoff=0.6,
        coverage_cutoff=0.0,
    )

    assert isinstance(result, MinimizerBenchmarkResult)
    assert set(result.records["minimizer"]) == {"standard", "greedy_set_cover"}
    assert (result.records["elapsed_seconds"] >= 0.0).all()
    assert {"minimizer", "mean_elapsed_seconds", "runs"}.issubset(
        result.summary().columns
    )
