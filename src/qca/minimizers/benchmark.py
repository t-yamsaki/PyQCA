"""Benchmarking utilities for minimization backends."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca.minimizers.backends import available_minimizers, normalize_minimizer_backend


@dataclass(frozen=True)
class MinimizerBenchmarkResult:
    """Tabular benchmark result for one or more minimization backends."""

    records: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Return per-backend aggregate timing and solution statistics."""
        if self.records.empty:
            return pd.DataFrame(
                columns=[
                    "minimizer",
                    "runs",
                    "mean_elapsed_seconds",
                    "min_elapsed_seconds",
                    "max_elapsed_seconds",
                    "mean_n_solution_terms",
                    "mean_consistency",
                    "mean_coverage",
                ]
            )

        return (
            self.records.groupby("minimizer", as_index=False)
            .agg(
                runs=("repeat", "count"),
                mean_elapsed_seconds=("elapsed_seconds", "mean"),
                min_elapsed_seconds=("elapsed_seconds", "min"),
                max_elapsed_seconds=("elapsed_seconds", "max"),
                mean_n_solution_terms=("n_solution_terms", "mean"),
                mean_consistency=("consistency", "mean"),
                mean_coverage=("coverage", "mean"),
            )
            .sort_values("mean_elapsed_seconds")
            .reset_index(drop=True)
        )


def benchmark_minimizers(
    qca_model: Any,
    minimizers: Sequence[str] | None = None,
    repeats: int = 1,
    suppress_warnings: bool = True,
    **fit_kwargs: Any,
) -> MinimizerBenchmarkResult:
    """Benchmark several minimizer backends against one QCA model.

    Parameters
    ----------
    qca_model
        Any PyQCA engine exposing ``fit(minimizer=...)``.
    minimizers
        Backend names or aliases. Defaults to all available backends.
    repeats
        Number of runs per backend.
    suppress_warnings
        Suppress expected minimization warnings during timing runs.
    **fit_kwargs
        Forwarded to ``qca_model.fit()``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    resolved_minimizers = [
        normalize_minimizer_backend(minimizer)
        for minimizer in (minimizers or available_minimizers())
    ]

    records: list[dict[str, Any]] = []
    for minimizer in resolved_minimizers:
        for repeat in range(1, repeats + 1):
            start = time.perf_counter()
            if suppress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = qca_model.fit(minimizer=minimizer, **fit_kwargs)
            else:
                result = qca_model.fit(minimizer=minimizer, **fit_kwargs)
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "minimizer": minimizer,
                    "repeat": repeat,
                    "elapsed_seconds": elapsed,
                    "n_solution_terms": len(result.solutions),
                    "selected_solution_type": result.selected_solution_type,
                    "consistency": result.consistency,
                    "coverage": result.coverage,
                    "formula": result.formula,
                }
            )

    return MinimizerBenchmarkResult(pd.DataFrame(records))


__all__ = ["MinimizerBenchmarkResult", "benchmark_minimizers"]
