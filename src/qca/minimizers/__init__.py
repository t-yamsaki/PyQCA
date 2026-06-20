"""Pluggable minimization backends."""

from qca.minimizers.algorithms import (
    build_coverage_table,
    can_merge,
    deduplicate_implicants,
    quine_mccluskey,
    select_essential_prime_implicants,
)
from qca.minimizers.backends import (
    MinimizerBackend,
    available_minimizers,
    get_minimizer_backend,
    normalize_minimizer_backend,
)
from qca.minimizers.benchmark import (
    MinimizerBenchmarkResult,
    benchmark_minimizers,
)
from qca.minimizers.engine import QCAMinimizer, minimize_truth_table
from qca.minimizers.implicant import Implicant, QMSolution
from qca.minimizers.remainder import (
    classify_truth_table_rows,
    filter_safe_remainders,
    filter_theory_consistent_remainders,
    generate_all_minterms,
    is_theory_consistent,
    row_to_pattern,
)
from qca.minimizers.set_cover import (
    SetCoverMinimizer,
    exact_set_cover,
    greedy_set_cover,
    select_set_cover,
)

__all__ = [
    "Implicant",
    "QMSolution",
    "MinimizerBackend",
    "available_minimizers",
    "get_minimizer_backend",
    "normalize_minimizer_backend",
    "can_merge",
    "quine_mccluskey",
    "deduplicate_implicants",
    "select_essential_prime_implicants",
    "greedy_set_cover",
    "exact_set_cover",
    "select_set_cover",
    "build_coverage_table",
    "generate_all_minterms",
    "row_to_pattern",
    "classify_truth_table_rows",
    "filter_safe_remainders",
    "filter_theory_consistent_remainders",
    "is_theory_consistent",
    "QCAMinimizer",
    "SetCoverMinimizer",
    "minimize_truth_table",
    "MinimizerBenchmarkResult",
    "benchmark_minimizers",
]
