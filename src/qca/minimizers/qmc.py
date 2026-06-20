"""Quine-McCluskey minimization backend."""

from qca.minimizers.algorithms import (
    build_coverage_table,
    can_merge,
    deduplicate_implicants,
    quine_mccluskey,
    select_essential_prime_implicants,
)
from qca.minimizers.engine import QCAMinimizer, minimize_truth_table
from qca.minimizers.implicant import Implicant, QMSolution

__all__ = [
    "Implicant",
    "QMSolution",
    "QCAMinimizer",
    "minimize_truth_table",
    "can_merge",
    "quine_mccluskey",
    "deduplicate_implicants",
    "select_essential_prime_implicants",
    "build_coverage_table",
]
