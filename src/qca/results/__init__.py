"""Result objects and export helpers."""

from qca.core.results import (
    NecessityResult,
    SufficiencyResult,
    TruthTableRow,
    necessity_results_to_df,
    sufficiency_results_to_df,
    truth_table_to_df,
)
from qca.results.solution import QCAFitResult

__all__ = [
    "QCAFitResult",
    "SufficiencyResult",
    "NecessityResult",
    "TruthTableRow",
    "sufficiency_results_to_df",
    "necessity_results_to_df",
    "truth_table_to_df",
]
