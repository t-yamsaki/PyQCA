"""Shared core objects used by PyQCA engines and minimizers."""

from qca.core.conditions import (
    ConditionKind,
    ConditionSpec,
    condition_specs_from_schema,
    normalize_condition_kind,
)
from qca.core.literals import (
    MultiValueLiteral,
    SetLiteral,
    parse_conjunction,
    parse_literal,
)
from qca.core.results import (
    NecessityResult,
    SufficiencyResult,
    TruthTableRow,
    necessity_results_to_df,
    sufficiency_results_to_df,
    truth_table_to_df,
)

__all__ = [
    "ConditionKind",
    "ConditionSpec",
    "condition_specs_from_schema",
    "normalize_condition_kind",
    "SetLiteral",
    "MultiValueLiteral",
    "parse_literal",
    "parse_conjunction",
    "SufficiencyResult",
    "NecessityResult",
    "TruthTableRow",
    "sufficiency_results_to_df",
    "necessity_results_to_df",
    "truth_table_to_df",
]
