"""PyQCA public package interface."""

from qca._version import __version__
from qca.calibration import calibrate_crisp, calibrate_logistic, calibrate_piecewise
from qca.core.conditions import (
    ConditionKind,
    ConditionSpec,
    condition_specs_from_schema,
)
from qca.core.literals import (
    MultiValueLiteral,
    SetLiteral,
    parse_conjunction,
    parse_literal,
)
from qca.core.results import NecessityResult, SufficiencyResult, TruthTableRow
from qca.engines import CSQCA, FSQCA, MVQCA, MixedQCA, QCAEngineBase
from qca.minimizers import (
    Implicant,
    QCAMinimizer,
    QMSolution,
    SetCoverMinimizer,
    available_minimizers,
    benchmark_minimizers,
    exact_set_cover,
    greedy_set_cover,
    minimize_truth_table,
    select_set_cover,
)
from qca.reporting import (
    ExperimentLogger,
    JupyterSummary,
    collect_reproducibility_metadata,
    generate_markdown_report,
    generate_report,
    jupyter_summary,
    log_experiment,
    to_latex_table,
)
from qca.results import QCAFitResult
from qca.sweep import (
    AnchorSensitivity,
    AnchorSensitivityResult,
    ThresholdSweep,
    ThresholdSweepResult,
)

__all__ = [
    "__version__",
    "SetLiteral",
    "MultiValueLiteral",
    "parse_literal",
    "parse_conjunction",
    "ConditionKind",
    "ConditionSpec",
    "condition_specs_from_schema",
    "SufficiencyResult",
    "NecessityResult",
    "TruthTableRow",
    "QCAEngineBase",
    "MixedQCA",
    "CSQCA",
    "FSQCA",
    "MVQCA",
    "QCAFitResult",
    "AnchorSensitivity",
    "AnchorSensitivityResult",
    "ThresholdSweep",
    "ThresholdSweepResult",
    "calibrate_crisp",
    "calibrate_piecewise",
    "calibrate_logistic",
    "generate_report",
    "generate_markdown_report",
    "to_latex_table",
    "collect_reproducibility_metadata",
    "ExperimentLogger",
    "log_experiment",
    "JupyterSummary",
    "jupyter_summary",
    "Implicant",
    "QMSolution",
    "QCAMinimizer",
    "SetCoverMinimizer",
    "minimize_truth_table",
    "available_minimizers",
    "greedy_set_cover",
    "exact_set_cover",
    "select_set_cover",
    "benchmark_minimizers",
]
