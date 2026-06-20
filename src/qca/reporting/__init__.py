"""Reporting helpers for PyQCA."""

from qca.reporting._formatters import (
    consistency_label,
    coverage_label,
    display_width,
    format_key_value,
    format_table_row,
    header,
    horizontal_rule,
    indent,
    ljust_display,
    necessity_label,
    rjust_display,
    section,
    strength_bar,
    subheader,
    truncate_display,
)
from qca.reporting.export import generate_markdown_report, to_latex_table
from qca.reporting.jupyter import JupyterSummary, jupyter_summary
from qca.reporting.report import generate_report
from qca.reporting.reproducibility import (
    ExperimentLogger,
    collect_reproducibility_metadata,
    log_experiment,
)

__all__ = [
    "display_width",
    "ljust_display",
    "rjust_display",
    "truncate_display",
    "header",
    "subheader",
    "section",
    "indent",
    "horizontal_rule",
    "consistency_label",
    "coverage_label",
    "necessity_label",
    "strength_bar",
    "format_table_row",
    "format_key_value",
    "generate_report",
    "generate_markdown_report",
    "to_latex_table",
    "collect_reproducibility_metadata",
    "ExperimentLogger",
    "log_experiment",
    "JupyterSummary",
    "jupyter_summary",
]
