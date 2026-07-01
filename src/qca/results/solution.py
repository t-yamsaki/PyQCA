"""Unified result objects returned by PyQCA engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from qca.core.results import SufficiencyResult, TruthTableRow, truth_table_to_df
from qca.minimizers.implicant import QMSolution

SOLUTION_TYPE_ORDER = ("complex", "parsimonious", "intermediate")


@dataclass(frozen=True)
class QCAFitResult:
    """High-level result returned by PyQCA engines.

    This wraps backend-specific minimization output while exposing a stable,
    inspectable PyQCA surface: truth table, solutions, aggregate consistency and
    coverage, case-level coverage, and export helpers.
    """

    truth_table_rows: list[TruthTableRow]
    minimization: QMSolution
    solution_metrics: dict[str, SufficiencyResult | None]
    case_coverage: pd.DataFrame
    condition_schema: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: dict[str, Any] = field(default_factory=dict)
    selected_solution_type: str = "intermediate"
    outcome: str | None = None
    case_id: str | None = None
    qca_type: str = "GSQCA"

    @property
    def truth_table(self) -> pd.DataFrame:
        """Truth table as a DataFrame."""
        return truth_table_to_df(self.truth_table_rows)

    @property
    def solutions(self) -> pd.DataFrame:
        """Solution terms with aggregate metrics for each solution type."""
        summary_df = self.minimization.summary()
        if summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "solution_type",
                    "term",
                    "complexity",
                    "n_minterms_covered",
                    "consistency",
                    "coverage",
                ]
            )

        df = summary_df.copy()
        df["consistency"] = df["solution_type"].map(
            lambda sol_type: self._metric_value(sol_type, "consistency")
        )
        df["coverage"] = df["solution_type"].map(
            lambda sol_type: self._metric_value(sol_type, "raw_coverage")
        )
        df = df[
            df["solution_type"].map(
                lambda sol_type: self.solution_metrics.get(sol_type) is not None
            )
        ]
        return df

    @property
    def consistency(self) -> float | None:
        """Consistency of the selected solution, or None if no solution exists."""
        metric = self.solution_metrics.get(self.selected_solution_type)
        return None if metric is None else metric.consistency

    @property
    def coverage(self) -> float | None:
        """Coverage of the selected solution, or None if no solution exists."""
        metric = self.solution_metrics.get(self.selected_solution_type)
        return None if metric is None else metric.raw_coverage

    @property
    def selected_solution(self) -> SufficiencyResult | None:
        """Aggregate sufficiency result for the selected solution type."""
        return self.solution_metrics.get(self.selected_solution_type)

    @property
    def qm_solution(self) -> QMSolution:
        """Compatibility alias for the backend QMC result."""
        return self.minimization

    @property
    def workflow(self) -> str | None:
        """Workflow label recorded by the engine, when available."""
        value = self.settings.get("workflow")
        return None if value is None else str(value)

    @property
    def formula(self) -> str | None:
        """Formula for the selected solution type."""
        return self.to_formula()

    @property
    def selected_formula(self) -> str | None:
        """Compatibility alias for ``formula``."""
        return self.formula

    @property
    def formulas(self) -> dict[str, str]:
        """Formula strings keyed by solution type."""
        return self.to_formulas()

    def to_formulas(self) -> dict[str, str]:
        """Return minimized formulas for all available solution types."""
        solutions = self.solutions
        if solutions.empty:
            return {}

        formulas: dict[str, str] = {}
        for solution_type in SOLUTION_TYPE_ORDER:
            terms = solutions.loc[
                solutions["solution_type"] == solution_type,
                "term",
            ].tolist()
            if terms:
                formulas[solution_type] = " + ".join(f"({term})" for term in terms)
        return formulas

    def to_formula(self, solution_type: str | None = None) -> str | None:
        """Return one minimized formula.

        When ``solution_type`` is omitted, the selected solution type is used.
        Available solution types are ``complex``, ``parsimonious``, and
        ``intermediate``.
        """
        resolved_type = (
            self.selected_solution_type if solution_type is None else solution_type
        )
        if resolved_type == "none":
            return None
        if resolved_type not in SOLUTION_TYPE_ORDER:
            available = ", ".join(SOLUTION_TYPE_ORDER)
            raise ValueError(
                f"Unknown solution_type {resolved_type!r}. "
                f"Available solution types: {available}."
            )
        return self.to_formulas().get(resolved_type)

    def export_formula(
        self,
        path: str | Path,
        solution_type: str | None = None,
    ) -> str | None:
        """Write one minimized formula to a text file and return it."""
        formula = self.to_formula(solution_type)
        Path(path).write_text("" if formula is None else formula, encoding="utf-8")
        return formula

    def _metric_value(self, solution_type: str, field_name: str) -> float | None:
        metric = self.solution_metrics.get(solution_type)
        if metric is None:
            return None
        return getattr(metric, field_name)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the solution table as a DataFrame."""
        return self.solutions

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render a compact Markdown report and optionally write it to disk."""
        lines = ["# QCA Fit Result", ""]
        lines.append("## Model")
        lines.append("")
        lines.append(f"- `qca_type`: {self.qca_type}")
        if self.outcome is not None:
            lines.append(f"- `outcome`: {self.outcome}")
        if self.case_id is not None:
            lines.append(f"- `case_id`: {self.case_id}")
        lines.append("")

        lines.append("## Settings")
        lines.append("")
        if self.settings:
            for key, value in self.settings.items():
                lines.append(f"- `{key}`: {value}")
        else:
            lines.append("- No settings recorded")
        lines.append("")

        lines.append("## Selected Solution")
        lines.append("")
        selected = self.selected_solution
        if selected is None:
            lines.append("No selected solution was found.")
        else:
            lines.append(f"- `solution_type`: {self.selected_solution_type}")
            lines.append(f"- `consistency`: {selected.consistency:.4f}")
            lines.append(f"- `coverage`: {selected.raw_coverage:.4f}")
        lines.append("")

        lines.append("## Formulas")
        lines.append("")
        formulas = self.to_formulas()
        if formulas:
            for solution_type, formula in formulas.items():
                lines.append(f"- `{solution_type}`: {formula}")
        else:
            lines.append("No formulas were found.")
        lines.append("")

        lines.append("## Solutions")
        lines.append("")
        solutions = self.solutions
        lines.append(
            self._dataframe_to_markdown(solutions)
            if not solutions.empty
            else "No solution terms were found."
        )
        lines.append("")

        lines.append("## Condition Schema")
        lines.append("")
        lines.append(
            self._dataframe_to_markdown(self.condition_schema)
            if not self.condition_schema.empty
            else "No condition schema was recorded."
        )
        lines.append("")

        lines.append("## Truth Table")
        lines.append("")
        truth_table = self.truth_table
        lines.append(
            self._dataframe_to_markdown(truth_table)
            if not truth_table.empty
            else "No truth-table rows were found."
        )
        lines.append("")

        markdown = "\n".join(lines)
        if path is not None:
            Path(path).write_text(markdown, encoding="utf-8")
        return markdown

    def to_markdown_report(self, path: str | Path | None = None) -> str:
        """Render the v0.6 automated Markdown report."""
        from qca.reporting import generate_markdown_report

        return generate_markdown_report(self, path=path)

    def to_latex(
        self,
        path: str | Path | None = None,
        *,
        table: str = "solutions",
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """Render one result table as LaTeX."""
        from qca.reporting import to_latex_table

        return to_latex_table(
            self,
            table=table,
            path=path,
            caption=caption,
            label=label,
        )

    def to_jupyter_summary(self):
        """Return a notebook-friendly summary object."""
        from qca.reporting import jupyter_summary

        return jupyter_summary(self)

    def _repr_html_(self) -> str:
        return self.to_jupyter_summary()._repr_html_()

    @staticmethod
    def _dataframe_to_markdown(df: pd.DataFrame) -> str:
        """Render a DataFrame as a simple GitHub-flavored Markdown table."""
        columns = [str(column) for column in df.columns]
        rows = [
            [QCAFitResult._format_markdown_cell(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    @staticmethod
    def _format_markdown_cell(value: Any) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, float):
            if pd.isna(value):
                return ""
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    def summary(self) -> str:
        """Return a short human-readable summary."""
        consistency = "N/A" if self.consistency is None else f"{self.consistency:.3f}"
        coverage = "N/A" if self.coverage is None else f"{self.coverage:.3f}"
        return (
            "QCAFitResult("
            f"selected_solution_type={self.selected_solution_type!r}, "
            f"consistency={consistency}, "
            f"coverage={coverage}, "
            f"n_solution_terms={len(self.solutions)}, "
            f"n_truth_table_rows={len(self.truth_table_rows)}"
            ")"
        )
