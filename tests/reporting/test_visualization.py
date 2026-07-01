"""Reporting and optional visualization tests."""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import GSQCA, generate_report
from qca.core.literals import SetLiteral
from qca.reporting import (
    consistency_label,
    display_width,
    format_key_value,
    generate_report as reporting_generate_report,
    strength_bar,
    truncate_display,
)


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "skill": [0.9, 0.8, 0.3, 0.2, 0.9, 0.8, 0.1, 0.2],
            "resources": [0.8, 0.3, 0.8, 0.2, 0.2, 0.9, 0.7, 0.1],
            "region": [
                "north",
                "north",
                "north",
                "south",
                "south",
                "east",
                "east",
                "south",
            ],
            "outcome": [0.9, 0.7, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1],
        }
    )


@pytest.fixture
def model(base_df) -> GSQCA:
    return GSQCA(
        data=base_df,
        case_id="case",
        conditions=["skill", "resources", "region"],
        condition_types={"skill": "fuzzy", "resources": "fuzzy", "region": "multi"},
        outcome="outcome",
    )


@pytest.fixture
def sufficiency_results(model) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.search_sufficient_configurations(
            max_depth=2,
            min_consistency=0.5,
            min_coverage=0.0,
            min_cases=1,
        )


@pytest.fixture
def necessity_results(model) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.search_necessary_conditions(min_consistency=0.5, min_coverage=0.0)


class TestReportFormatters:
    def test_display_width_counts_wide_chars(self):
        assert display_width("QCA") == 3
        assert display_width("ＡＢ") == 4

    def test_truncate_display(self):
        assert truncate_display("abcdef", 4) == "a..."

    def test_strength_bar(self):
        assert strength_bar(0.5, width=10) == "[#####-----] 0.500"

    def test_consistency_label(self):
        assert consistency_label(0.9) == "strong"
        assert consistency_label(0.8) == "moderate"

    def test_format_key_value(self):
        line = format_key_value("outcome", "success")
        assert "outcome" in line
        assert "success" in line


class TestGenerateReport:
    def test_top_level_and_reporting_export_match(self):
        assert generate_report is reporting_generate_report

    def test_generate_report_from_model(
        self, model, sufficiency_results, necessity_results
    ):
        report = generate_report(
            model, sufficiency_results, necessity_results, max_rows=3
        )
        assert "PyQCA Analysis Report" in report
        assert "Sufficiency Analysis" in report
        assert "Necessity Analysis" in report
        assert "Truth Table" in report
        assert "outcome" in report

    def test_generate_report_from_fit_result(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(minimizer="qmc", outcome_threshold=0.75)
        report = generate_report(result, max_rows=3)
        assert "Fit Result" in report
        assert "Solution Terms" in report
        assert "Logical Minimization" in report

    def test_generate_report_writes_file(self, model):
        path = Path.cwd() / ".pyqca_report_test.txt"
        try:
            report = generate_report(model, path=path, max_rows=2)
            assert path.read_text(encoding="utf-8") == report
        finally:
            if path.exists():
                path.unlink()


class TestVisualizationOptional:
    def test_import_qca_viz_without_matplotlib_dependency(self):
        viz = importlib.import_module("qca.viz")
        assert hasattr(viz, "plot_xy")
        assert hasattr(viz, "plot_sufficiency_bar")

    def test_color_helpers_do_not_need_matplotlib(self):
        from qca.viz import consistency_color, scatter_point_color

        assert consistency_color(0.9).startswith("#")
        assert scatter_point_color(0.8, 0.8).startswith("#")

    def test_plot_functions_raise_or_run_based_on_matplotlib(self, model):
        from qca.viz import plot_xy

        has_matplotlib = importlib.util.find_spec("matplotlib") is not None
        if not has_matplotlib:
            with pytest.raises(ImportError, match=r"pyqca\[viz\]"):
                plot_xy(model, [SetLiteral("skill")])
        else:
            import matplotlib

            matplotlib.use("Agg")
            ax = plot_xy(model, [SetLiteral("skill")], annotate=False)
            assert ax is not None

    def test_truth_table_matrix_without_matplotlib(self, model):
        from qca.viz.truth_table import _build_condition_matrix

        rows = model.build_truth_table()
        matrix, maps = _build_condition_matrix(
            rows,
            model.all_condition_names,
            model.set_conditions,
            model.multivalue_conditions,
        )
        assert matrix.shape == (len(rows), len(model.all_condition_names))
        assert "region" in maps

    def test_parse_best_literals_without_matplotlib(self, model, sufficiency_results):
        from qca.viz.dashboard import _parse_best_literals

        literals = _parse_best_literals(sufficiency_results, model)
        assert literals is None or all(hasattr(literal, "name") for literal in literals)
