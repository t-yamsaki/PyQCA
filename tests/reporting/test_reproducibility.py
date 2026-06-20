"""Phase 15: README Roadmap v0.6 reporting and reproducibility tests."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import (
    FSQCA,
    ExperimentLogger,
    collect_reproducibility_metadata,
    generate_markdown_report,
    jupyter_summary,
    log_experiment,
    to_latex_table,
)
from qca.reporting import JupyterSummary
from qca.viz import configuration_matrix, plot_configuration_chart


@pytest.fixture
def report_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 7)],
            "A": [1, 1, 1, 0, 0, 0],
            "B": [1, 1, 0, 1, 0, 0],
            "Y": [1, 1, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def fit_result(report_df):
    model = FSQCA(
        report_df,
        outcome="Y",
        conditions=["A", "B"],
        condition_types={"A": "crisp", "B": "crisp"},
        case_id="case",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(consistency_cutoff=0.5, outcome_threshold=0.5)


def test_v06_automated_markdown_report(fit_result) -> None:
    path = Path.cwd() / ".pyqca_v06_report.md"
    try:
        markdown = generate_markdown_report(fit_result, path=path)

        assert path.read_text(encoding="utf-8") == markdown
        assert "# fsQCA Report" in markdown
        assert "## Reproducibility Metadata" in markdown
        assert "## Solutions" in markdown
        assert "`qca_version`" in markdown
    finally:
        if path.exists():
            path.unlink()


def test_v06_latex_table_export(fit_result) -> None:
    latex = to_latex_table(
        fit_result,
        table="solutions",
        caption="QCA solutions",
        label="tab:qca-solutions",
    )

    assert "\\begin{table}" in latex
    assert "\\begin{tabular}" in latex
    assert "\\caption{QCA solutions}" in latex
    assert "solution\\_type" in latex


def test_v06_reproducibility_metadata(fit_result) -> None:
    metadata = collect_reproducibility_metadata(
        fit_result,
        name="smoke",
        extra={"seed": 123},
    )

    assert metadata["name"] == "smoke"
    assert metadata["object_type"] == "QCAFitResult"
    assert metadata["qca_version"]
    assert metadata["python_version"]
    assert metadata["settings"]["minimizer"] == "standard"
    assert metadata["extra"]["seed"] == 123


def test_v06_experiment_logging(fit_result) -> None:
    path = Path.cwd() / ".pyqca_v06_experiments.jsonl"
    try:
        logger = ExperimentLogger(path)
        record = logger.log(fit_result, name="run-1", notes="acceptance")
        record_2 = log_experiment(fit_result, path, name="run-2")

        records = logger.read()
        assert record["name"] == "run-1"
        assert record_2["name"] == "run-2"
        assert len(records) == 2
        assert records[0]["notes"] == "acceptance"
    finally:
        if path.exists():
            path.unlink()


def test_v06_jupyter_friendly_summary(fit_result) -> None:
    summary = jupyter_summary(fit_result)

    assert isinstance(summary, JupyterSummary)
    assert "QCAFitResult" in str(summary)
    assert "<table" in summary._repr_html_()
    assert "fsQCA" in summary._repr_html_()
    assert fit_result._repr_html_() == fit_result.to_jupyter_summary()._repr_html_()


def test_v06_configuration_matrix() -> None:
    df = pd.DataFrame(
        {
            "term": ["A * B", "A * ~B", "~A * B"],
            "consistency": [0.95, 0.80, 0.70],
            "coverage": [0.50, 0.30, 0.20],
        }
    )

    matrix, rows = configuration_matrix(df)

    assert matrix.loc[0, "A"] == 1
    assert matrix.loc[1, "B"] == -1
    assert matrix.loc[2, "A"] == -1
    assert rows["configuration_label"].tolist() == ["A * B", "A * ~B", "~A * B"]


def test_v06_configuration_chart_optional() -> None:
    df = pd.DataFrame(
        {
            "term": ["A * B", "A * ~B"],
            "consistency": [0.95, 0.80],
        }
    )
    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    if not has_matplotlib:
        with pytest.raises(ImportError, match=r"pyqca\[viz\]"):
            plot_configuration_chart(df)
    else:
        import matplotlib

        matplotlib.use("Agg")
        fig = plot_configuration_chart(df)
        assert fig.axes
