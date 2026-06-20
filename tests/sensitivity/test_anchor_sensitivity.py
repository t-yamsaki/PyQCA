"""Phase 14: README Roadmap v0.5 fsQCA anchor sensitivity tests."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import AnchorSensitivity, ThresholdSweep
from qca.sweep import AnchorSensitivityResult


@pytest.fixture
def raw_anchor_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 9)],
            "Budget": [1, 2, 3, 4, 5, 6, 7, 8],
            "Capacity": [0.05, 0.10, 0.30, 0.45, 0.70, 0.80, 0.90, 0.95],
            "Y": [0.05, 0.10, 0.30, 0.45, 0.75, 0.85, 0.95, 1.00],
        }
    )


def test_v05_full_membership_anchor_sweep(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.full_membership(
            "Budget",
            values=[7, 8],
            full_out=1,
            crossover=4,
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(result, AnchorSensitivityResult)
    assert result.sweep_type == "full_membership"
    assert result.summary_df["full_in"].tolist() == [7.0, 8.0]
    assert {"expression", "inclS", "covS", "valid"}.issubset(result.summary_df)
    assert result.stability["n_runs"] == 2
    assert result.calibrated_frames


def test_v05_crossover_anchor_sweep(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.crossover(
            "Budget",
            values=[4, 5],
            full_out=1,
            full_in=8,
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert result.sweep_type == "crossover"
    assert result.summary_df["crossover"].tolist() == [4.0, 5.0]
    assert result.settings["condition"] == "Budget"


def test_v05_full_nonmembership_anchor_sweep(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.full_nonmembership(
            "Budget",
            values=[1, 2],
            crossover=4,
            full_in=8,
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert result.sweep_type == "full_nonmembership"
    assert result.summary_df["full_out"].tolist() == [1.0, 2.0]
    assert result.stability["full_out_range"] == (1.0, 2.0)


def test_v05_multi_anchor_grid_analysis(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.grid(
            "Budget",
            {
                "full_out": [1, 2],
                "crossover": [4, 5],
                "full_in": [8],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert result.sweep_type == "multi_anchor"
    assert len(result.summary_df) == 4
    assert result.summary_df["combo_id"].tolist() == [1, 2, 3, 4]
    assert {"full_out", "crossover", "full_in"}.issubset(result.summary_df)


def test_v05_threshold_sweep_fuzzy_anchor_adapter(raw_anchor_df) -> None:
    sweep = ThresholdSweep(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.fuzzy_anchors(
            "Budget",
            {
                "full_out": [1],
                "crossover": [4, 5],
                "full_in": [8],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    assert isinstance(result, AnchorSensitivityResult)
    assert result.settings["mode"] == "anchorSensitivity"
    assert len(result.summary_df) == 2


def test_v05_calibration_stability_markdown_export(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.grid(
            "Budget",
            {
                "full_out": [1],
                "crossover": [4],
                "full_in": [7, 8],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    path = Path.cwd() / ".pyqca_v05_anchor_test.md"
    try:
        markdown = result.to_markdown(path)

        assert path.read_text(encoding="utf-8") == markdown
        assert "# fsQCA Anchor Sensitivity Result" in markdown
        assert "## Calibration Stability" in markdown
        assert "## Summary" in markdown
    finally:
        if path.exists():
            path.unlink()


def test_v05_anchor_sensitivity_plots_are_optional(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.grid(
            "Budget",
            {
                "full_out": [1],
                "crossover": [4, 5],
                "full_in": [7, 8],
            },
            incl_cut=0.5,
            outcome_threshold=0.5,
        )

    has_matplotlib = importlib.util.find_spec("matplotlib") is not None
    if not has_matplotlib:
        with pytest.raises(ImportError):
            result.plot_heatmap()
        with pytest.raises(ImportError):
            result.plot_trajectory()
    else:
        import matplotlib

        matplotlib.use("Agg")
        heatmap = result.plot_heatmap()
        trajectory = result.plot_trajectory()
        assert heatmap.axes
        assert trajectory.axes


def test_v05_anchor_grid_rejects_unordered_anchors(raw_anchor_df) -> None:
    sweep = AnchorSensitivity(
        raw_anchor_df,
        outcome="Y",
        conditions=["Budget", "Capacity"],
        case_id="case",
    )

    with pytest.raises(ValueError, match="full_out < crossover < full_in"):
        sweep.grid(
            "Budget",
            {
                "full_out": [5],
                "crossover": [4],
                "full_in": [8],
            },
        )
