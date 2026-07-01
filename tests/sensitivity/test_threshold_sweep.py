"""ThresholdSweep acceptance tests."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pandas as pd
import pytest

from qca import ThresholdSweep
from qca.sweep import (
    ThresholdSweepResult,
    prepare_threshold_data,
    qca_bin,
    validate_pre_calibrated,
)


@pytest.fixture
def raw_threshold_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": [f"c{i}" for i in range(1, 9)],
            "X1": [1, 2, 3, 4, 5, 6, 7, 8],
            "X2": [8, 7, 6, 5, 4, 3, 2, 1],
            "X3_cal": [0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1],
            "Y": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def test_v04_qca_bin_uses_greater_equal_rule() -> None:
    assert qca_bin([0, 1, 2], 1).tolist() == [0, 1, 1]


def test_v04_prepare_threshold_data_matches_thsqca_rules(raw_threshold_df) -> None:
    dat_bin, specs = prepare_threshold_data(
        raw_threshold_df,
        case_id="case",
        outcome="Y",
        conditions=["X1", "X2", "X3_cal"],
        outcome_threshold=1,
        condition_thresholds={"X1": 5, "X2": 5},
        pre_calibrated=["X3_cal"],
    )

    assert dat_bin["__tsqca_Y__"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert dat_bin["X1"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert dat_bin["X2"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert dat_bin["X3_cal"].tolist() == raw_threshold_df["X3_cal"].tolist()
    assert {spec.name: spec.kind for spec in specs} == {
        "X1": "crisp",
        "X2": "crisp",
        "X3_cal": "fuzzy",
    }


def test_v04_prepare_threshold_data_supports_negated_outcome(raw_threshold_df) -> None:
    dat_bin, _ = prepare_threshold_data(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1"],
        outcome_threshold=1,
        condition_thresholds={"X1": 5},
        negate_outcome=True,
    )

    assert dat_bin["__tsqca_Y__"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]


def test_v04_pre_calibrated_validation_rejects_out_of_range(
    raw_threshold_df,
) -> None:
    bad = raw_threshold_df.copy()
    bad["X3_cal"] = bad["X3_cal"] + 1

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_pre_calibrated(["X3_cal"], ["X3_cal"], bad)


def test_v04_outcome_threshold_sweep(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.outcome(
            thresholds=[0, 1],
            condition_thresholds={"X1": 5, "X2": 5},
            incl_cut=0.5,
            n_cut=1,
        )

    assert isinstance(result, ThresholdSweepResult)
    assert result.settings["mode"] == "otSweep"
    assert result.summary_df["thrY"].tolist() == [0.0, 1.0]
    assert {"expression", "inclS", "covS", "n_solutions"}.issubset(
        result.summary_df.columns
    )
    assert len(result.details) == 2
    assert result.stability["n_runs"] == 2
    assert result.truth_tables


def test_threshold_sweep_passes_pri_cut_to_fit(raw_threshold_df, monkeypatch) -> None:
    from qca.sweep import threshold as threshold_module

    seen: list[float | None] = []
    original_fit = threshold_module.GSQCA.fit

    def spy_fit(self, *args, **kwargs):
        seen.append(kwargs.get("pri_cut"))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(threshold_module.GSQCA, "fit", spy_fit)
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep.outcome(
            thresholds=[1],
            condition_thresholds={"X1": 5, "X2": 5},
            incl_cut=0.5,
            pri_cut=0.4,
        )

    assert seen == [0.4]


def test_v04_single_condition_threshold_sweep(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.condition(
            "X1",
            thresholds=[4, 5],
            outcome_threshold=1,
            condition_thresholds={"X2": 5},
            incl_cut=0.5,
        )

    assert result.settings["mode"] == "ctSweepS"
    assert result.summary_df["threshold"].tolist() == [4.0, 5.0]
    assert "thr_X1" in result.summary_df.columns


def test_v04_multi_condition_threshold_grid_search(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.multi_condition(
            {"X1": [4, 5], "X2": [4, 5]},
            outcome_threshold=1,
            incl_cut=0.5,
        )

    assert result.settings["mode"] == "ctSweepM"
    assert len(result.summary_df) == 4
    assert result.summary_df["combo_id"].tolist() == [1, 2, 3, 4]
    assert {"thr_X1", "thr_X2"}.issubset(result.summary_df.columns)


def test_v04_dual_outcome_condition_threshold_sweep(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.dual(
            {"X1": [4, 5], "X2": [5]},
            outcome_thresholds=[0, 1],
            incl_cut=0.5,
        )

    assert result.settings["mode"] == "dtSweep"
    assert len(result.summary_df) == 4
    assert {"thrY", "thrX", "combo_id"}.issubset(result.summary_df.columns)


def test_v04_sweep_result_markdown_export(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.outcome(
            thresholds=[1],
            condition_thresholds={"X1": 5, "X2": 5},
            incl_cut=0.5,
        )

    path = Path.cwd() / ".pyqca_v04_sweep_test.md"
    try:
        markdown = result.to_markdown(path)

        assert path.read_text(encoding="utf-8") == markdown
        assert "# Threshold Sweep Result" in markdown
        assert "## Stability" in markdown
        assert "## Summary" in markdown
    finally:
        if path.exists():
            path.unlink()


def test_v04_sweep_plots_are_optional(raw_threshold_df) -> None:
    sweep = ThresholdSweep(
        raw_threshold_df,
        outcome="Y",
        conditions=["X1", "X2"],
        case_id="case",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sweep.dual(
            {"X1": [4, 5], "X2": [5]},
            outcome_thresholds=[0, 1],
            incl_cut=0.5,
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
