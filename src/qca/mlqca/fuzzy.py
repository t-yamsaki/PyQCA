"""fsQCA anchor proposals, calibration, and sensitivity adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca.calibration._validators import validate_anchor_ordering
from qca.calibration.piecewise import piecewise_fuzzy_series
from qca.engines import FSQCA
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.calibration import (
    _normalize_direction,
    _resolve_conditions,
    _resolve_direction,
)
from qca.mlqca.validation import ValidatedMLQCAInput
from qca.results import QCAFitResult
from qca.sweep import AnchorSensitivity, AnchorSensitivityResult

FUZZY_PROPOSAL_COLUMNS = (
    "feature",
    "output",
    "full_out",
    "crossover",
    "full_in",
    "direction",
    "anchor_source",
    "direction_source",
    "inferred_direction",
    "n_xgboost_cutoffs",
)


@dataclass(frozen=True)
class FuzzyCalibrationSet:
    """Fuzzy-set data generated from mlQCA anchor proposals."""

    data: pd.DataFrame
    outcome: str
    conditions: tuple[str, ...]
    source_conditions: tuple[str, ...]
    proposals: pd.DataFrame
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(self.proposals, pd.DataFrame):
            raise TypeError("proposals must be a pandas DataFrame.")
        if len(self.conditions) != len(self.source_conditions):
            raise ValueError(
                "conditions and source_conditions must have the same length."
            )
        object.__setattr__(self, "data", self.data.copy())
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(
            self,
            "source_conditions",
            tuple(self.source_conditions),
        )
        object.__setattr__(self, "proposals", self.proposals.copy())

    @property
    def condition_map(self) -> dict[str, str]:
        """Map raw source conditions to calibrated fuzzy columns."""
        return dict(zip(self.source_conditions, self.conditions, strict=True))


@dataclass(frozen=True)
class MLQCAFSQCAResult:
    """Result of fitting fsQCA to mlQCA-calibrated fuzzy conditions."""

    calibration: FuzzyCalibrationSet
    engine: FSQCA
    fit_result: QCAFitResult

    @property
    def consistency(self) -> float | None:
        return self.fit_result.consistency

    @property
    def coverage(self) -> float | None:
        return self.fit_result.coverage


@dataclass(frozen=True)
class MLQCAAnchorSensitivityResult:
    """Anchor-sensitivity output with raw and transformed anchor metadata."""

    feature: str
    proposal: dict[str, Any]
    raw_anchor_grid: dict[str, list[float]]
    analysis_anchor_grid: dict[str, list[float]]
    transformed_for_low_direction: bool
    result: AnchorSensitivityResult

    @property
    def summary_df(self) -> pd.DataFrame:
        return self.result.summary_df

    @property
    def stability(self) -> dict[str, Any]:
        return self.result.stability


def propose_fuzzy_anchors(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    conditions: Sequence[str] | None = None,
    *,
    anchor_overrides: Mapping[str, Sequence[float]] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
    output_suffix: str = "__fuzzy",
) -> pd.DataFrame:
    """Propose three fsQCA anchors for each selected condition."""
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    if not isinstance(predictor, PredictorFitResult):
        raise TypeError("predictor must be a PredictorFitResult object.")
    if not output_suffix:
        raise ValueError("output_suffix must be a non-empty string.")

    selected = _resolve_conditions(validated, conditions)
    overrides = dict(anchor_overrides or {})
    direction_values = dict(direction_overrides or {})
    _check_override_names(selected, overrides, "anchor_overrides")
    _check_override_names(selected, direction_values, "direction_overrides")

    specs = {spec.name: spec for spec in validated.condition_specs}
    importance = predictor.feature_importance.set_index("feature", drop=False)
    records: list[dict[str, Any]] = []
    for feature in selected:
        spec = specs[feature]
        anchors, source, n_cutoffs = _resolve_fuzzy_anchors(
            feature,
            theoretical_cutoffs=spec.theoretical_cutoffs,
            overrides=overrides,
            cutoffs=predictor.cutoff_candidates,
        )
        direction, direction_source = _resolve_direction(
            feature,
            spec.direction,
            direction_values,
            importance,
        )
        records.append(
            {
                "feature": feature,
                "output": f"{feature}{output_suffix}",
                "full_out": anchors[0],
                "crossover": anchors[1],
                "full_in": anchors[2],
                "direction": direction,
                "anchor_source": source,
                "direction_source": direction_source,
                "inferred_direction": direction_source == "shap",
                "n_xgboost_cutoffs": n_cutoffs,
            }
        )
    return pd.DataFrame(records, columns=list(FUZZY_PROPOSAL_COLUMNS))


def calibrate_fuzzy_conditions(
    validated: ValidatedMLQCAInput,
    proposals: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> FuzzyCalibrationSet:
    """Apply piecewise fuzzy calibration using mlQCA anchor proposals."""
    _validate_fuzzy_proposals(validated, proposals)
    work = validated.data.copy()
    outputs: list[str] = []
    sources: list[str] = []
    for row in proposals.itertuples(index=False):
        membership = piecewise_fuzzy_series(
            work[row.feature],
            row.full_out,
            row.crossover,
            row.full_in,
        )
        if row.direction == "low":
            membership = 1.0 - membership
        if not allow_missing and membership.isna().any():
            raise ValueError(
                f"Calibrated condition {row.output!r} contains missing values."
            )
        work[row.output] = membership
        outputs.append(str(row.output))
        sources.append(str(row.feature))

    return FuzzyCalibrationSet(
        data=work,
        outcome=validated.outcome,
        conditions=tuple(outputs),
        source_conditions=tuple(sources),
        proposals=proposals,
        case_id=validated.case_id,
    )


def fit_fsqca_from_calibration(
    calibration: FuzzyCalibrationSet,
    **fit_kwargs: Any,
) -> MLQCAFSQCAResult:
    """Build and fit FSQCA from mlQCA fuzzy calibration output."""
    if not isinstance(calibration, FuzzyCalibrationSet):
        raise TypeError("calibration must be a FuzzyCalibrationSet.")
    required = [*calibration.conditions, calibration.outcome]
    missing = [
        column
        for column in required
        if calibration.data[column].isna().any()
    ]
    if missing:
        raise ValueError(
            "FSQCA cannot fit calibrated data containing missing values: "
            f"{missing}"
        )
    engine = FSQCA(
        calibration.data,
        outcome=calibration.outcome,
        conditions=calibration.conditions,
        condition_types={
            condition: "fuzzy" for condition in calibration.conditions
        },
        case_id=calibration.case_id,
    )
    return MLQCAFSQCAResult(
        calibration=calibration,
        engine=engine,
        fit_result=engine.fit(**fit_kwargs),
    )


def fit_fsqca_from_predictor(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    conditions: Sequence[str] | None = None,
    *,
    anchor_overrides: Mapping[str, Sequence[float]] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
    output_suffix: str = "__fuzzy",
    **fit_kwargs: Any,
) -> MLQCAFSQCAResult:
    """Propose anchors, calibrate fuzzy sets, and fit FSQCA."""
    proposals = propose_fuzzy_anchors(
        validated,
        predictor,
        conditions,
        anchor_overrides=anchor_overrides,
        direction_overrides=direction_overrides,
        output_suffix=output_suffix,
    )
    calibration = calibrate_fuzzy_conditions(validated, proposals)
    return fit_fsqca_from_calibration(calibration, **fit_kwargs)


def proposal_anchor_grid(
    proposals: pd.DataFrame,
    feature: str,
    *,
    span: float = 0.2,
) -> dict[str, list[float]]:
    """Generate an ordered three-by-three-by-three grid around one proposal."""
    if not 0.0 < float(span) < 0.5:
        raise ValueError("span must be greater than 0 and less than 0.5.")
    row = _proposal_row(proposals, feature)
    full_out = float(row["full_out"])
    crossover = float(row["crossover"])
    full_in = float(row["full_in"])
    validate_anchor_ordering(full_out, crossover, full_in)
    low_gap = crossover - full_out
    high_gap = full_in - crossover
    middle_delta = min(low_gap, high_gap) * span
    return {
        "full_out": [
            full_out - low_gap * span,
            full_out,
            full_out + low_gap * span,
        ],
        "crossover": [
            crossover - middle_delta,
            crossover,
            crossover + middle_delta,
        ],
        "full_in": [
            full_in - high_gap * span,
            full_in,
            full_in + high_gap * span,
        ],
    }


def run_anchor_sensitivity(
    validated: ValidatedMLQCAInput,
    calibration: FuzzyCalibrationSet,
    feature: str,
    *,
    anchor_grid: Mapping[str, Sequence[float]] | None = None,
    span: float = 0.2,
    **sweep_kwargs: Any,
) -> MLQCAAnchorSensitivityResult:
    """Connect an mlQCA fuzzy proposal to ``AnchorSensitivity``."""
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    if not isinstance(calibration, FuzzyCalibrationSet):
        raise TypeError("calibration must be a FuzzyCalibrationSet.")
    if feature not in calibration.source_conditions:
        raise ValueError(f"Unknown calibrated feature {feature!r}.")

    proposal = _proposal_row(calibration.proposals, feature)
    raw_grid = (
        proposal_anchor_grid(calibration.proposals, feature, span=span)
        if anchor_grid is None
        else _normalize_anchor_grid(anchor_grid)
    )
    direction = _normalize_direction(str(proposal["direction"]))
    output = calibration.condition_map[feature]
    other_outputs = [
        condition
        for source, condition in calibration.condition_map.items()
        if source != feature
    ]
    work = calibration.data.copy()
    analysis_feature = feature
    analysis_grid = raw_grid
    transformed = direction == "low"
    if transformed:
        analysis_feature = f"{feature}__mlqca_inverted"
        work[analysis_feature] = -pd.to_numeric(
            work[feature],
            errors="raise",
        )
        analysis_grid = _invert_anchor_grid(raw_grid)

    sweep = AnchorSensitivity(
        work,
        outcome=calibration.outcome,
        conditions=[analysis_feature, *other_outputs],
        case_id=calibration.case_id,
    )
    result = sweep.grid(
        analysis_feature,
        analysis_grid,
        out_col=output,
        **sweep_kwargs,
    )
    result.settings.update(
        {
            "mlqca_feature": feature,
            "mlqca_direction": direction,
            "mlqca_raw_anchor_grid": raw_grid,
            "mlqca_transformed_for_low_direction": transformed,
        }
    )
    return MLQCAAnchorSensitivityResult(
        feature=feature,
        proposal=proposal,
        raw_anchor_grid=raw_grid,
        analysis_anchor_grid=analysis_grid,
        transformed_for_low_direction=transformed,
        result=result,
    )


def _resolve_fuzzy_anchors(
    feature: str,
    *,
    theoretical_cutoffs: tuple[float, ...],
    overrides: Mapping[str, Sequence[float]],
    cutoffs: pd.DataFrame,
) -> tuple[tuple[float, float, float], str, int]:
    if feature in overrides:
        return _anchor_triplet(overrides[feature], feature), "override", 0
    if len(theoretical_cutoffs) >= 3:
        return (
            _anchor_triplet(theoretical_cutoffs, feature),
            "theoretical",
            0,
        )
    if not cutoffs.empty:
        required = {"feature", "cutoff"}
        missing = sorted(required - set(cutoffs.columns))
        if missing:
            raise ValueError(
                f"predictor cutoff candidates are missing columns: {missing}"
            )
        rows = cutoffs.loc[cutoffs["feature"] == feature].copy()
        if "cutoff_rank" in rows:
            rows = rows.sort_values("cutoff_rank", kind="stable")
        values = list(dict.fromkeys(pd.to_numeric(rows["cutoff"]).tolist()))
        if len(values) >= 3:
            return _anchor_triplet(values[:3], feature), "xgboost", len(values)
    raise ValueError(
        f"Three fuzzy anchors are not available for {feature!r}. "
        "Provide anchor_overrides or three theoretical_cutoffs."
    )


def _anchor_triplet(
    values: Sequence[float],
    feature: str,
) -> tuple[float, float, float]:
    numeric = sorted({float(value) for value in values})
    if len(numeric) < 3:
        raise ValueError(
            f"Fuzzy anchor proposal for {feature!r} requires three "
            "distinct values."
        )
    anchors = (numeric[0], numeric[len(numeric) // 2], numeric[-1])
    validate_anchor_ordering(*anchors)
    return anchors


def _check_override_names(
    conditions: tuple[str, ...],
    overrides: Mapping[str, object],
    name: str,
) -> None:
    unknown = sorted(set(overrides) - set(conditions))
    if unknown:
        raise ValueError(f"{name} contains unknown conditions: {unknown}")


def _validate_fuzzy_proposals(
    validated: ValidatedMLQCAInput,
    proposals: pd.DataFrame,
) -> None:
    if not isinstance(proposals, pd.DataFrame):
        raise TypeError("proposals must be a pandas DataFrame.")
    required = {
        "feature",
        "output",
        "full_out",
        "crossover",
        "full_in",
        "direction",
    }
    missing = sorted(required - set(proposals.columns))
    if missing:
        raise ValueError(f"proposals is missing required columns: {missing}")
    if proposals.empty:
        raise ValueError("proposals must contain at least one condition.")
    if proposals["feature"].duplicated().any():
        raise ValueError("proposals contains duplicate features.")
    unknown = sorted(set(proposals["feature"]) - set(validated.candidates))
    if unknown:
        raise ValueError(f"proposals contains unknown conditions: {unknown}")
    collisions = sorted(set(proposals["output"]) & set(validated.data.columns))
    if collisions:
        raise ValueError(
            f"Calibrated output columns already exist in data: {collisions}"
        )
    for row in proposals.itertuples(index=False):
        validate_anchor_ordering(row.full_out, row.crossover, row.full_in)
        _normalize_direction(str(row.direction))


def _proposal_row(
    proposals: pd.DataFrame,
    feature: str,
) -> dict[str, Any]:
    if "feature" not in proposals:
        raise ValueError("proposals must contain a 'feature' column.")
    rows = proposals.loc[proposals["feature"] == feature]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one fuzzy anchor proposal for {feature!r}, got {len(rows)}."
        )
    return rows.iloc[0].to_dict()


def _normalize_anchor_grid(
    anchor_grid: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    required = ("full_out", "crossover", "full_in")
    missing = [key for key in required if key not in anchor_grid]
    if missing:
        raise ValueError(f"anchor_grid is missing required keys: {missing}")
    grid = {
        key: [float(value) for value in anchor_grid[key]]
        for key in required
    }
    if any(not values for values in grid.values()):
        raise ValueError("anchor_grid values cannot be empty.")
    return grid


def _invert_anchor_grid(
    grid: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    return {
        "full_out": sorted(-float(value) for value in grid["full_in"]),
        "crossover": sorted(-float(value) for value in grid["crossover"]),
        "full_in": sorted(-float(value) for value in grid["full_out"]),
    }


__all__ = [
    "FUZZY_PROPOSAL_COLUMNS",
    "FuzzyCalibrationSet",
    "MLQCAAnchorSensitivityResult",
    "MLQCAFSQCAResult",
    "calibrate_fuzzy_conditions",
    "fit_fsqca_from_calibration",
    "fit_fsqca_from_predictor",
    "proposal_anchor_grid",
    "propose_fuzzy_anchors",
    "run_anchor_sensitivity",
]
