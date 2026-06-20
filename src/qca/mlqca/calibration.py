"""Crisp calibration planning and execution for mlQCA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from qca.calibration import crisp_calibrate_series
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.validation import ValidatedMLQCAInput

CalibrationDirection = Literal["high", "low"]

PROPOSAL_COLUMNS = (
    "feature",
    "output",
    "threshold",
    "direction",
    "operator",
    "threshold_source",
    "direction_source",
    "inferred_direction",
    "cutoff_rank",
    "split_count",
    "split_frequency",
    "shap_mean",
)


@dataclass(frozen=True)
class CrispCalibrationSet:
    """Calibrated crisp-set data and the proposal table that produced it."""

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
        """Map raw source conditions to calibrated output columns."""
        return dict(zip(self.source_conditions, self.conditions, strict=True))


def propose_crisp_calibrations(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    conditions: Sequence[str] | None = None,
    *,
    direction_overrides: Mapping[str, str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    output_suffix: str = "__crisp",
) -> pd.DataFrame:
    """Create one crisp calibration proposal per selected condition."""
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    if not isinstance(predictor, PredictorFitResult):
        raise TypeError("predictor must be a PredictorFitResult object.")
    if not output_suffix:
        raise ValueError("output_suffix must be a non-empty string.")

    selected = _resolve_conditions(validated, conditions)
    directions = dict(direction_overrides or {})
    thresholds = dict(threshold_overrides or {})
    _reject_unknown_overrides(selected, directions, "direction_overrides")
    _reject_unknown_overrides(selected, thresholds, "threshold_overrides")

    specs = {spec.name: spec for spec in validated.condition_specs}
    importance = _indexed_table(predictor.feature_importance, "feature")
    cutoffs = predictor.cutoff_candidates.copy()
    records: list[dict[str, object]] = []

    for feature in selected:
        spec = specs[feature]
        threshold, threshold_source, cutoff_row = _resolve_threshold(
            feature,
            spec.theoretical_cutoffs,
            thresholds,
            cutoffs,
            data_type=spec.data_type,
        )
        direction, direction_source = _resolve_direction(
            feature,
            spec.direction,
            directions,
            importance,
        )
        importance_row = importance.loc[feature]
        records.append(
            {
                "feature": feature,
                "output": f"{feature}{output_suffix}",
                "threshold": threshold,
                "direction": direction,
                "operator": ">=" if direction == "high" else "<=",
                "threshold_source": threshold_source,
                "direction_source": direction_source,
                "inferred_direction": direction_source == "shap",
                "cutoff_rank": cutoff_row.get("cutoff_rank"),
                "split_count": cutoff_row.get("count", 0),
                "split_frequency": cutoff_row.get("frequency", 0.0),
                "shap_mean": importance_row.get("shap_mean"),
            }
        )

    return pd.DataFrame(records, columns=list(PROPOSAL_COLUMNS))


def calibrate_crisp_conditions(
    validated: ValidatedMLQCAInput,
    proposals: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> CrispCalibrationSet:
    """Apply crisp calibration proposals to validated mlQCA input."""
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    _validate_proposals(proposals, validated)

    work = validated.data.copy()
    outputs: list[str] = []
    sources: list[str] = []
    for row in proposals.itertuples(index=False):
        direction = "ge" if row.direction == "high" else "le"
        membership = crisp_calibrate_series(
            work[row.feature],
            threshold=float(row.threshold),
            direction=direction,
        )
        if not allow_missing and membership.isna().any():
            raise ValueError(
                f"Calibrated condition {row.output!r} contains missing values. "
                "Use complete data before connecting to CSQCA."
            )
        work[row.output] = membership
        outputs.append(str(row.output))
        sources.append(str(row.feature))

    return CrispCalibrationSet(
        data=work,
        outcome=validated.outcome,
        conditions=tuple(outputs),
        source_conditions=tuple(sources),
        proposals=proposals,
        case_id=validated.case_id,
    )


def _resolve_conditions(
    validated: ValidatedMLQCAInput,
    conditions: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = (
        validated.candidates
        if conditions is None
        else tuple(str(value).strip() for value in conditions)
    )
    if not selected or any(not value for value in selected):
        raise ValueError("conditions must contain at least one non-empty name.")
    if len(set(selected)) != len(selected):
        raise ValueError("conditions cannot contain duplicates.")
    unknown = sorted(set(selected) - set(validated.candidates))
    if unknown:
        raise ValueError(f"Unknown mlQCA conditions: {unknown}")
    return selected


def _reject_unknown_overrides(
    conditions: tuple[str, ...],
    overrides: Mapping[str, object],
    name: str,
) -> None:
    unknown = sorted(set(overrides) - set(conditions))
    if unknown:
        raise ValueError(f"{name} contains unknown conditions: {unknown}")


def _indexed_table(table: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in table:
        raise ValueError(f"predictor feature importance is missing {key!r}.")
    if table[key].duplicated().any():
        raise ValueError("predictor feature importance contains duplicates.")
    return table.set_index(key, drop=False)


def _resolve_threshold(
    feature: str,
    theoretical_cutoffs: tuple[float, ...],
    overrides: Mapping[str, float],
    cutoffs: pd.DataFrame,
    *,
    data_type: str,
) -> tuple[float, str, dict[str, object]]:
    if feature in overrides:
        return float(overrides[feature]), "override", {}
    if theoretical_cutoffs:
        midpoint = len(theoretical_cutoffs) // 2
        return float(theoretical_cutoffs[midpoint]), "theoretical", {}
    if not cutoffs.empty:
        required = {"feature", "cutoff"}
        missing = sorted(required - set(cutoffs.columns))
        if missing:
            raise ValueError(
                f"predictor cutoff candidates are missing columns: {missing}"
            )
        feature_rows = cutoffs.loc[cutoffs["feature"] == feature]
        if not feature_rows.empty:
            if "cutoff_rank" in feature_rows:
                feature_rows = feature_rows.sort_values(
                    "cutoff_rank",
                    kind="stable",
                )
            row = feature_rows.iloc[0].to_dict()
            return float(row["cutoff"]), "xgboost", row
    if data_type == "binary":
        return 0.5, "binary_default", {}
    raise ValueError(
        f"No crisp calibration threshold is available for {feature!r}. "
        "Provide threshold_overrides or theoretical_cutoffs."
    )


def _resolve_direction(
    feature: str,
    schema_direction: str,
    overrides: Mapping[str, str],
    importance: pd.DataFrame,
) -> tuple[CalibrationDirection, str]:
    if feature in overrides:
        return _normalize_direction(overrides[feature]), "override"
    if schema_direction != "infer":
        return _normalize_direction(schema_direction), "schema"
    if feature not in importance.index:
        raise ValueError(f"No feature importance is available for {feature!r}.")
    shap_direction = importance.loc[feature].get("shap_direction")
    if isinstance(shap_direction, str) and shap_direction in {"high", "low"}:
        return _normalize_direction(shap_direction), "shap"
    shap_mean = pd.to_numeric(
        pd.Series([importance.loc[feature].get("shap_mean")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(shap_mean) or float(shap_mean) == 0.0:
        raise ValueError(
            f"Cannot infer crisp calibration direction for {feature!r}. "
            "Provide direction_overrides or set MLConditionSpec.direction."
        )
    return ("high" if float(shap_mean) > 0.0 else "low"), "shap"


def _normalize_direction(value: str) -> CalibrationDirection:
    normalized = str(value).strip().lower()
    aliases = {
        "high": "high",
        "positive": "high",
        "ge": "high",
        ">=": "high",
        "low": "low",
        "negative": "low",
        "le": "low",
        "<=": "low",
    }
    direction = aliases.get(normalized)
    if direction is None:
        raise ValueError(
            f"Unknown crisp calibration direction {value!r}. Use high or low."
        )
    return direction  # type: ignore[return-value]


def _validate_proposals(
    proposals: pd.DataFrame,
    validated: ValidatedMLQCAInput,
) -> None:
    if not isinstance(proposals, pd.DataFrame):
        raise TypeError("proposals must be a pandas DataFrame.")
    required = {"feature", "output", "threshold", "direction"}
    missing = sorted(required - set(proposals.columns))
    if missing:
        raise ValueError(f"proposals is missing required columns: {missing}")
    if proposals.empty:
        raise ValueError("proposals must contain at least one condition.")
    if proposals["feature"].duplicated().any():
        raise ValueError("proposals contains duplicate features.")
    if proposals["output"].duplicated().any():
        raise ValueError("proposals contains duplicate output columns.")
    unknown = sorted(set(proposals["feature"]) - set(validated.candidates))
    if unknown:
        raise ValueError(f"proposals contains unknown conditions: {unknown}")
    collisions = sorted(set(proposals["output"]) & set(validated.data.columns))
    if collisions:
        raise ValueError(
            "Calibrated output columns already exist in data: "
            f"{collisions}"
        )
    for direction in proposals["direction"]:
        _normalize_direction(str(direction))


__all__ = [
    "CalibrationDirection",
    "CrispCalibrationSet",
    "calibrate_crisp_conditions",
    "propose_crisp_calibrations",
]
