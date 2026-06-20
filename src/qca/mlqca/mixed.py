"""Mixed-type calibration and MixedQCA connection for mlQCA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from qca.core.conditions import ConditionSpec
from qca.engines import MixedQCA
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.calibration import (
    calibrate_crisp_conditions,
    propose_crisp_calibrations,
)
from qca.mlqca.fuzzy import (
    calibrate_fuzzy_conditions,
    propose_fuzzy_anchors,
)
from qca.mlqca.validation import ValidatedMLQCAInput
from qca.results import QCAFitResult

MixedConditionKind = Literal["crisp", "fuzzy", "multi"]


@dataclass(frozen=True)
class MixedCalibrationSet:
    """Calibrated data and schema for a mixed-type mlQCA workflow."""

    data: pd.DataFrame
    outcome: str
    condition_specs: tuple[ConditionSpec, ...]
    proposals: pd.DataFrame
    source_map: dict[str, str]
    case_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(self.proposals, pd.DataFrame):
            raise TypeError("proposals must be a pandas DataFrame.")
        object.__setattr__(self, "data", self.data.copy())
        object.__setattr__(self, "condition_specs", tuple(self.condition_specs))
        object.__setattr__(self, "proposals", self.proposals.copy())
        object.__setattr__(self, "source_map", dict(self.source_map))

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.condition_specs)

    @property
    def condition_types(self) -> dict[str, str]:
        return {spec.name: spec.kind for spec in self.condition_specs}

    @property
    def workflow(self) -> str:
        kinds = {
            spec.kind for spec in self.condition_specs
        }
        ordered = [kind for kind in ("crisp", "fuzzy", "multi") if kind in kinds]
        return f"mixed-{'-'.join(ordered)}"


@dataclass(frozen=True)
class MLQCAMixedResult:
    """Result of fitting MixedQCA to mlQCA-calibrated conditions."""

    calibration: MixedCalibrationSet
    engine: MixedQCA
    fit_result: QCAFitResult

    @property
    def consistency(self) -> float | None:
        return self.fit_result.consistency

    @property
    def coverage(self) -> float | None:
        return self.fit_result.coverage


def calibrate_mixed_conditions(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    condition_kinds: Mapping[str, str],
    *,
    conditions: Sequence[str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    anchor_overrides: Mapping[str, Sequence[float]] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
) -> MixedCalibrationSet:
    """Calibrate crisp/fuzzy conditions and pass multi-value conditions through."""
    selected = (
        validated.candidates
        if conditions is None
        else tuple(str(value).strip() for value in conditions)
    )
    _validate_mixed_kinds(validated, selected, condition_kinds)
    normalized = {
        feature: _normalize_mixed_kind(condition_kinds[feature])
        for feature in selected
    }
    crisp = tuple(feature for feature in selected if normalized[feature] == "crisp")
    fuzzy = tuple(feature for feature in selected if normalized[feature] == "fuzzy")
    multi = tuple(feature for feature in selected if normalized[feature] == "multi")

    work = validated.data.copy()
    proposals: list[pd.DataFrame] = []
    specs_by_source: dict[str, ConditionSpec] = {}
    source_map: dict[str, str] = {}

    if crisp:
        crisp_proposals = propose_crisp_calibrations(
            validated,
            predictor,
            crisp,
            threshold_overrides=_subset(threshold_overrides, crisp),
            direction_overrides=_subset(direction_overrides, crisp),
        )
        crisp_set = calibrate_crisp_conditions(validated, crisp_proposals)
        for source, output in crisp_set.condition_map.items():
            work[output] = crisp_set.data[output]
            source_map[source] = output
            specs_by_source[source] = ConditionSpec(
                output,
                "crisp",
                domain=(0.0, 1.0),
                calibrated=True,
            )
        proposals.append(_tag_proposals(crisp_proposals, "crisp"))

    if fuzzy:
        fuzzy_proposals = propose_fuzzy_anchors(
            validated,
            predictor,
            fuzzy,
            anchor_overrides=_subset(anchor_overrides, fuzzy),
            direction_overrides=_subset(direction_overrides, fuzzy),
        )
        fuzzy_set = calibrate_fuzzy_conditions(validated, fuzzy_proposals)
        for source, output in fuzzy_set.condition_map.items():
            work[output] = fuzzy_set.data[output]
            source_map[source] = output
            specs_by_source[source] = ConditionSpec(
                output,
                "fuzzy",
                domain=(0.0, 1.0),
                calibrated=True,
            )
        proposals.append(_tag_proposals(fuzzy_proposals, "fuzzy"))

    for source in multi:
        source_map[source] = source
        domain = tuple(pd.unique(work[source].dropna()).tolist())
        specs_by_source[source] = ConditionSpec(
            source,
            "multi",
            domain=domain,
            calibrated=False,
        )
        proposals.append(
            pd.DataFrame(
                [
                    {
                        "feature": source,
                        "output": source,
                        "kind": "multi",
                        "calibration_source": "passthrough",
                    }
                ]
            )
        )

    return MixedCalibrationSet(
        data=work,
        outcome=validated.outcome,
        condition_specs=tuple(specs_by_source[source] for source in selected),
        proposals=pd.concat(proposals, ignore_index=True, sort=False),
        source_map=source_map,
        case_id=validated.case_id,
    )


def fit_mixedqca_from_calibration(
    calibration: MixedCalibrationSet,
    **fit_kwargs: Any,
) -> MLQCAMixedResult:
    """Build and fit MixedQCA from mixed mlQCA calibration output."""
    if not isinstance(calibration, MixedCalibrationSet):
        raise TypeError("calibration must be a MixedCalibrationSet.")
    required = [*calibration.conditions, calibration.outcome]
    missing = [
        column
        for column in required
        if calibration.data[column].isna().any()
    ]
    if missing:
        raise ValueError(
            "MixedQCA cannot fit calibrated data containing missing values: "
            f"{missing}"
        )
    engine = MixedQCA.from_condition_specs(
        calibration.data,
        outcome=calibration.outcome,
        condition_specs=calibration.condition_specs,
        case_id=calibration.case_id,
    )
    return MLQCAMixedResult(
        calibration=calibration,
        engine=engine,
        fit_result=engine.fit(**fit_kwargs),
    )


def fit_mixedqca_from_predictor(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    condition_kinds: Mapping[str, str],
    *,
    conditions: Sequence[str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    anchor_overrides: Mapping[str, Sequence[float]] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
    **fit_kwargs: Any,
) -> MLQCAMixedResult:
    """Calibrate a mixed schema and fit MixedQCA."""
    calibration = calibrate_mixed_conditions(
        validated,
        predictor,
        condition_kinds,
        conditions=conditions,
        threshold_overrides=threshold_overrides,
        anchor_overrides=anchor_overrides,
        direction_overrides=direction_overrides,
    )
    return fit_mixedqca_from_calibration(calibration, **fit_kwargs)


def _validate_mixed_kinds(
    validated: ValidatedMLQCAInput,
    selected: Sequence[str],
    condition_kinds: Mapping[str, str],
) -> None:
    if not selected or any(not feature for feature in selected):
        raise ValueError("conditions must contain at least one condition.")
    if len(set(selected)) != len(selected):
        raise ValueError("conditions cannot contain duplicates.")
    unknown = sorted(set(selected) - set(validated.candidates))
    if unknown:
        raise ValueError(f"Unknown mlQCA conditions: {unknown}")
    missing = [feature for feature in selected if feature not in condition_kinds]
    if missing:
        raise ValueError(f"condition_kinds is missing entries for: {missing}")
    extra = sorted(set(condition_kinds) - set(selected))
    if extra:
        raise ValueError(f"condition_kinds contains unselected conditions: {extra}")
    for feature in selected:
        _normalize_mixed_kind(condition_kinds[feature])


def _normalize_mixed_kind(value: str) -> MixedConditionKind:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "crisp": "crisp",
        "csqca": "crisp",
        "fuzzy": "fuzzy",
        "fsqca": "fuzzy",
        "multi": "multi",
        "multi-value": "multi",
        "mvqca": "multi",
    }
    kind = aliases.get(normalized)
    if kind is None:
        raise ValueError(
            f"Unknown mixed condition kind {value!r}. "
            "Available kinds: crisp, fuzzy, multi."
        )
    return kind  # type: ignore[return-value]


def _subset(
    mapping: Mapping[str, Any] | None,
    conditions: Sequence[str],
) -> dict[str, Any] | None:
    if mapping is None:
        return None
    return {
        condition: mapping[condition]
        for condition in conditions
        if condition in mapping
    }


def _tag_proposals(proposals: pd.DataFrame, kind: str) -> pd.DataFrame:
    tagged = proposals.copy()
    tagged.insert(2, "kind", kind)
    tagged["calibration_source"] = tagged.get(
        "threshold_source",
        tagged.get("anchor_source"),
    )
    return tagged


__all__ = [
    "MLQCAMixedResult",
    "MixedCalibrationSet",
    "MixedConditionKind",
    "calibrate_mixed_conditions",
    "fit_mixedqca_from_calibration",
    "fit_mixedqca_from_predictor",
]
