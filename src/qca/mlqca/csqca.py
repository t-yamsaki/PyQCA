"""csQCA connection and necessity analysis for mlQCA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qca.core.literals import SetLiteral
from qca.engines import CSQCA
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.calibration import (
    CrispCalibrationSet,
    calibrate_crisp_conditions,
    propose_crisp_calibrations,
)
from qca.mlqca.validation import ValidatedMLQCAInput
from qca.results import QCAFitResult


@dataclass(frozen=True)
class MLQCACSQCAResult:
    """Result of fitting csQCA to mlQCA-calibrated conditions."""

    calibration: CrispCalibrationSet
    engine: CSQCA
    necessity_table: pd.DataFrame
    fit_result: QCAFitResult

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CrispCalibrationSet):
            raise TypeError("calibration must be a CrispCalibrationSet.")
        if not isinstance(self.engine, CSQCA):
            raise TypeError("engine must be a CSQCA object.")
        if not isinstance(self.necessity_table, pd.DataFrame):
            raise TypeError("necessity_table must be a pandas DataFrame.")
        if not isinstance(self.fit_result, QCAFitResult):
            raise TypeError("fit_result must be a QCAFitResult.")
        object.__setattr__(
            self,
            "necessity_table",
            self.necessity_table.copy(),
        )

    @property
    def condition_schema(self) -> pd.DataFrame:
        return self.fit_result.condition_schema

    @property
    def consistency(self) -> float | None:
        return self.fit_result.consistency

    @property
    def coverage(self) -> float | None:
        return self.fit_result.coverage


def build_necessity_table(
    engine: CSQCA,
    calibration: CrispCalibrationSet | None = None,
    *,
    include_negations: bool = True,
) -> pd.DataFrame:
    """Evaluate necessity for every calibrated condition."""
    if not isinstance(engine, CSQCA):
        raise TypeError("engine must be a CSQCA object.")

    source_map = (
        {}
        if calibration is None
        else {
            output: source
            for source, output in calibration.condition_map.items()
        }
    )
    rows: list[dict[str, Any]] = []
    for condition in engine.conditions:
        for negated in (False, True) if include_negations else (False,):
            result = engine.evaluate_necessity(
                SetLiteral(condition, negated=negated)
            )
            row = result.to_dict()
            row.update(
                {
                    "source_feature": source_map.get(condition, condition),
                    "calibrated_condition": condition,
                    "presence": "absent" if negated else "present",
                    "negated": negated,
                }
            )
            rows.append(row)

    columns = [
        "source_feature",
        "calibrated_condition",
        "condition",
        "presence",
        "negated",
        "consistency",
        "coverage",
        "n_cases_in",
        "is_necessary",
        "is_trivial",
        "strength",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["consistency", "coverage", "source_feature", "negated"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def fit_csqca_from_calibration(
    calibration: CrispCalibrationSet,
    *,
    include_negations_in_necessity: bool = True,
    **fit_kwargs: Any,
) -> MLQCACSQCAResult:
    """Build and fit CSQCA from mlQCA crisp calibration output."""
    if not isinstance(calibration, CrispCalibrationSet):
        raise TypeError("calibration must be a CrispCalibrationSet.")
    required = [*calibration.conditions, calibration.outcome]
    with_missing = [
        column
        for column in required
        if calibration.data[column].isna().any()
    ]
    if with_missing:
        raise ValueError(
            "CSQCA cannot fit calibrated data containing missing values: "
            f"{with_missing}"
        )

    engine = CSQCA(
        calibration.data,
        outcome=calibration.outcome,
        conditions=calibration.conditions,
        case_id=calibration.case_id,
    )
    necessity = build_necessity_table(
        engine,
        calibration,
        include_negations=include_negations_in_necessity,
    )
    fit_result = engine.fit(**fit_kwargs)
    return MLQCACSQCAResult(
        calibration=calibration,
        engine=engine,
        necessity_table=necessity,
        fit_result=fit_result,
    )


def fit_csqca_from_predictor(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    conditions: Sequence[str] | None = None,
    *,
    direction_overrides: Mapping[str, str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    output_suffix: str = "__crisp",
    include_negations_in_necessity: bool = True,
    **fit_kwargs: Any,
) -> MLQCACSQCAResult:
    """Calibrate predictor-selected conditions and fit CSQCA."""
    proposals = propose_crisp_calibrations(
        validated,
        predictor,
        conditions,
        direction_overrides=direction_overrides,
        threshold_overrides=threshold_overrides,
        output_suffix=output_suffix,
    )
    calibration = calibrate_crisp_conditions(validated, proposals)
    return fit_csqca_from_calibration(
        calibration,
        include_negations_in_necessity=include_negations_in_necessity,
        **fit_kwargs,
    )


__all__ = [
    "MLQCACSQCAResult",
    "build_necessity_table",
    "fit_csqca_from_calibration",
    "fit_csqca_from_predictor",
]
