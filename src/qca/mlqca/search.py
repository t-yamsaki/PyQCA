"""Condition-combination search and Pareto evaluation for mlQCA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.config import MLQCAConfig
from qca.mlqca.csqca import fit_csqca_from_predictor
from qca.mlqca.results import MLQCAResult
from qca.mlqca.validation import ValidatedMLQCAInput
from qca.results import QCAFitResult

SearchErrorMode = Literal["record", "raise"]

MODEL_COLUMNS = (
    "model_id",
    "mode",
    "conditions",
    "n_conditions",
    "selection_rank_sum",
    "valid",
    "consistency",
    "coverage",
    "selected_solution_type",
    "formula",
    "n_terms",
    "complexity",
    "n_necessary",
    "max_necessity_consistency",
    "error",
)


def generate_condition_combinations(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    config: MLQCAConfig,
    *,
    conservative_models: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Generate radical or conservative condition combinations."""
    _validate_search_inputs(validated, predictor, config)
    ranking = _condition_ranking(validated, predictor)
    available = set(validated.candidates)
    excluded = set(config.excluded_conditions)
    required = _required_conditions(validated, config)

    unknown_constraints = sorted((required | excluded) - available)
    if unknown_constraints:
        raise ValueError(
            "Search constraints contain unknown conditions: "
            f"{unknown_constraints}"
        )
    if required & excluded:
        overlap = sorted(required & excluded)
        raise ValueError(f"Required and excluded conditions overlap: {overlap}")
    if len(required) > config.model_size:
        raise ValueError(
            "The number of required conditions cannot exceed model_size."
        )

    if conservative_models is not None:
        if config.mode != "conservative":
            raise ValueError(
                "conservative_models can only be used in conservative mode."
            )
        models = _normalize_explicit_models(
            conservative_models,
            available=available,
            required=required,
            excluded=excluded,
            model_size=config.model_size,
        )
        _validate_model_count(models, config.max_models)
        return models

    if config.mode == "conservative" and not required:
        raise ValueError(
            "Conservative search requires MLConditionSpec.required, "
            "required_conditions, or explicit conservative_models."
        )

    required_ordered = tuple(
        feature for feature in validated.candidates if feature in required
    )
    ranked = [
        feature
        for feature in ranking["feature"].astype(str)
        if feature not in excluded
    ]
    pool = list(
        dict.fromkeys([*required_ordered, *ranked[: config.top_k]])
    )
    optional = [feature for feature in pool if feature not in required]
    n_optional = config.model_size - len(required)
    if len(optional) < n_optional:
        raise ValueError(
            "Not enough eligible conditions to construct the requested models."
        )

    models = tuple(
        tuple(
            feature
            for feature in pool
            if feature in required_ordered or feature in optional_combo
        )
        for optional_combo in combinations(optional, n_optional)
    )
    _validate_model_count(models, config.max_models)
    return models


def search_csqca_models(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    config: MLQCAConfig | None = None,
    *,
    conservative_models: Sequence[Sequence[str]] | None = None,
    direction_overrides: Mapping[str, str] | None = None,
    threshold_overrides: Mapping[str, float] | None = None,
    on_error: SearchErrorMode = "record",
    **fit_kwargs: Any,
) -> MLQCAResult:
    """Evaluate condition combinations with the PyQCA csQCA engine."""
    resolved_config = config or MLQCAConfig(mode="radical")
    _validate_search_inputs(validated, predictor, resolved_config)
    if on_error not in {"record", "raise"}:
        raise ValueError("on_error must be 'record' or 'raise'.")

    models = generate_condition_combinations(
        validated,
        predictor,
        resolved_config,
        conservative_models=conservative_models,
    )
    ranking = _condition_ranking(validated, predictor)
    rank_map = dict(
        zip(
            ranking["feature"].astype(str),
            ranking["selection_rank"].astype(int),
            strict=True,
        )
    )
    resolved_fit_kwargs = _resolve_fit_kwargs(resolved_config, fit_kwargs)

    rows: list[dict[str, Any]] = []
    qca_results: list[QCAFitResult | None] = []
    proposal_frames: list[pd.DataFrame] = []
    for index, conditions in enumerate(models, start=1):
        model_id = f"m{index:04d}"
        try:
            analysis = fit_csqca_from_predictor(
                validated,
                predictor,
                conditions,
                direction_overrides=_subset_mapping(
                    direction_overrides,
                    conditions,
                ),
                threshold_overrides=_subset_mapping(
                    threshold_overrides,
                    conditions,
                ),
                **resolved_fit_kwargs,
            )
            fit_result = analysis.fit_result
            row = _successful_model_row(
                model_id,
                resolved_config.mode,
                conditions,
                fit_result,
                analysis.necessity_table,
                rank_map,
            )
            proposals = analysis.calibration.proposals.copy()
            proposals.insert(0, "model_id", model_id)
            proposal_frames.append(proposals)
            qca_results.append(fit_result)
        except Exception as exc:
            if on_error == "raise":
                raise
            row = _failed_model_row(
                model_id,
                resolved_config.mode,
                conditions,
                rank_map,
                exc,
            )
            qca_results.append(None)
        rows.append(row)

    candidate_models = pd.DataFrame(rows, columns=list(MODEL_COLUMNS))
    pareto_models = pareto_frontier(candidate_models)
    best_model_id = _select_best_model(candidate_models)
    proposals = (
        pd.concat(proposal_frames, ignore_index=True)
        if proposal_frames
        else pd.DataFrame()
    )
    return MLQCAResult(
        feature_importance=predictor.feature_importance,
        cutoff_candidates=predictor.cutoff_candidates,
        calibration_proposals=proposals,
        condition_ranking=ranking,
        candidate_models=candidate_models,
        qca_results=tuple(qca_results),
        best_model_id=best_model_id,
        pareto_models=pareto_models,
        settings={
            **resolved_config.to_dict(),
            "search_backend": "csqca",
            "fit_kwargs": resolved_fit_kwargs,
            "n_generated_models": len(models),
            "pareto_objectives": {
                "maximize": ["consistency", "coverage"],
                "minimize": ["complexity"],
            },
        },
        metadata={
            "predictor_backend": predictor.backend,
            "predictor_metrics": predictor.metrics,
            "predictor_settings": predictor.settings,
        },
    )


def pareto_frontier(
    models: pd.DataFrame,
    *,
    maximize: Sequence[str] = ("consistency", "coverage"),
    minimize: Sequence[str] = ("complexity",),
) -> pd.DataFrame:
    """Return non-dominated valid models for the requested objectives."""
    if not isinstance(models, pd.DataFrame):
        raise TypeError("models must be a pandas DataFrame.")
    objectives = [*maximize, *minimize]
    missing = [column for column in objectives if column not in models]
    if missing:
        raise ValueError(f"models is missing Pareto objective columns: {missing}")

    eligible = models.copy()
    if "valid" in eligible:
        eligible = eligible.loc[eligible["valid"].fillna(False).astype(bool)]
    eligible = eligible.dropna(subset=objectives).reset_index(drop=True)
    if eligible.empty:
        result = models.iloc[0:0].copy()
        result["pareto_rank"] = pd.Series(dtype=int)
        return result

    values = eligible.loc[:, objectives].astype(float)
    is_pareto = np.ones(len(eligible), dtype=bool)
    maximize_count = len(maximize)
    for candidate_index in range(len(eligible)):
        candidate = values.iloc[candidate_index].to_numpy(dtype=float)
        for challenger_index in range(len(eligible)):
            if candidate_index == challenger_index:
                continue
            challenger = values.iloc[challenger_index].to_numpy(dtype=float)
            at_least_as_good = bool(
                np.all(challenger[:maximize_count] >= candidate[:maximize_count])
                and np.all(challenger[maximize_count:] <= candidate[maximize_count:])
            )
            strictly_better = bool(
                np.any(challenger[:maximize_count] > candidate[:maximize_count])
                or np.any(challenger[maximize_count:] < candidate[maximize_count:])
            )
            if at_least_as_good and strictly_better:
                is_pareto[candidate_index] = False
                break

    frontier = eligible.loc[is_pareto].copy()
    frontier = frontier.sort_values(
        ["coverage", "consistency", "complexity", "model_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    frontier["pareto_rank"] = range(1, len(frontier) + 1)
    return frontier


def _validate_search_inputs(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
    config: MLQCAConfig,
) -> None:
    if not isinstance(validated, ValidatedMLQCAInput):
        raise TypeError("validated must be a ValidatedMLQCAInput object.")
    if not isinstance(predictor, PredictorFitResult):
        raise TypeError("predictor must be a PredictorFitResult object.")
    if not isinstance(config, MLQCAConfig):
        raise TypeError("config must be an MLQCAConfig object.")
    if config.engine != "csqca":
        raise ValueError("P3 condition search currently supports engine='csqca'.")


def _condition_ranking(
    validated: ValidatedMLQCAInput,
    predictor: PredictorFitResult,
) -> pd.DataFrame:
    table = predictor.feature_importance.copy()
    if "feature" not in table:
        raise ValueError("predictor feature importance must contain 'feature'.")
    table = table.loc[table["feature"].isin(validated.candidates)].copy()
    present = set(table["feature"].astype(str))
    missing = [
        feature for feature in validated.candidates if feature not in present
    ]
    if missing:
        table = pd.concat(
            [table, pd.DataFrame({"feature": missing})],
            ignore_index=True,
        )

    if "shap_rank" in table and table["shap_rank"].notna().any():
        table = table.sort_values(
            ["shap_rank", "feature"],
            kind="stable",
            na_position="last",
        )
    elif "shap_mean_abs" in table and table["shap_mean_abs"].notna().any():
        table = table.sort_values(
            ["shap_mean_abs", "feature"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        )
    elif "split_count" in table:
        table = table.sort_values(
            ["split_count", "feature"],
            ascending=[False, True],
            kind="stable",
        )
    table = table.reset_index(drop=True)
    table["selection_rank"] = range(1, len(table) + 1)
    return table


def _required_conditions(
    validated: ValidatedMLQCAInput,
    config: MLQCAConfig,
) -> set[str]:
    schema_required = {
        spec.name for spec in validated.condition_specs if spec.required
    }
    return schema_required | set(config.required_conditions)


def _normalize_explicit_models(
    models: Sequence[Sequence[str]],
    *,
    available: set[str],
    required: set[str],
    excluded: set[str],
    model_size: int,
) -> tuple[tuple[str, ...], ...]:
    normalized: list[tuple[str, ...]] = []
    seen: set[frozenset[str]] = set()
    for model in models:
        conditions = tuple(str(value).strip() for value in model)
        if len(conditions) != model_size:
            raise ValueError(
                "Every conservative model must contain exactly "
                f"model_size={model_size} conditions."
            )
        if any(not value for value in conditions) or len(set(conditions)) != len(
            conditions
        ):
            raise ValueError(
                "Conservative models cannot contain empty or duplicate conditions."
            )
        unknown = sorted(set(conditions) - available)
        if unknown:
            raise ValueError(
                f"Conservative model contains unknown conditions: {unknown}"
            )
        missing_required = sorted(required - set(conditions))
        if missing_required:
            raise ValueError(
                "Conservative model is missing required conditions: "
                f"{missing_required}"
            )
        forbidden = sorted(excluded & set(conditions))
        if forbidden:
            raise ValueError(
                f"Conservative model contains excluded conditions: {forbidden}"
            )
        key = frozenset(conditions)
        if key not in seen:
            normalized.append(conditions)
            seen.add(key)
    if not normalized:
        raise ValueError("conservative_models must contain at least one model.")
    return tuple(normalized)


def _validate_model_count(
    models: Sequence[Sequence[str]],
    max_models: int,
) -> None:
    if len(models) > max_models:
        raise ValueError(
            f"Search would generate {len(models)} models, exceeding "
            f"max_models={max_models}."
        )


def _resolve_fit_kwargs(
    config: MLQCAConfig,
    fit_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = {
        "consistency_cutoff": config.consistency_cutoff,
        "frequency_cutoff": config.frequency_cutoff,
        "minimizer": config.minimizer,
    }
    resolved.update(fit_kwargs)
    return resolved


def _subset_mapping(
    mapping: Mapping[str, Any] | None,
    conditions: tuple[str, ...],
) -> dict[str, Any] | None:
    if mapping is None:
        return None
    return {
        condition: mapping[condition]
        for condition in conditions
        if condition in mapping
    }


def _successful_model_row(
    model_id: str,
    mode: str,
    conditions: tuple[str, ...],
    fit_result: QCAFitResult,
    necessity: pd.DataFrame,
    rank_map: Mapping[str, int],
) -> dict[str, Any]:
    selected = fit_result.selected_solution_type
    selected_terms = fit_result.solutions.loc[
        fit_result.solutions["solution_type"] == selected
    ]
    valid = fit_result.selected_solution is not None
    complexity = (
        int(selected_terms["complexity"].sum())
        if valid and not selected_terms.empty
        else pd.NA
    )
    necessary = necessity.loc[necessity["is_necessary"].astype(bool)]
    max_necessity = pd.to_numeric(
        necessity["consistency"],
        errors="coerce",
    ).max()
    return {
        "model_id": model_id,
        "mode": mode,
        "conditions": tuple(conditions),
        "n_conditions": len(conditions),
        "selection_rank_sum": sum(rank_map[c] for c in conditions),
        "valid": valid,
        "consistency": fit_result.consistency,
        "coverage": fit_result.coverage,
        "selected_solution_type": selected,
        "formula": fit_result.formula,
        "n_terms": len(selected_terms) if valid else 0,
        "complexity": complexity,
        "n_necessary": len(necessary),
        "max_necessity_consistency": (
            None if pd.isna(max_necessity) else float(max_necessity)
        ),
        "error": None,
    }


def _failed_model_row(
    model_id: str,
    mode: str,
    conditions: tuple[str, ...],
    rank_map: Mapping[str, int],
    error: Exception,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "mode": mode,
        "conditions": tuple(conditions),
        "n_conditions": len(conditions),
        "selection_rank_sum": sum(rank_map[c] for c in conditions),
        "valid": False,
        "consistency": None,
        "coverage": None,
        "selected_solution_type": "none",
        "formula": None,
        "n_terms": 0,
        "complexity": pd.NA,
        "n_necessary": 0,
        "max_necessity_consistency": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _select_best_model(models: pd.DataFrame) -> str | None:
    valid = models.loc[models["valid"].fillna(False).astype(bool)].copy()
    valid = valid.dropna(subset=["coverage", "consistency", "complexity"])
    if valid.empty:
        return None
    ordered = valid.sort_values(
        [
            "coverage",
            "consistency",
            "complexity",
            "selection_rank_sum",
            "model_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    return str(ordered.iloc[0]["model_id"])


__all__ = [
    "SearchErrorMode",
    "generate_condition_combinations",
    "pareto_frontier",
    "search_csqca_models",
]
