"""XGBoost predictive backend for machine-learning-enhanced QCA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from qca.mlqca._optional import require_mlqca_dependencies
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.config import MLQCAConfig
from qca.mlqca.cutoffs import extract_split_cutoffs, xgboost_split_nodes
from qca.mlqca.importance import extract_xgboost_importance
from qca.mlqca.validation import ValidatedMLQCAInput


class XGBoostBackend:
    """Fit XGBoost and extract mlQCA condition and cutoff evidence."""

    name = "xgboost"

    def __init__(
        self,
        config: MLQCAConfig | None = None,
        *,
        model_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or MLQCAConfig()
        merged = dict(self.config.model_params)
        if model_params:
            merged.update(model_params)
        self.model_params = merged

    def fit(self, validated: ValidatedMLQCAInput) -> PredictorFitResult:
        """Fit a binary XGBoost classifier and extract model evidence."""
        if not isinstance(validated, ValidatedMLQCAInput):
            raise TypeError("validated must be a ValidatedMLQCAInput object.")
        require_mlqca_dependencies()

        xgb, metrics_module, sklearn_version = _load_xgboost_runtime()
        params = self._resolved_model_params()
        model = xgb.XGBClassifier(**params)
        features = validated.features
        target = validated.target
        model.fit(features, target)

        probabilities = np.asarray(model.predict_proba(features), dtype=float)[:, 1]
        predicted = np.asarray(model.predict(features), dtype=int)
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(
            features,
            feature_names=list(validated.candidates),
        )
        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
            validate_features=True,
        )
        split_nodes = xgboost_split_nodes(booster, validated.candidates)
        feature_importance = extract_xgboost_importance(
            booster,
            validated.candidates,
            contributions=contributions,
            feature_values=features,
            split_nodes=split_nodes,
        )
        cutoff_candidates = extract_split_cutoffs(
            booster,
            validated.candidates,
            max_cutoffs_per_condition=self.config.max_cutoffs_per_condition,
        )
        predictions = _prediction_frame(
            validated,
            probabilities=probabilities,
            predicted=predicted,
        )
        metrics = _training_metrics(
            target.to_numpy(dtype=int),
            probabilities=probabilities,
            predicted=predicted,
            metrics_module=metrics_module,
        )
        return PredictorFitResult(
            backend=self.name,
            model=model,
            feature_importance=feature_importance,
            cutoff_candidates=cutoff_candidates,
            predictions=predictions,
            metrics=metrics,
            settings={
                "backend": self.name,
                "feature_names": list(validated.candidates),
                "n_cases": validated.n_cases,
                "model_params": params,
                "importance_method": "xgboost_pred_contribs",
                "evaluation_scope": "training",
                "xgboost_version": xgb.__version__,
                "scikit_learn_version": sklearn_version,
            },
        )

    def _resolved_model_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "verbosity": 0,
        }
        defaults.update(self.model_params)
        return defaults


def fit_xgboost_predictor(
    validated: ValidatedMLQCAInput,
    config: MLQCAConfig | None = None,
    *,
    model_params: Mapping[str, Any] | None = None,
) -> PredictorFitResult:
    """Fit the standard mlQCA XGBoost backend."""
    return XGBoostBackend(config, model_params=model_params).fit(validated)


def _load_xgboost_runtime() -> tuple[Any, Any, str]:
    import sklearn
    import xgboost as xgb
    from sklearn import metrics

    return xgb, metrics, sklearn.__version__


def _prediction_frame(
    validated: ValidatedMLQCAInput,
    *,
    probabilities: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    if len(probabilities) != validated.n_cases or len(predicted) != validated.n_cases:
        raise ValueError("XGBoost prediction lengths do not match the input data.")
    frame = pd.DataFrame(
        {
            "row_index": validated.data.index,
            "observed": validated.target.to_numpy(dtype=int),
            "probability": probabilities,
            "predicted": predicted,
        }
    )
    if validated.case_id is not None:
        frame.insert(
            0,
            validated.case_id,
            validated.data[validated.case_id].to_numpy(copy=True),
        )
    return frame


def _training_metrics(
    target: np.ndarray,
    *,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    metrics_module: Any,
) -> dict[str, float]:
    return {
        "accuracy": float(metrics_module.accuracy_score(target, predicted)),
        "roc_auc": float(metrics_module.roc_auc_score(target, probabilities)),
        "log_loss": float(
            metrics_module.log_loss(target, probabilities, labels=[0, 1])
        ),
    }


__all__ = ["XGBoostBackend", "fit_xgboost_predictor"]
