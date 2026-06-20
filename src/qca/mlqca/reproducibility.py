"""Reproducibility metadata for mlQCA workflows."""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd

from qca._version import __version__
from qca.mlqca.backend import PredictorFitResult
from qca.mlqca.config import MLQCAConfig
from qca.mlqca.validation import ValidatedMLQCAInput


def collect_mlqca_reproducibility(
    validated: ValidatedMLQCAInput | None = None,
    predictor: PredictorFitResult | None = None,
    config: MLQCAConfig | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect JSON-friendly metadata for an mlQCA run."""
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "qca_version": __version__,
        "pandas_version": pd.__version__,
        "optional_dependencies": {
            name: _package_version(name)
            for name in ("xgboost", "scikit-learn", "shap", "matplotlib")
        },
    }
    if validated is not None:
        metadata["data"] = {
            "fingerprint_sha256": dataframe_fingerprint(validated.data),
            "shape": list(validated.data.shape),
            "columns": [str(column) for column in validated.data.columns],
            "outcome": validated.outcome,
            "case_id": validated.case_id,
            "candidates": list(validated.candidates),
        }
    if predictor is not None:
        metadata["predictor"] = {
            "backend": predictor.backend,
            "metrics": dict(predictor.metrics),
            "settings": dict(predictor.settings),
        }
    if config is not None:
        metadata["config"] = config.to_dict()
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of a DataFrame and its schema."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    digest = hashlib.sha256()
    digest.update("|".join(str(column) for column in data.columns).encode())
    digest.update("|".join(str(dtype) for dtype in data.dtypes).encode())
    hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


__all__ = [
    "collect_mlqca_reproducibility",
    "dataframe_fingerprint",
]
