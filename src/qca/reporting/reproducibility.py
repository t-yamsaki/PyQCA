"""Reproducibility metadata and experiment logging."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qca._version import __version__


def collect_reproducibility_metadata(
    obj: Any | None = None,
    *,
    name: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect JSON-serializable reproducibility metadata for an analysis object."""
    metadata: dict[str, Any] = {
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "qca_version": __version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    if obj is not None:
        metadata.update(_analysis_object_metadata(obj))
    if extra:
        metadata["extra"] = _jsonable(dict(extra))
    return _jsonable(metadata)


@dataclass
class ExperimentLogger:
    """Append-only JSONL experiment logger."""

    path: str | Path

    def log(
        self,
        obj: Any | None = None,
        *,
        name: str | None = None,
        notes: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one experiment record and return the stored metadata."""
        record = collect_reproducibility_metadata(obj, name=name, extra=extra)
        if notes is not None:
            record["notes"] = notes
        output_path = Path(self.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def read(self) -> list[dict[str, Any]]:
        """Read all JSONL records from the experiment log."""
        input_path = Path(self.path)
        if not input_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    records.append(json.loads(value))
        return records


def log_experiment(
    obj: Any | None,
    path: str | Path,
    *,
    name: str | None = None,
    notes: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one experiment record to ``path``."""
    return ExperimentLogger(path).log(obj, name=name, notes=notes, extra=extra)


def _analysis_object_metadata(obj: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "object_type": type(obj).__name__,
    }

    for attr in ("qca_type", "outcome", "case_id", "selected_solution_type"):
        if hasattr(obj, attr):
            metadata[attr] = getattr(obj, attr)

    if hasattr(obj, "settings"):
        metadata["settings"] = obj.settings
    if hasattr(obj, "workflow"):
        metadata["workflow"] = obj.workflow
    if hasattr(obj, "consistency"):
        metadata["consistency"] = obj.consistency
    if hasattr(obj, "coverage"):
        metadata["coverage"] = obj.coverage
    if hasattr(obj, "condition_schema"):
        schema = obj.condition_schema
        if isinstance(schema, pd.DataFrame):
            metadata["condition_schema"] = schema.to_dict(orient="records")
    if hasattr(obj, "truth_table"):
        truth_table = obj.truth_table
        if isinstance(truth_table, pd.DataFrame):
            metadata["n_truth_table_rows"] = len(truth_table)
    if hasattr(obj, "solutions"):
        solutions = obj.solutions
        if isinstance(solutions, pd.DataFrame):
            metadata["n_solution_rows"] = len(solutions)
    if hasattr(obj, "data"):
        data = obj.data
        if isinstance(data, pd.DataFrame):
            metadata["data_shape"] = list(data.shape)
            metadata["data_columns"] = list(data.columns)
    if hasattr(obj, "conditions"):
        metadata["conditions"] = list(obj.conditions)
    if hasattr(obj, "n_cases"):
        metadata["n_cases"] = obj.n_cases
    if hasattr(obj, "n_conditions"):
        metadata["n_conditions"] = obj.n_conditions

    return metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.tolist())
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ExperimentLogger",
    "collect_reproducibility_metadata",
    "log_experiment",
]
