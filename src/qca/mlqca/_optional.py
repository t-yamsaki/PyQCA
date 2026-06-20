"""Optional dependency helpers for machine-learning-enhanced QCA."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

MLQCAExtra = Literal["mlqca", "mlqca-explain"]

_DEPENDENCIES: dict[MLQCAExtra, tuple[str, ...]] = {
    "mlqca": ("xgboost", "sklearn"),
    "mlqca-explain": ("xgboost", "sklearn", "shap"),
}


def optional_dependency_status(
    extra: MLQCAExtra = "mlqca",
) -> dict[str, bool]:
    """Return availability flags for one mlQCA optional dependency group."""
    return {
        package: find_spec(package) is not None
        for package in _resolve_dependencies(extra)
    }


def mlqca_dependencies_available(extra: MLQCAExtra = "mlqca") -> bool:
    """Return whether all dependencies for ``extra`` are installed."""
    return all(optional_dependency_status(extra).values())


def require_mlqca_dependencies(extra: MLQCAExtra = "mlqca") -> None:
    """Raise an actionable error when optional mlQCA dependencies are absent."""
    missing = [
        package
        for package, available in optional_dependency_status(extra).items()
        if not available
    ]
    if missing:
        packages = ", ".join(missing)
        raise ImportError(
            f"mlQCA requires optional dependencies that are not installed: "
            f"{packages}. Install them with `pip install pyqca[{extra}]`."
        )


def _resolve_dependencies(extra: MLQCAExtra) -> tuple[str, ...]:
    try:
        return _DEPENDENCIES[extra]
    except KeyError as exc:
        available = ", ".join(sorted(_DEPENDENCIES))
        raise ValueError(
            f"Unknown mlQCA optional dependency group {extra!r}. "
            f"Available groups: {available}."
        ) from exc


__all__ = [
    "MLQCAExtra",
    "mlqca_dependencies_available",
    "optional_dependency_status",
    "require_mlqca_dependencies",
]
