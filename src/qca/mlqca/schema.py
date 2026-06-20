"""Input schema objects for machine-learning-enhanced QCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MLConditionDataType = Literal["continuous", "ordinal", "binary", "nominal"]
MLConditionDirection = Literal["high", "low", "infer"]


@dataclass(frozen=True)
class MLConditionSpec:
    """Describe one uncalibrated candidate condition for mlQCA."""

    name: str
    data_type: MLConditionDataType = "continuous"
    direction: MLConditionDirection = "infer"
    required: bool = False
    enabled: bool = True
    theoretical_cutoffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        data_type = str(self.data_type).strip().lower()
        direction = str(self.direction).strip().lower()
        if not name:
            raise ValueError("MLConditionSpec.name must be a non-empty string.")
        if data_type not in {"continuous", "ordinal", "binary", "nominal"}:
            raise ValueError(
                "MLConditionSpec.data_type must be continuous, ordinal, "
                f"binary, or nominal. Got {self.data_type!r}."
            )
        if direction not in {"high", "low", "infer"}:
            raise ValueError(
                "MLConditionSpec.direction must be high, low, or infer. "
                f"Got {self.direction!r}."
            )

        cutoffs = tuple(float(value) for value in self.theoretical_cutoffs)
        if any(not _is_finite(value) for value in cutoffs):
            raise ValueError("theoretical_cutoffs must contain finite values.")
        if len(set(cutoffs)) != len(cutoffs):
            raise ValueError("theoretical_cutoffs cannot contain duplicates.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "theoretical_cutoffs", tuple(sorted(cutoffs)))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable schema record."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "direction": self.direction,
            "required": self.required,
            "enabled": self.enabled,
            "theoretical_cutoffs": list(self.theoretical_cutoffs),
        }


def _is_finite(value: float) -> bool:
    return value not in {float("inf"), float("-inf")} and value == value


__all__ = [
    "MLConditionDataType",
    "MLConditionDirection",
    "MLConditionSpec",
]
