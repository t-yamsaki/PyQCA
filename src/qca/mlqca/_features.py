"""Shared, metadata-aware resolution of XGBoost feature keys."""

from collections.abc import Sequence
from typing import Any


def normalize_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature.")
    if any(not name.strip() for name in names):
        raise ValueError("feature_names cannot contain empty names.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names cannot contain duplicates.")
    return names


class FeatureResolver:
    """Use named metadata, explicit unnamed metadata, or unambiguous aliases."""

    def __init__(self, booster: Any, names: tuple[str, ...]) -> None:
        missing = object()
        metadata = getattr(booster, "feature_names", missing)
        positional = {f"f{i}": name for i, name in enumerate(names)}
        self.ambiguous: set[str] = set()
        if metadata is missing:
            # Compatibility for lightweight adapters without model metadata.
            self.mapping = {name: name for name in names}
            for key, name in positional.items():
                if key in self.mapping and self.mapping[key] != name:
                    self.ambiguous.add(key)
                else:
                    self.mapping[key] = name
        elif metadata is None:
            self.mapping = positional
            num_features = getattr(booster, "num_features", None)
            if callable(num_features) and num_features() != len(names):
                raise ValueError("feature_names must match the model feature count.")
        else:
            model_names = normalize_feature_names(metadata)
            if model_names != names:
                raise ValueError(
                    "feature_names must match booster.feature_names in model order."
                )
            self.mapping = {name: name for name in names}

    def resolve(self, key: Any) -> str:
        key = str(key)
        if key in self.ambiguous:
            raise ValueError(
                f"Ambiguous XGBoost feature key {key!r}; "
                "provide booster.feature_names metadata."
            )
        if key not in self.mapping:
            raise ValueError(f"Unknown XGBoost feature key {key!r} for feature_names.")
        return self.mapping[key]
