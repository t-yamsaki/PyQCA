"""Multi-value QCA engine."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from qca.engines._helpers import (
    normalize_engine_condition_types,
    resolve_engine_conditions,
)
from qca.engines.base import QCAEngineBase


class MVQCA(QCAEngineBase):
    """First-class mvQCA engine for categorical multi-value conditions."""

    qca_type = "mvQCA"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        conditions: Sequence[str] | None = None,
        case_id: str | None = None,
        condition_types: dict[str, str] | None = None,
        multivalue_conditions: Sequence[str] | None = None,
    ) -> None:
        resolved_conditions = resolve_engine_conditions(
            engine_name="MVQCA",
            conditions=conditions,
            legacy_conditions=multivalue_conditions,
            legacy_name="multivalue_conditions",
        )

        if condition_types is None:
            super().__init__(
                data=data,
                case_id=case_id,
                set_conditions=[],
                multivalue_conditions=resolved_conditions,
                outcome=outcome,
            )
            return

        normalized_types = normalize_engine_condition_types(
            engine_name="MVQCA",
            condition_types=condition_types,
            allowed_types={"multi"},
        )
        super().__init__(
            data=data,
            case_id=case_id,
            outcome=outcome,
            conditions=resolved_conditions,
            condition_types=normalized_types,
        )

    def _validate(self) -> None:
        super()._validate()
        self._validate_multivalue_schema()
        self._validate_multivalue_domains()

    def _validate_multivalue_schema(self) -> None:
        non_multi = [spec.name for spec in self.condition_specs if not spec.is_multi]
        if non_multi:
            raise ValueError(
                f"MVQCA requires multi-value condition specs. Non-multi: {non_multi}"
            )
        if self.set_conditions:
            raise ValueError(
                "MVQCA does not support crisp/fuzzy set conditions. "
                f"Got {self.set_conditions}."
            )

    def _validate_multivalue_domains(self) -> None:
        degenerate: list[str] = []
        for condition in self.multivalue_conditions:
            values = self.data[condition].dropna().unique()
            if len(values) < 2:
                degenerate.append(condition)
        if degenerate:
            raise ValueError(
                "MVQCA requires at least two observed values for each "
                f"multi-value condition. Degenerate: {degenerate}"
            )

    def __repr__(self) -> str:
        return (
            f"MVQCA("
            f"n_cases={self.n_cases}, "
            f"conditions={self.conditions}, "
            f"outcome={self.outcome!r}"
            f")"
        )


__all__ = ["MVQCA"]
