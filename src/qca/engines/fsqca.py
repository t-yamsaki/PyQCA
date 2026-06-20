"""Fuzzy-set QCA engine."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from qca.engines._helpers import (
    normalize_engine_condition_types,
    resolve_engine_conditions,
)
from qca.engines.base import QCAEngineBase


class FSQCA(QCAEngineBase):
    """First-class fsQCA engine for calibrated set conditions."""

    qca_type = "fsQCA"

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        conditions: Sequence[str] | None = None,
        case_id: str | None = None,
        condition_types: dict[str, str] | None = None,
        set_conditions: Sequence[str] | None = None,
    ) -> None:
        resolved_conditions = resolve_engine_conditions(
            engine_name="FSQCA",
            conditions=conditions,
            legacy_conditions=set_conditions,
            legacy_name="set_conditions",
        )

        if condition_types is None:
            super().__init__(
                data=data,
                case_id=case_id,
                set_conditions=resolved_conditions,
                multivalue_conditions=[],
                outcome=outcome,
            )
            return

        normalized_types = normalize_engine_condition_types(
            engine_name="FSQCA",
            condition_types=condition_types,
            allowed_types={"crisp", "fuzzy"},
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
        self._validate_set_schema()

    def _validate_set_schema(self) -> None:
        non_set = [spec.name for spec in self.condition_specs if not spec.is_set]
        if non_set:
            raise ValueError(
                f"FSQCA requires crisp/fuzzy set condition specs. Non-set: {non_set}"
            )
        if self.multivalue_conditions:
            raise ValueError(
                "FSQCA does not support multi-value conditions. "
                f"Got {self.multivalue_conditions}."
            )

    def __repr__(self) -> str:
        return (
            f"FSQCA("
            f"n_cases={self.n_cases}, "
            f"conditions={self.conditions}, "
            f"outcome={self.outcome!r}"
            f")"
        )


__all__ = ["FSQCA"]
