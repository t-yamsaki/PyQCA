"""Crisp-set QCA engine."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from qca.engines._helpers import (
    normalize_engine_condition_types,
    resolve_engine_conditions,
)
from qca.engines.base import QCAEngineBase


class CSQCA(QCAEngineBase):
    """First-class csQCA engine for crisp binary set conditions."""

    qca_type = "csQCA"

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
            engine_name="CSQCA",
            conditions=conditions,
            legacy_conditions=set_conditions,
            legacy_name="set_conditions",
        )

        if condition_types is None:
            normalized_types = {condition: "crisp" for condition in resolved_conditions}
        else:
            normalized_types = normalize_engine_condition_types(
                engine_name="CSQCA",
                condition_types=condition_types,
                allowed_types={"crisp"},
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
        self._validate_crisp_schema()
        self._validate_crisp_columns([*self.conditions, self.outcome])

    def _validate_crisp_schema(self) -> None:
        non_crisp = [spec.name for spec in self.condition_specs if spec.kind != "crisp"]
        if non_crisp:
            raise ValueError(
                f"CSQCA requires crisp condition specs. Non-crisp: {non_crisp}"
            )
        if self.multivalue_conditions:
            raise ValueError(
                "CSQCA does not support multi-value conditions. "
                f"Got {self.multivalue_conditions}."
            )

    def _validate_crisp_columns(self, columns: Sequence[str]) -> None:
        for column in columns:
            try:
                values = (
                    pd.to_numeric(self.data[column], errors="raise")
                    .dropna()
                    .astype(float)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSQCA requires crisp/binary values in {column!r}."
                ) from exc

            if not values.isin([0.0, 1.0]).all():
                raise ValueError(f"CSQCA requires crisp/binary values in {column!r}.")

    def __repr__(self) -> str:
        return (
            f"CSQCA("
            f"n_cases={self.n_cases}, "
            f"conditions={self.conditions}, "
            f"outcome={self.outcome!r}"
            f")"
        )


__all__ = ["CSQCA"]
