"""Generalized-set QCA engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from qca.core.conditions import ConditionSpec, condition_specs_from_schema
from qca.engines.base import QCAEngineBase


class GSQCA(QCAEngineBase):
    """Generalized-set QCA over crisp, fuzzy, and multi-value conditions.

    ``GSQCA`` provides a generalized-set interface in which crisp-set,
    fuzzy-set, and multi-value QCA can be handled by a common truth-table and
    minimization workflow. The implementation is designed to make the
    generalized-set assumptions explicit in PyQCA rather than to claim
    one-to-one compatibility with any single external package. Multi-value
    conditions can be represented either by a crisp categorical column or by
    value-specific calibrated membership columns through
    :class:`qca.ConditionSpec.value_columns`.
    """

    qca_type = "GSQCA"
    workflow_prefix = "GSQCA"

    def __init__(
        self,
        data: pd.DataFrame,
        case_id: str | None = None,
        set_conditions: Sequence[str] | None = None,
        multivalue_conditions: Sequence[str] | None = None,
        outcome: str | None = None,
        conditions: Sequence[str] | None = None,
        condition_types: dict[str, str] | None = None,
        condition_specs: Any | None = None,
        schema: Any | None = None,
    ) -> None:
        if schema is not None and condition_specs is not None:
            raise ValueError("schema and condition_specs cannot be combined.")

        normalized_specs = (
            condition_specs_from_schema(schema)
            if schema is not None
            else condition_specs
        )
        super().__init__(
            data=data,
            case_id=case_id,
            set_conditions=set_conditions,
            multivalue_conditions=multivalue_conditions,
            outcome=outcome,
            conditions=conditions,
            condition_types=condition_types,
            condition_specs=normalized_specs,
        )

    @classmethod
    def from_schema(
        cls,
        data: pd.DataFrame,
        outcome: str,
        schema: Any,
        case_id: str | None = None,
    ) -> GSQCA:
        """Build ``GSQCA`` from a unified condition schema."""
        return cls(data=data, case_id=case_id, outcome=outcome, schema=schema)

    @classmethod
    def from_condition_specs(
        cls,
        data: pd.DataFrame,
        outcome: str,
        condition_specs: Sequence[ConditionSpec],
        case_id: str | None = None,
    ) -> GSQCA:
        """Build ``GSQCA`` from normalized ``ConditionSpec`` objects."""
        return cls(
            data=data,
            case_id=case_id,
            outcome=outcome,
            condition_specs=condition_specs,
        )

    @property
    def schema(self) -> pd.DataFrame:
        """Alias for the normalized condition schema."""
        return self.condition_schema

    @property
    def workflow_kinds(self) -> tuple[str, ...]:
        """Condition kinds present in this workflow, in canonical order."""
        present = set(self.condition_types.values())
        return tuple(kind for kind in ("crisp", "fuzzy", "multi") if kind in present)

    @property
    def workflow(self) -> str:
        """Human-readable workflow label derived from the condition schema."""
        kinds = self.workflow_kinds
        if len(kinds) == 1:
            return f"{self.workflow_prefix}-{kinds[0]}-only"
        return f"{self.workflow_prefix}-{'-'.join(kinds)}"

    @property
    def is_generalized_workflow(self) -> bool:
        """Whether the model combines more than one condition kind."""
        return len(self.workflow_kinds) > 1

    @property
    def has_crisp_conditions(self) -> bool:
        """Return whether crisp-set conditions are present."""
        return "crisp" in self.workflow_kinds

    @property
    def has_fuzzy_conditions(self) -> bool:
        """Return whether fuzzy-set conditions are present."""
        return "fuzzy" in self.workflow_kinds

    @property
    def has_multivalue_conditions(self) -> bool:
        """Return whether multi-value conditions are present."""
        return "multi" in self.workflow_kinds

    def _validate(self) -> None:
        super()._validate()
        if not self.conditions:
            raise ValueError("GSQCA requires at least one condition.")

    def __repr__(self) -> str:
        return (
            "GSQCA("
            f"n_cases={self.n_cases}, "
            f"workflow={self.workflow!r}, "
            f"conditions={self.conditions}, "
            f"outcome={self.outcome!r}"
            ")"
        )


__all__ = ["GSQCA"]
