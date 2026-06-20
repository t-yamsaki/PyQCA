"""Mixed-type QCA facade.

``MixedQCA`` is the high-level engine for workflows that combine crisp,
fuzzy-set, and multi-value conditions. The shared implementation lives in
``qca.engines.base.QCAEngineBase`` so csQCA, fsQCA, mvQCA, and MixedQCA can be
first-class engines over the same core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from qca.core.conditions import ConditionSpec, condition_specs_from_schema
from qca.engines.base import QCAEngineBase


class MixedQCA(QCAEngineBase):
    """High-level facade for mixed crisp, fuzzy, and multi-value QCA.

    ``MixedQCA`` accepts all schema styles supported by the common engine:
    normalized ``ConditionSpec`` objects, ``conditions`` + ``condition_types``,
    and the legacy ``set_conditions`` / ``multivalue_conditions`` split. For
    v0.2 it also accepts a user-facing ``schema`` alias containing a list of
    dictionaries or a pandas DataFrame.
    """

    qca_type = "MixedQCA"

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
    ) -> MixedQCA:
        """Build ``MixedQCA`` from the v0.2 unified condition schema."""
        return cls(data=data, case_id=case_id, outcome=outcome, schema=schema)

    @classmethod
    def from_condition_specs(
        cls,
        data: pd.DataFrame,
        outcome: str,
        condition_specs: Sequence[ConditionSpec],
        case_id: str | None = None,
    ) -> MixedQCA:
        """Build ``MixedQCA`` from normalized ``ConditionSpec`` objects."""
        return cls(
            data=data,
            case_id=case_id,
            outcome=outcome,
            condition_specs=condition_specs,
        )

    @property
    def schema(self) -> pd.DataFrame:
        """Alias for the normalized v0.2 condition schema."""
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
            return f"{kinds[0]}-only"
        return f"mixed-{'-'.join(kinds)}"

    @property
    def is_mixed_workflow(self) -> bool:
        """Whether the model combines more than one condition kind."""
        return len(self.workflow_kinds) > 1

    @property
    def has_crisp_conditions(self) -> bool:
        return "crisp" in self.workflow_kinds

    @property
    def has_fuzzy_conditions(self) -> bool:
        return "fuzzy" in self.workflow_kinds

    @property
    def has_multivalue_conditions(self) -> bool:
        return "multi" in self.workflow_kinds

    def _validate(self) -> None:
        super()._validate()
        if not self.conditions:
            raise ValueError("MixedQCA requires at least one condition.")

    def __repr__(self) -> str:
        return (
            "MixedQCA("
            f"n_cases={self.n_cases}, "
            f"workflow={self.workflow!r}, "
            f"conditions={self.conditions}, "
            f"outcome={self.outcome!r}"
            ")"
        )


__all__ = ["MixedQCA"]
