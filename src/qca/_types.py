"""Shared type definitions for PyQCA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from qca.core.literals import MultiValueLiteral, SetLiteral


# Common mappings used by engines and minimizers.

#: A truth-table configuration keyed by condition name.
ConfigDict = dict[str, Any]

#: Expected condition directions used to construct intermediate solutions.
#: Example: ``{"skill": 1, "resources": 1, "region": "north"}``.
DirectionalExpectations = dict[str, Any]

#: Observed or allowed values for each condition.
#: Example: ``{"skill": [0, 1], "region": ["north", "south"]}``.
ConditionDomains = dict[str, list[Any]]

#: Flags identifying multi-value conditions.
#: Example: ``{"region": True, "skill": False}``.
IsMultiValueMap = dict[str, bool]


# Literal aliases are guarded to avoid a runtime import cycle between this
# module and ``qca.core.literals``.

if TYPE_CHECKING:
    #: A crisp/fuzzy set literal or a multi-value literal.
    Literal = SetLiteral | MultiValueLiteral


@runtime_checkable
class TruthTableRowProtocol(Protocol):
    """Protocol for one row of a QCA truth table."""

    config: ConfigDict
    outcome_mean: float
    outcome_raw_consist: float


@runtime_checkable
class CalibrationResultProtocol(Protocol):
    """Protocol for calibration results with anchor metadata."""

    anchor_quantiles: tuple[float, float, float]
    full_out: float
    crossover: float
    full_in: float


# Explicit aliases for function signatures and stored analysis data.

#: Anchor metadata returned by generic calibration helpers.
AnchorDict: TypeAlias = dict[str, Any]

#: Truth-table row indices used as minimization minterms.
MintermList: TypeAlias = list[int]

#: An implicant pattern containing condition values or the don't-care marker.
ImplicantPattern: TypeAlias = tuple[Any, ...]
