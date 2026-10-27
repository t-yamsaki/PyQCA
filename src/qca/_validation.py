"""Shared validation for concrete category values."""

from typing import Any

from qca._constants import DONT_CARE


def validate_category_value(value: Any) -> None:
    """Reject the reserved wildcard, including inside category unions."""
    if isinstance(value, str) and value == DONT_CARE:
        raise ValueError(
            "Category '*' is reserved for the internal don't-care wildcard. "
            "Recode this category to a different value before analysis."
        )
    if isinstance(value, (tuple, frozenset)):
        for member in value:
            validate_category_value(member)
