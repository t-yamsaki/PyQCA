"""Text formatting helpers for PyQCA reports."""

from __future__ import annotations

import unicodedata
from typing import Any

from qca._constants import (
    CONSISTENCY_MODERATE,
    CONSISTENCY_NECESSITY,
    CONSISTENCY_STRONG,
    COVERAGE_MINIMUM,
    REPORT_WIDTH,
)


def display_width(text: str) -> int:
    """Return display width, counting East Asian wide/fullwidth chars as two."""
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def ljust_display(text: str, width: int, fillchar: str = " ") -> str:
    """Left-pad based on display width."""
    value = str(text)
    return value + fillchar * max(0, width - display_width(value))


def rjust_display(text: str, width: int, fillchar: str = " ") -> str:
    """Right-pad based on display width."""
    value = str(text)
    return fillchar * max(0, width - display_width(value)) + value


def truncate_display(text: str, max_width: int, suffix: str = "...") -> str:
    """Truncate a string to a display width."""
    value = str(text)
    if display_width(value) <= max_width:
        return value

    suffix_width = display_width(suffix)
    result = ""
    current = 0
    for char in value:
        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if current + char_width + suffix_width > max_width:
            break
        result += char
        current += char_width
    return result + suffix


def header(title: str, width: int = REPORT_WIDTH, char: str = "=") -> str:
    """Return a top-level report header."""
    line = char * width
    return f"\n{line}\n{title}\n{line}"


def subheader(title: str, width: int = REPORT_WIDTH, char: str = "-") -> str:
    """Return a second-level report header."""
    line = char * width
    return f"\n{line}\n{title}\n{line}"


def section(title: str) -> str:
    """Return a compact section label."""
    return f"\n[{title}]"


def indent(text: str, n: int = 2) -> str:
    """Indent each line by n spaces."""
    pad = " " * n
    return "\n".join(pad + line for line in str(text).splitlines())


def horizontal_rule(width: int = REPORT_WIDTH, char: str = "-") -> str:
    """Return a horizontal rule."""
    return char * width


def consistency_label(value: float) -> str:
    """Return a coarse qualitative label for consistency."""
    if value >= CONSISTENCY_STRONG:
        return "strong"
    if value >= CONSISTENCY_MODERATE:
        return "moderate"
    if value >= COVERAGE_MINIMUM:
        return "weak"
    return "very weak"


def coverage_label(value: float) -> str:
    """Return a coarse qualitative label for coverage."""
    if value >= 0.50:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def necessity_label(consistency: float) -> str:
    """Return a coarse qualitative label for necessity consistency."""
    if consistency >= 0.95:
        return "almost always necessary"
    if consistency >= CONSISTENCY_NECESSITY:
        return "necessary"
    if consistency >= CONSISTENCY_MODERATE:
        return "weakly necessary"
    return "not necessary"


def strength_bar(value: float, width: int = 20) -> str:
    """Render a compact ASCII bar for a score in [0, 1]."""
    clipped = min(max(float(value), 0.0), 1.0)
    filled = int(round(clipped * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {clipped:.3f}"


def format_table_row(
    cells: list[str],
    widths: list[int],
    sep: str = " ",
    align: list[str] | None = None,
) -> str:
    """Format fixed-width table cells."""
    if align is None:
        align = ["l"] * len(cells)
    parts: list[str] = []
    for cell, width, direction in zip(cells, widths, align, strict=False):
        formatter = rjust_display if direction == "r" else ljust_display
        parts.append(formatter(str(cell), width))
    return sep.join(parts)


def format_key_value(
    key: str, value: Any, key_width: int = 18, indent_n: int = 2
) -> str:
    """Format a key/value report line."""
    return f"{' ' * indent_n}{ljust_display(key, key_width)}: {value}"
