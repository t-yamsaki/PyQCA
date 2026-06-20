"""Shared constants for PyQCA."""

from __future__ import annotations

# Fuzzy-set membership values

#: Fuzzy-set crossover point, where membership is maximally ambiguous.
CROSSOVER_POINT: float = 0.5

#: Threshold used to classify a membership score as in-set or out-of-set.
MEMBERSHIP_THRESHOLD: float = 0.5

#: Complete fuzzy-set membership.
FULL_MEMBERSHIP: float = 1.0

#: Complete fuzzy-set non-membership.
NO_MEMBERSHIP: float = 0.0


# Floating-point tolerances

#: Tolerance used when detecting values at the fuzzy-set crossover point.
CROSSOVER_EPSILON: float = 1e-9

#: Small positive value used to prevent division by zero.
ZERO_EPSILON: float = 1e-12


# Consistency thresholds

#: Reference threshold for strong sufficiency consistency.
CONSISTENCY_STRONG: float = 0.85

#: Reference threshold for moderate sufficiency consistency.
CONSISTENCY_MODERATE: float = 0.75

#: Reference threshold for necessity consistency.
CONSISTENCY_NECESSITY: float = 0.90


# Coverage thresholds

#: Reference minimum coverage for retaining a non-trivial solution.
COVERAGE_MINIMUM: float = 0.10


# Direct-calibration membership anchors

#: Default membership assigned to the fully-out calibration anchor.
CALIBRATION_FULL_OUT: float = 0.05

#: Default membership assigned to the fully-in calibration anchor.
CALIBRATION_FULL_IN: float = 0.95


# Truth-table row classifications

#: Row satisfying the positive-outcome inclusion rule.
ROW_POSITIVE: str = "positive"

#: Row satisfying the negative-outcome exclusion rule.
ROW_NEGATIVE: str = "negative"

#: Logically possible but empirically unobserved row.
ROW_REMAINDER: str = "remainder"

#: Row with conflicting positive and negative evidence.
ROW_CONTRADICTION: str = "contradiction"


# Quine-McCluskey representation

#: Wildcard marking a condition eliminated from an implicant.
DONT_CARE: str = "*"


# Visualization palette

#: Color for high consistency or coverage.
COLOR_HIGH: str = "#2ecc71"

#: Color for medium consistency or coverage.
COLOR_MEDIUM: str = "#f39c12"

#: Color for low consistency or coverage.
COLOR_LOW: str = "#e74c3c"

#: General accent color.
COLOR_ACCENT: str = "#3498db"

#: Light grid color.
COLOR_GRID: str = "#ecf0f1"

#: Green highlight color retained for report and plot compatibility.
COLOR_GREEN: str = "#2ecc71"


# Reporting

#: Default text report width in characters.
REPORT_WIDTH: int = 80
