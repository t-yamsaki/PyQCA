"""Core constants and protocol tests."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestConstants:
    """Regression coverage."""

    def test_membership_threshold_is_crossover_point(self):
        """Regression coverage."""
        from qca._constants import CROSSOVER_POINT, MEMBERSHIP_THRESHOLD

        assert MEMBERSHIP_THRESHOLD == CROSSOVER_POINT

    def test_calibration_anchors_ordering(self):
        """Regression coverage."""
        from qca._constants import (
            CALIBRATION_FULL_IN,
            CALIBRATION_FULL_OUT,
            CROSSOVER_POINT,
        )

        assert CALIBRATION_FULL_OUT < CROSSOVER_POINT < CALIBRATION_FULL_IN

    def test_consistency_thresholds_ordering(self):
        """Regression coverage."""
        from qca._constants import (
            CONSISTENCY_MODERATE,
            CONSISTENCY_NECESSITY,
            CONSISTENCY_STRONG,
        )

        assert CONSISTENCY_MODERATE < CONSISTENCY_STRONG < CONSISTENCY_NECESSITY

    def test_membership_bounds(self):
        """Regression coverage."""
        from qca._constants import FULL_MEMBERSHIP, NO_MEMBERSHIP

        assert NO_MEMBERSHIP == 0.0
        assert FULL_MEMBERSHIP == 1.0

    def test_row_labels_are_distinct(self):
        """Regression coverage."""
        from qca._constants import (
            ROW_CONTRADICTION,
            ROW_NEGATIVE,
            ROW_POSITIVE,
            ROW_REMAINDER,
        )

        labels = [ROW_POSITIVE, ROW_NEGATIVE, ROW_REMAINDER, ROW_CONTRADICTION]
        assert len(labels) == len(set(labels))

    def test_epsilons_are_positive(self):
        """Regression coverage."""
        from qca._constants import CROSSOVER_EPSILON, ZERO_EPSILON

        assert CROSSOVER_EPSILON > 0
        assert ZERO_EPSILON > 0

    def test_epsilons_ordering(self):
        """Regression coverage."""
        from qca._constants import CROSSOVER_EPSILON, ZERO_EPSILON

        assert ZERO_EPSILON < CROSSOVER_EPSILON

    def test_color_strings_are_hex(self):
        """Regression coverage."""
        from qca._constants import (
            COLOR_ACCENT,
            COLOR_GRID,
            COLOR_HIGH,
            COLOR_LOW,
            COLOR_MEDIUM,
        )

        for color in [COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW, COLOR_ACCENT, COLOR_GRID]:
            assert color.startswith("#")
            assert len(color) == 7, f"{color!r} must use the #RRGGBB format."

    def test_dont_care_is_not_empty(self):
        """Regression coverage."""
        from qca._constants import DONT_CARE

        assert DONT_CARE != ""
        assert isinstance(DONT_CARE, str)

    def test_report_width_is_positive(self):
        """Regression coverage."""
        from qca._constants import REPORT_WIDTH

        assert isinstance(REPORT_WIDTH, int)
        assert REPORT_WIDTH > 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestTypes:
    """Regression coverage."""

    def test_truth_table_row_protocol_satisfied_by_dataclass(self):
        """Regression coverage."""
        from dataclasses import dataclass
        from typing import Any

        from qca._types import TruthTableRowProtocol

        @dataclass
        class MockRow:
            config: dict[str, Any]
            outcome_mean: float
            outcome_raw_consist: float

        row = MockRow(
            config={"skill": 1, "region": "north"},
            outcome_mean=0.85,
            outcome_raw_consist=0.82,
        )
        assert isinstance(row, TruthTableRowProtocol)

    def test_truth_table_row_protocol_not_satisfied_without_field(self):
        """Regression coverage."""
        from qca._types import TruthTableRowProtocol

        class Incomplete:
            config = {"skill": 1}
            outcome_mean = 0.85

        assert not isinstance(Incomplete(), TruthTableRowProtocol)

    def test_calibration_result_protocol_satisfied(self):
        """Regression coverage."""
        from qca._types import CalibrationResultProtocol

        class MockAnchors:
            anchor_quantiles = (0.05, 0.50, 0.95)
            full_out = 0.1
            crossover = 0.5
            full_in = 0.9

        assert isinstance(MockAnchors(), CalibrationResultProtocol)

    def test_type_aliases_are_importable(self):
        """Regression coverage."""
        from typing import get_origin

        from qca._types import (
            AnchorDict,
            ConditionDomains,
            ConfigDict,
            DirectionalExpectations,
            ImplicantPattern,
            IsMultiValueMap,
            MintermList,
        )

        assert get_origin(ConfigDict) is dict
        assert get_origin(DirectionalExpectations) is dict
        assert get_origin(ConditionDomains) is dict
        assert get_origin(IsMultiValueMap) is dict
        assert get_origin(AnchorDict) is dict
        assert get_origin(MintermList) is list
        assert get_origin(ImplicantPattern) is tuple

    def test_no_circular_import(self):
        """Regression coverage."""
        import sys

        core_keys_before = {k for k in sys.modules if k.startswith("qca.core")}

        import qca._types  # noqa: F401

        core_keys_after = {k for k in sys.modules if k.startswith("qca.core")}

        newly_loaded = core_keys_after - core_keys_before
        assert not newly_loaded, (
            f"_types.py loaded qca.core modules at runtime: {newly_loaded}\n"
            "Check that imports remain inside the TYPE_CHECKING guard."
        )

    def test_implicant_pattern_alias(self):
        """Regression coverage."""
        from qca._types import ImplicantPattern

        sample: ImplicantPattern = (1, 0, "*")
        assert isinstance(sample, tuple)
