"""フェーズ1: _constants.py / _types.py の動作確認テスト"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# _constants.py のテスト
# ---------------------------------------------------------------------------


class TestConstants:
    """定数値の整合性テスト。"""

    def test_membership_threshold_is_crossover_point(self):
        """MEMBERSHIP_THRESHOLD と CROSSOVER_POINT は同値であること。"""
        from qca._constants import CROSSOVER_POINT, MEMBERSHIP_THRESHOLD

        assert MEMBERSHIP_THRESHOLD == CROSSOVER_POINT

    def test_calibration_anchors_ordering(self):
        """キャリブレーションアンカーは FULL_OUT < CROSSOVER < FULL_IN の順。"""
        from qca._constants import (
            CALIBRATION_FULL_IN,
            CALIBRATION_FULL_OUT,
            CROSSOVER_POINT,
        )

        assert CALIBRATION_FULL_OUT < CROSSOVER_POINT < CALIBRATION_FULL_IN

    def test_consistency_thresholds_ordering(self):
        """一貫性閾値は MODERATE < STRONG < NECESSITY の順であること。"""
        from qca._constants import (
            CONSISTENCY_MODERATE,
            CONSISTENCY_NECESSITY,
            CONSISTENCY_STRONG,
        )

        assert CONSISTENCY_MODERATE < CONSISTENCY_STRONG < CONSISTENCY_NECESSITY

    def test_membership_bounds(self):
        """完全非帰属・完全帰属は [0, 1] の境界値であること。"""
        from qca._constants import FULL_MEMBERSHIP, NO_MEMBERSHIP

        assert NO_MEMBERSHIP == 0.0
        assert FULL_MEMBERSHIP == 1.0

    def test_row_labels_are_distinct(self):
        """真理表行の分類ラベルが互いに重複しないこと。"""
        from qca._constants import (
            ROW_CONTRADICTION,
            ROW_NEGATIVE,
            ROW_POSITIVE,
            ROW_REMAINDER,
        )

        labels = [ROW_POSITIVE, ROW_NEGATIVE, ROW_REMAINDER, ROW_CONTRADICTION]
        assert len(labels) == len(set(labels))

    def test_epsilons_are_positive(self):
        """イプシロン値は正の値であること。"""
        from qca._constants import CROSSOVER_EPSILON, ZERO_EPSILON

        assert CROSSOVER_EPSILON > 0
        assert ZERO_EPSILON > 0

    def test_epsilons_ordering(self):
        """ZERO_EPSILON < CROSSOVER_EPSILON であること（用途の粒度が異なる）。"""
        from qca._constants import CROSSOVER_EPSILON, ZERO_EPSILON

        assert ZERO_EPSILON < CROSSOVER_EPSILON

    def test_color_strings_are_hex(self):
        """カラー定数は '#RRGGBB' 形式であること。"""
        from qca._constants import (
            COLOR_ACCENT,
            COLOR_GRID,
            COLOR_HIGH,
            COLOR_LOW,
            COLOR_MEDIUM,
        )

        for color in [COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW, COLOR_ACCENT, COLOR_GRID]:
            assert color.startswith("#"), f"{color!r} は '#' で始まる必要があります"
            assert len(color) == 7, (
                f"{color!r} は '#RRGGBB' の 7 文字である必要があります"
            )

    def test_dont_care_is_not_empty(self):
        """DONT_CARE は空文字列でないこと（条件値との混同を防ぐ）。"""
        from qca._constants import DONT_CARE

        assert DONT_CARE != ""
        assert isinstance(DONT_CARE, str)

    def test_report_width_is_positive(self):
        """REPORT_WIDTH は正の整数であること。"""
        from qca._constants import REPORT_WIDTH

        assert isinstance(REPORT_WIDTH, int)
        assert REPORT_WIDTH > 0


# ---------------------------------------------------------------------------
# _types.py のテスト
# ---------------------------------------------------------------------------


class TestTypes:
    """型定義・Protocol の動作確認テスト。"""

    def test_truth_table_row_protocol_satisfied_by_dataclass(self):
        """dataclass が TruthTableRowProtocol を満たすこと（runtime_checkable 確認）。"""
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
        """必須フィールドが欠けたオブジェクトは Protocol を満たさないこと。"""
        from qca._types import TruthTableRowProtocol

        class Incomplete:
            config = {"skill": 1}
            outcome_mean = 0.85
            # outcome_raw_consist が欠落

        assert not isinstance(Incomplete(), TruthTableRowProtocol)

    def test_calibration_result_protocol_satisfied(self):
        """CalibrationResultProtocol を満たすオブジェクトの確認。"""
        from qca._types import CalibrationResultProtocol

        class MockAnchors:
            anchor_quantiles = (0.05, 0.50, 0.95)
            full_out = 0.1
            crossover = 0.5
            full_in = 0.9

        assert isinstance(MockAnchors(), CalibrationResultProtocol)

    def test_type_aliases_are_importable(self):
        """型エイリアスが正常に import できること。"""
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
        """_types.py が実行時に core モジュールを import しないこと。
        （TYPE_CHECKING ガード内のみでの参照を確認）"""
        import sys

        # _types を import する前に core が未ロードであることを確認
        core_keys_before = {k for k in sys.modules if k.startswith("qca.core")}

        # _types を import
        import qca._types  # noqa: F401

        core_keys_after = {k for k in sys.modules if k.startswith("qca.core")}

        # _types の import によって core モジュールが新たにロードされていないこと
        newly_loaded = core_keys_after - core_keys_before
        assert not newly_loaded, (
            f"_types.py が実行時に core モジュールをロードしています: {newly_loaded}\n"
            "TYPE_CHECKING ガードの外に import が存在しないか確認してください。"
        )

    def test_implicant_pattern_alias(self):
        """ImplicantPattern が tuple 型エイリアスであること。"""
        from qca._types import ImplicantPattern

        # TypeAlias は実行時には元の型と等価
        sample: ImplicantPattern = (1, 0, "*")
        assert isinstance(sample, tuple)
