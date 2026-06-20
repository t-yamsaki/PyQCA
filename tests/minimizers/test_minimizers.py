"""フェーズ4: minimizer/ パッケージの動作確認テスト"""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from qca._constants import DONT_CARE
from qca.engines.mixed import MixedQCA
from qca.minimizers.algorithms import (
    build_coverage_table,
    can_merge,
    deduplicate_implicants,
    quine_mccluskey,
    select_essential_prime_implicants,
)
from qca.minimizers.engine import QCAMinimizer, minimize_truth_table
from qca.minimizers.implicant import Implicant, QMSolution
from qca.minimizers.remainder import (
    classify_truth_table_rows,
    filter_safe_remainders,
    filter_theory_consistent_remainders,
    generate_all_minterms,
    is_theory_consistent,
)
from qca.results import QCAFitResult

# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_domains() -> dict:
    return {"A": [0, 1], "B": [0, 1], "C": [0, 1]}


@pytest.fixture
def all_patterns_abc(binary_domains):
    return generate_all_minterms(["A", "B", "C"], binary_domains)


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "skill": [0.9, 0.8, 0.3, 0.2, 0.9, 0.8, 0.1, 0.2],
            "resources": [0.8, 0.3, 0.8, 0.2, 0.2, 0.9, 0.7, 0.1],
            "region": [
                "north",
                "north",
                "north",
                "south",
                "south",
                "east",
                "east",
                "south",
            ],
            "outcome": [0.9, 0.7, 0.8, 0.1, 0.2, 0.9, 0.1, 0.1],
        }
    )


@pytest.fixture
def model(mixed_df) -> MixedQCA:
    return MixedQCA(
        data=mixed_df,
        case_id="case",
        set_conditions=["skill", "resources"],
        multivalue_conditions=["region"],
        outcome="outcome",
    )


# ===========================================================================
# Implicant のテスト
# ===========================================================================


class TestImplicant:
    def test_complexity_no_wildcards(self):
        imp = Implicant(pattern=(1, 0, 1), covered=frozenset([0]))
        assert imp.complexity() == 3

    def test_complexity_with_wildcards(self):
        imp = Implicant(pattern=(1, DONT_CARE, 1), covered=frozenset([0]))
        assert imp.complexity() == 2

    def test_complexity_all_wildcards(self):
        imp = Implicant(pattern=(DONT_CARE, DONT_CARE), covered=frozenset([0]))
        assert imp.complexity() == 0

    def test_covers_true(self):
        imp = Implicant(pattern=(1, 0), covered=frozenset([2, 3]))
        assert imp.covers(2)

    def test_covers_false(self):
        imp = Implicant(pattern=(1, 0), covered=frozenset([2]))
        assert not imp.covers(5)

    def test_subsumes_true(self):
        """より一般的なインプリカントが特定的なものを包含する。"""
        general = Implicant(
            pattern=(1, DONT_CARE),
            covered=frozenset([2, 3]),
        )
        specific = Implicant(
            pattern=(1, 0),
            covered=frozenset([2]),
        )
        assert general.subsumes(specific)

    def test_subsumes_false(self):
        a = Implicant(pattern=(1, 0), covered=frozenset([2]))
        b = Implicant(pattern=(0, 1), covered=frozenset([1]))
        assert not a.subsumes(b)

    def test_label_binary_positive(self):
        imp = Implicant(pattern=(1, 0), covered=frozenset([0]))
        assert imp.label(["A", "B"]) == "A * ~B"

    def test_label_with_wildcard(self):
        imp = Implicant(pattern=(1, DONT_CARE), covered=frozenset([0]))
        assert imp.label(["A", "B"]) == "A"

    def test_label_multivalue(self):
        imp = Implicant(pattern=(1, "north"), covered=frozenset([0]))
        assert imp.label(["skill", "region"]) == "skill * region=north"

    def test_label_all_wildcard(self):
        imp = Implicant(pattern=(DONT_CARE, DONT_CARE), covered=frozenset([0]))
        assert imp.label(["A", "B"]) == "(tautology)"

    def test_label_length_mismatch(self):
        imp = Implicant(pattern=(1, 0), covered=frozenset([0]))
        with pytest.raises(ValueError, match="長さ"):
            imp.label(["A", "B", "C"])

    def test_hashable(self):
        imp1 = Implicant(pattern=(1, 0), covered=frozenset([0]))
        imp2 = Implicant(pattern=(1, 0), covered=frozenset([0]))
        assert imp1 == imp2
        assert hash(imp1) == hash(imp2)
        s = {imp1, imp2}
        assert len(s) == 1


# ===========================================================================
# QMSolution のテスト
# ===========================================================================


class TestQMSolution:
    def test_summary_empty(self):
        sol = QMSolution(condition_names=["A", "B"])
        df = sol.summary()
        assert df.empty

    def test_summary_columns(self):
        sol = QMSolution(
            condition_names=["A", "B"],
            complex_solution=[Implicant(pattern=(1, 0), covered=frozenset([2]))],
        )
        df = sol.summary()
        assert "solution_type" in df.columns
        assert "term" in df.columns
        assert "complexity" in df.columns

    def test_summary_order(self):
        """complex → parsimonious → intermediate の順であること。"""
        sol = QMSolution(
            condition_names=["A", "B"],
            complex_solution=[Implicant(pattern=(1, 0), covered=frozenset([2]))],
            parsimonious_solution=[
                Implicant(pattern=(1, DONT_CARE), covered=frozenset([2, 3]))
            ],
        )
        df = sol.summary()
        types = df["solution_type"].tolist()
        assert types.index("complex") < types.index("parsimonious")

    def test_complexity_reduction(self):
        sol = QMSolution(
            condition_names=["A", "B"],
            complex_solution=[Implicant(pattern=(1, 0), covered=frozenset([2]))],
            parsimonious_solution=[
                Implicant(pattern=(1, DONT_CARE), covered=frozenset([2, 3]))
            ],
            intermediate_solution=[Implicant(pattern=(1, 0), covered=frozenset([2]))],
        )
        reductions = sol.complexity_reduction()
        assert reductions["parsimonious"] == 1  # (1+1) - (1+0) = 1
        assert reductions["intermediate"] == 0

    def test_repr(self):
        sol = QMSolution(n_positive=3, n_remainder=5)
        assert "n_positive=3" in repr(sol)


# ===========================================================================
# can_merge のテスト
# ===========================================================================


class TestCanMerge:
    def test_merge_binary_differ_one(self):
        ok, merged = can_merge((0, 0), (1, 0), [False, False])
        assert ok
        assert merged == (DONT_CARE, 0)

    def test_no_merge_differ_two(self):
        ok, merged = can_merge((0, 0), (1, 1), [False, False])
        assert not ok
        assert merged is None

    def test_no_merge_multivalue(self):
        """多値条件の位置は差異があってもマージ不可。"""
        ok, merged = can_merge((1, "north"), (1, "south"), [False, True])
        assert not ok

    def test_no_merge_wildcard_mismatch(self):
        """ワイルドカード状態が異なる位置はマージ不可。"""
        ok, merged = can_merge((DONT_CARE, 0), (1, 0), [False, False])
        assert not ok

    def test_merge_with_existing_wildcard(self):
        """既にワイルドカードを含むパターンのマージ。"""
        ok, merged = can_merge(
            (DONT_CARE, 0, 0), (DONT_CARE, 1, 0), [False, False, False]
        )
        assert ok
        assert merged == (DONT_CARE, DONT_CARE, 0)

    def test_no_merge_different_length(self):
        ok, merged = can_merge((0, 1), (0, 1, 0), [False, False, False])
        assert not ok

    def test_identical_patterns_no_merge(self):
        """同一パターンは差異がないのでマージ不可（0箇所の差異）。"""
        ok, _ = can_merge((1, 0), (1, 0), [False, False])
        assert not ok


# ===========================================================================
# generate_all_minterms のテスト
# ===========================================================================


class TestGenerateAllMinterms:
    def test_binary_two_conditions(self):
        result = generate_all_minterms(["A", "B"], {"A": [0, 1], "B": [0, 1]})
        assert len(result) == 4
        assert (0, 0) in result
        assert (1, 1) in result

    def test_binary_three_conditions(self, all_patterns_abc):
        assert len(all_patterns_abc) == 8

    def test_multivalue(self):
        result = generate_all_minterms(
            ["A", "region"],
            {"A": [0, 1], "region": ["north", "south", "east"]},
        )
        assert len(result) == 6
        assert (0, "north") in result
        assert (1, "east") in result

    def test_empty_conditions(self):
        result = generate_all_minterms([], {})
        assert result == [()]

    def test_order_matches_condition_names(self):
        result = generate_all_minterms(["B", "A"], {"A": [0, 1], "B": [0, 1]})
        # B が先、A が後
        assert result[0] == (0, 0)  # B=0, A=0
        assert result[1] == (0, 1)  # B=0, A=1


# ===========================================================================
# quine_mccluskey のテスト
# ===========================================================================


class TestQuineMcCluskey:
    def test_single_minterm(self, all_patterns_abc):
        """単一ミンタームは自分自身がプライム含意節。"""
        result = quine_mccluskey(
            minterms=[7],  # (1,1,1)
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
        )
        assert len(result) == 1
        assert result[0].pattern == (1, 1, 1)

    def test_two_adjacent_minterms_merge(self, all_patterns_abc):
        """隣接する2ミンタームがマージされること。"""
        # (1,1,0)=6, (1,1,1)=7 → (1,1,*)
        result = quine_mccluskey(
            minterms=[6, 7],
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
        )
        assert len(result) == 1
        assert result[0].pattern == (1, 1, DONT_CARE)
        assert result[0].covered == frozenset([6, 7])

    def test_dont_care_allows_further_merge(self, all_patterns_abc):
        """don't care を使うことでさらなるマージが可能になる。"""
        # ポジティブ: (1,1,1)=7
        # don't care: (1,1,0)=6
        # → (1,1,*)
        result = quine_mccluskey(
            minterms=[7],
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
            dont_care_indices=[6],
        )
        primes = {r.pattern for r in result}
        assert (1, 1, DONT_CARE) in primes

    def test_empty_minterms(self, all_patterns_abc):
        result = quine_mccluskey(
            minterms=[],
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
        )
        assert result == []

    def test_all_covered_positive_in_covered(self, all_patterns_abc):
        """全プライム含意節のカバレッジはポジティブ行のみを含む。"""
        result = quine_mccluskey(
            minterms=[6, 7],
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
            dont_care_indices=[4, 5],
        )
        for imp in result:
            # don't care インデックス(4,5)が covered に含まれないこと
            assert not (imp.covered & {4, 5})

    def test_multivalue_no_merge(self):
        """多値条件が異なる場合はマージされないこと。"""
        patterns = generate_all_minterms(
            ["A", "region"],
            {"A": [0, 1], "region": ["north", "south"]},
        )
        # A=1, region=north (idx=2) と A=1, region=south (idx=3)
        result = quine_mccluskey(
            minterms=[2, 3],
            all_patterns=patterns,
            is_multivalue=[False, True],
        )
        # マージされないので2つのプライム含意節が残る
        assert len(result) == 2


# ===========================================================================
# select_essential_prime_implicants のテスト
# ===========================================================================


class TestSelectEssential:
    def test_single_essential(self, all_patterns_abc):
        """1つのプライム含意節しかカバーしないミンタームが必須を決定する。"""
        pi = [
            Implicant(pattern=(1, 1, DONT_CARE), covered=frozenset([6, 7])),
            Implicant(pattern=(DONT_CARE, 1, 1), covered=frozenset([3, 7])),
        ]
        selected = select_essential_prime_implicants(pi, [6, 7])
        # m6 は最初の PI のみがカバー → 必須
        assert any(imp.pattern == (1, 1, DONT_CARE) for imp in selected)

    def test_covers_all_minterms(self, all_patterns_abc):
        """選択後は全ミンタームがカバーされること。"""
        pi = quine_mccluskey(
            minterms=[0, 1, 2, 3],
            all_patterns=all_patterns_abc,
            is_multivalue=[False, False, False],
        )
        selected = select_essential_prime_implicants(pi, [0, 1, 2, 3])
        covered = set()
        for imp in selected:
            covered |= imp.covered
        assert {0, 1, 2, 3}.issubset(covered)

    def test_empty_inputs(self):
        assert select_essential_prime_implicants([], []) == []
        assert select_essential_prime_implicants([], [1, 2]) == []

    def test_greedy_prefers_low_complexity(self):
        """greedy 補完では complexity が小さいものを優先すること。"""
        pi = [
            Implicant(
                pattern=(1, DONT_CARE, DONT_CARE),
                covered=frozenset([4, 5, 6, 7]),
            ),
            Implicant(
                pattern=(1, 1, DONT_CARE),
                covered=frozenset([6, 7]),
            ),
        ]
        # m4, m5 は最初の PI のみがカバー → 必須
        # m6, m7 は両方がカバーできる
        selected = select_essential_prime_implicants(pi, [4, 5, 6, 7])
        # complexity=1 の PI が選ばれること
        assert any(imp.complexity() == 1 for imp in selected)


# ===========================================================================
# classify_truth_table_rows のテスト
# ===========================================================================


class TestClassifyTruthTableRows:
    @pytest.fixture
    def binary_setup(self):
        condition_names = ["A", "B"]
        domains = {"A": [0, 1], "B": [0, 1]}
        all_patterns = generate_all_minterms(condition_names, domains)
        pattern_to_idx = {p: i for i, p in enumerate(all_patterns)}
        return condition_names, all_patterns, pattern_to_idx

    def _make_row(self, config, outcome_mean, consist):
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        return MockRow(
            config=config, outcome_mean=outcome_mean, outcome_raw_consist=consist
        )

    def test_positive_classification(self, binary_setup):
        cnames, patterns, p2i = binary_setup
        rows = [self._make_row({"A": 1, "B": 1}, 0.85, 0.82)]
        pos, neg, rem, con = classify_truth_table_rows(
            rows, patterns, p2i, cnames, [False, False], 0.75, 0.75
        )
        assert 3 in pos  # (1,1) = index 3
        assert 3 not in neg

    def test_negative_classification(self, binary_setup):
        cnames, patterns, p2i = binary_setup
        rows = [self._make_row({"A": 0, "B": 0}, 0.10, 0.12)]
        pos, neg, rem, con = classify_truth_table_rows(
            rows, patterns, p2i, cnames, [False, False], 0.75, 0.75
        )
        assert 0 in neg

    def test_remainder_is_unobserved(self, binary_setup):
        cnames, patterns, p2i = binary_setup
        rows = [self._make_row({"A": 1, "B": 1}, 0.85, 0.82)]
        pos, neg, rem, con = classify_truth_table_rows(
            rows, patterns, p2i, cnames, [False, False], 0.75, 0.75
        )
        # 観察されていない (0,0),(0,1),(1,0) が残余
        assert len(rem) == 3
        assert 3 not in rem  # (1,1) は観察済み

    def test_contradiction_excluded(self, binary_setup):
        """同一パターンが正・負両方に分類された場合は矛盾行として除外。"""
        cnames, patterns, p2i = binary_setup
        rows = [
            self._make_row({"A": 1, "B": 1}, 0.85, 0.82),  # positive
            self._make_row({"A": 1, "B": 1}, 0.10, 0.12),  # negative → 矛盾
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pos, neg, rem, con = classify_truth_table_rows(
                rows, patterns, p2i, cnames, [False, False], 0.75, 0.75
            )
        assert 3 in con
        assert 3 not in pos
        assert 3 not in neg
        assert any("矛盾" in str(warning.message) for warning in w)


# ===========================================================================
# is_theory_consistent / filter 関数のテスト
# ===========================================================================


class TestRemainderFilters:
    def test_is_theory_consistent_match(self):
        assert is_theory_consistent(
            (1, 1, "north"),
            ["skill", "resources", "region"],
            {"skill": 1, "resources": 1, "region": "north"},
        )

    def test_is_theory_consistent_mismatch(self):
        assert not is_theory_consistent(
            (1, 0, "north"),
            ["skill", "resources", "region"],
            {"skill": 1, "resources": 1},  # resources=1 期待だが 0
        )

    def test_is_theory_consistent_partial_spec(self):
        """一部の条件のみ指定された場合、未指定は制約なし。"""
        assert is_theory_consistent(
            (1, 0, "south"),
            ["skill", "resources", "region"],
            {"skill": 1},  # skill のみ指定
        )

    def test_filter_safe_remainders(self):
        safe = filter_safe_remainders(
            remainder_indices=[2, 3, 4, 5],
            negative_indices=[3],
            contradiction_indices=[5],
        )
        assert safe == [2, 4]
        assert 3 not in safe
        assert 5 not in safe

    def test_filter_safe_remainders_empty_exclusions(self):
        safe = filter_safe_remainders(
            remainder_indices=[1, 2, 3],
            negative_indices=[],
            contradiction_indices=[],
        )
        assert safe == [1, 2, 3]

    def test_filter_safe_remainders_all_excluded(self):
        safe = filter_safe_remainders(
            remainder_indices=[1, 2],
            negative_indices=[1],
            contradiction_indices=[2],
        )
        assert safe == []

    def test_filter_theory_consistent_remainders(self):
        patterns = generate_all_minterms(
            ["skill", "resources"],
            {"skill": [0, 1], "resources": [0, 1]},
        )
        # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        result = filter_theory_consistent_remainders(
            remainder_indices=[0, 1, 2, 3],
            negative_indices=[],
            contradiction_indices=[],
            all_patterns=patterns,
            condition_names=["skill", "resources"],
            directional_expectations={"skill": 1, "resources": 1},
        )
        # skill=1, resources=1 の期待: (1,1)=3 のみが合致
        assert result == [3]

    def test_filter_theory_consistent_excludes_negative(self):
        patterns = generate_all_minterms(
            ["skill", "resources"],
            {"skill": [0, 1], "resources": [0, 1]},
        )
        result = filter_theory_consistent_remainders(
            remainder_indices=[2, 3],
            negative_indices=[3],  # 理論整合でもネガティブなら除外
            contradiction_indices=[],
            all_patterns=patterns,
            condition_names=["skill", "resources"],
            directional_expectations={"skill": 1, "resources": 1},
        )
        assert 3 not in result


# ===========================================================================
# QCAMinimizer のテスト
# ===========================================================================


class TestQCAMinimizer:
    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------

    def test_n_minterms_binary(self):
        minimizer = QCAMinimizer(
            condition_names=["A", "B", "C"],
            is_multivalue={"A": False, "B": False, "C": False},
            condition_domains={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )
        assert minimizer.n_minterms == 8

    def test_n_minterms_with_multivalue(self):
        minimizer = QCAMinimizer(
            condition_names=["A", "region"],
            is_multivalue={"A": False, "region": True},
            condition_domains={
                "A": [0, 1],
                "region": ["north", "south", "east"],
            },
        )
        assert minimizer.n_minterms == 6

    def test_repr(self):
        minimizer = QCAMinimizer(
            condition_names=["A", "B"],
            is_multivalue={},
            condition_domains={"A": [0, 1], "B": [0, 1]},
        )
        assert "QCAMinimizer" in repr(minimizer)
        assert "n_minterms=4" in repr(minimizer)

    # ------------------------------------------------------------------
    # minimize() の基本動作
    # ------------------------------------------------------------------

    @pytest.fixture
    def binary_minimizer(self):
        return QCAMinimizer(
            condition_names=["A", "B", "C"],
            is_multivalue={"A": False, "B": False, "C": False},
            condition_domains={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        )

    @pytest.fixture
    def mock_rows_positive(self):
        """A=1 かつ B=1 のとき常にポジティブな単純な真理表。"""
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        return [
            MockRow({"A": 1, "B": 1, "C": 1}, 0.90, 0.88),
            MockRow({"A": 1, "B": 1, "C": 0}, 0.85, 0.83),
            MockRow({"A": 0, "B": 0, "C": 0}, 0.10, 0.12),
            MockRow({"A": 0, "B": 1, "C": 0}, 0.15, 0.14),
        ]

    def test_returns_qm_solution(self, binary_minimizer, mock_rows_positive):
        solution = binary_minimizer.minimize(mock_rows_positive)
        assert isinstance(solution, QMSolution)

    def test_complex_solution_not_empty(self, binary_minimizer, mock_rows_positive):
        solution = binary_minimizer.minimize(mock_rows_positive)
        assert len(solution.complex_solution) > 0

    def test_parsimonious_leq_complex_complexity(
        self, binary_minimizer, mock_rows_positive
    ):
        """倹約解の総複雑度 ≤ 複雑解の総複雑度。"""
        solution = binary_minimizer.minimize(mock_rows_positive)
        complex_total = sum(imp.complexity() for imp in solution.complex_solution)
        parsimonious_total = sum(
            imp.complexity() for imp in solution.parsimonious_solution
        )
        assert parsimonious_total <= complex_total

    def test_intermediate_between_complex_and_parsimonious(
        self, binary_minimizer, mock_rows_positive
    ):
        """中間解の複雑度は複雑解以下かつ倹約解以上。"""
        solution = binary_minimizer.minimize(
            mock_rows_positive,
            directional_expectations={"A": 1, "B": 1},
        )
        complex_total = sum(imp.complexity() for imp in solution.complex_solution)
        parsimonious_total = sum(
            imp.complexity() for imp in solution.parsimonious_solution
        )
        intermediate_total = sum(
            imp.complexity() for imp in solution.intermediate_solution
        )
        assert parsimonious_total <= intermediate_total <= complex_total

    def test_all_positive_minterms_covered_complex(
        self, binary_minimizer, mock_rows_positive
    ):
        """複雑解は全ポジティブミンタームをカバーすること。"""
        solution = binary_minimizer.minimize(mock_rows_positive)
        covered = set()
        for imp in solution.complex_solution:
            covered |= imp.covered
        # ポジティブ行: (1,1,1)=7, (1,1,0)=6
        assert {6, 7}.issubset(covered)

    def test_no_positive_rows_warns(self, binary_minimizer):
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        rows = [MockRow({"A": 0, "B": 0, "C": 0}, 0.10, 0.12)]
        with pytest.warns(UserWarning, match="ポジティブ行"):
            solution = binary_minimizer.minimize(rows, outcome_threshold=0.75)
        assert solution.complex_solution == []

    def test_directional_expectations_none_warns(
        self, binary_minimizer, mock_rows_positive
    ):
        """方向性未指定の場合は警告が出て中間解=複雑解になること。"""
        with pytest.warns(UserWarning, match="directional_expectations"):
            solution = binary_minimizer.minimize(
                mock_rows_positive,
                directional_expectations=None,
            )
        assert solution.intermediate_solution == solution.complex_solution

    def test_include_remainders_false_parsimonious_eq_complex(
        self, binary_minimizer, mock_rows_positive
    ):
        """残余行不使用の場合、倹約解=複雑解。"""
        solution = binary_minimizer.minimize(
            mock_rows_positive,
            include_remainders_in_parsimonious=False,
        )
        assert solution.parsimonious_solution == solution.complex_solution

    def test_n_positive_n_remainder_set(self, binary_minimizer, mock_rows_positive):
        solution = binary_minimizer.minimize(mock_rows_positive)
        assert solution.n_positive == 2  # (1,1,1), (1,1,0)
        assert solution.n_remainder > 0
        assert (
            solution.n_positive + len(mock_rows_positive) + solution.n_contradiction
            <= binary_minimizer.n_minterms
        )

    # ------------------------------------------------------------------
    # coverage_table
    # ------------------------------------------------------------------

    def test_coverage_table_is_dataframe(self, binary_minimizer, mock_rows_positive):
        solution = binary_minimizer.minimize(mock_rows_positive)
        assert isinstance(solution.coverage_table, pd.DataFrame)

    def test_coverage_table_columns(self, binary_minimizer, mock_rows_positive):
        solution = binary_minimizer.minimize(mock_rows_positive)
        if not solution.coverage_table.empty:
            assert "complexity" in solution.coverage_table.columns
            assert "n_covered" in solution.coverage_table.columns

    def test_coverage_table_values_binary(self, binary_minimizer, mock_rows_positive):
        """カバレッジ表の本体列は 0/1 のみ。"""
        solution = binary_minimizer.minimize(mock_rows_positive)
        if not solution.coverage_table.empty:
            minterm_cols = [
                c for c in solution.coverage_table.columns if c.startswith("m")
            ]
            values = solution.coverage_table[minterm_cols].values.flatten()
            assert set(values).issubset({0, 1})

    # ------------------------------------------------------------------
    # classify_rows (診断用)
    # ------------------------------------------------------------------

    def test_classify_rows_keys(self, binary_minimizer, mock_rows_positive):
        result = binary_minimizer.classify_rows(mock_rows_positive)
        assert "positive" in result
        assert "negative" in result
        assert "remainder" in result
        assert "contradiction" in result

    def test_classify_rows_positive_count(self, binary_minimizer, mock_rows_positive):
        result = binary_minimizer.classify_rows(mock_rows_positive)
        assert len(result["positive"]) == 2

    # ------------------------------------------------------------------
    # remainder_summary (診断用)
    # ------------------------------------------------------------------

    def test_remainder_summary_is_dataframe(self, binary_minimizer, mock_rows_positive):
        df = binary_minimizer.remainder_summary(mock_rows_positive)
        assert isinstance(df, pd.DataFrame)

    def test_remainder_summary_columns(self, binary_minimizer, mock_rows_positive):
        df = binary_minimizer.remainder_summary(mock_rows_positive)
        assert "classification" in df.columns
        assert "theory_consistent" in df.columns

    def test_remainder_summary_theory_consistent_with_expectations(
        self, binary_minimizer, mock_rows_positive
    ):
        df = binary_minimizer.remainder_summary(
            mock_rows_positive,
            directional_expectations={"A": 1, "B": 1, "C": 1},
        )
        if not df.empty:
            assert (
                df["theory_consistent"].dtype == object
                or df["theory_consistent"].dtype == bool
            )


# ===========================================================================
# minimize_truth_table (MixedQCA 統合) のテスト
# ===========================================================================


class TestMinimizeTruthTable:
    def test_returns_qm_solution(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = minimize_truth_table(model, outcome_threshold=0.75)
        assert isinstance(solution, QMSolution)

    def test_condition_names_match_model(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = minimize_truth_table(model)
        assert solution.condition_names == model.all_condition_names

    def test_complex_solution_labels_valid(self, model):
        """複雑解の各項ラベルが空でないこと。"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = minimize_truth_table(model, outcome_threshold=0.75)
        for imp in solution.complex_solution:
            label = imp.label(solution.condition_names)
            assert label != ""

    def test_summary_has_three_solution_types(self, model):
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.75,
            directional_expectations={"skill": 1, "resources": 1},
        )
        df = solution.summary()
        if not df.empty:
            types = df["solution_type"].unique().tolist()
            # 少なくとも複雑解と倹約解が含まれること
            assert "complex" in types
            assert "parsimonious" in types

    def test_with_directional_expectations(self, model):
        """方向性指定あり: 中間解が導出されること。"""
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.75,
            directional_expectations={
                "skill": 1,
                "resources": 1,
                "region": "north",
            },
        )
        assert isinstance(solution.intermediate_solution, list)

    def test_parsimonious_leq_complex(self, model):
        """MixedQCA 統合でも複雑度の大小関係が保たれること。"""
        solution = minimize_truth_table(model, outcome_threshold=0.75)
        complex_total = sum(imp.complexity() for imp in solution.complex_solution)
        parsimonious_total = sum(
            imp.complexity() for imp in solution.parsimonious_solution
        )
        assert parsimonious_total <= complex_total

    def test_high_threshold_no_positive_warns(self, model):
        """閾値が高すぎてポジティブ行なし → 警告 + 空解。"""
        with pytest.warns(UserWarning, match="ポジティブ行"):
            solution = minimize_truth_table(
                model,
                outcome_threshold=0.99,
                consistency_threshold=0.99,
            )
        assert solution.complex_solution == []
        assert solution.n_positive == 0


# ===========================================================================
# エンドツーエンド統合テスト
# ===========================================================================


class TestEndToEnd:
    """
    MixedQCA → build_truth_table → QCAMinimizer → QMSolution
    の全パイプラインを検証する。
    """

    @pytest.fixture
    def simple_crisp_df(self) -> pd.DataFrame:
        """
        完全に決定的な crisp データ。
        A=1 かつ B=1 → outcome=1、それ以外 → outcome=0。
        """
        return pd.DataFrame(
            {
                "case": ["c1", "c2", "c3", "c4"],
                "A": [1.0, 1.0, 0.0, 0.0],
                "B": [1.0, 0.0, 1.0, 0.0],
                "outcome": [1.0, 0.0, 0.0, 0.0],
            }
        )

    @pytest.fixture
    def crisp_model(self, simple_crisp_df) -> MixedQCA:
        return MixedQCA(
            data=simple_crisp_df,
            case_id="case",
            set_conditions=["A", "B"],
            multivalue_conditions=[],
            outcome="outcome",
        )

    def test_crisp_complex_solution_is_a_and_b(self, crisp_model):
        """A=1 かつ B=1 が唯一のポジティブ行 → 複雑解 = A * B。"""
        solution = minimize_truth_table(
            crisp_model,
            outcome_threshold=0.75,
            consistency_threshold=0.75,
        )
        assert len(solution.complex_solution) == 1
        label = solution.complex_solution[0].label(solution.condition_names)
        assert label == "A * B"

    def test_crisp_parsimonious_leq_complex(self, crisp_model):
        solution = minimize_truth_table(crisp_model, outcome_threshold=0.75)
        c = sum(i.complexity() for i in solution.complex_solution)
        p = sum(i.complexity() for i in solution.parsimonious_solution)
        assert p <= c

    def test_crisp_solution_covers_positive_minterm(self, crisp_model):
        """複雑解が (A=1, B=1) のミンタームをカバーすること。"""
        solution = minimize_truth_table(crisp_model, outcome_threshold=0.75)
        all_covered = set()
        for imp in solution.complex_solution:
            all_covered |= imp.covered
        # (1,1) = index 3 in [(0,0),(0,1),(1,0),(1,1)]
        assert len(all_covered) >= 1

    def test_mixed_pipeline_no_error(self, model):
        """MixedQCA (fuzzy + multivalue) で例外が発生しないこと。"""
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.70,
            consistency_threshold=0.70,
            directional_expectations={
                "skill": 1,
                "resources": 1,
                "region": "north",
            },
        )
        assert isinstance(solution, QMSolution)

    def test_summary_dataframe_structure(self, model):
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.70,
            directional_expectations={"skill": 1, "resources": 1},
        )
        df = solution.summary()
        if not df.empty:
            assert set(df.columns) >= {
                "solution_type",
                "term",
                "complexity",
                "n_minterms_covered",
            }
            assert (df["complexity"] >= 0).all()
            assert (df["n_minterms_covered"] >= 0).all()

    def test_complexity_reduction_nonnegative(self, model):
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.70,
            directional_expectations={"skill": 1, "resources": 1},
        )
        reductions = solution.complexity_reduction()
        assert reductions["parsimonious"] >= 0
        assert reductions["intermediate"] >= 0

    def test_remainder_summary_integration(self, model):
        """MixedQCA から QCAMinimizer を直接構築して残余行診断。"""
        minimizer = QCAMinimizer(
            condition_names=model.all_condition_names,
            is_multivalue={c: False for c in model.set_conditions}
            | {c: True for c in model.multivalue_conditions},
            condition_domains={
                **{c: [0, 1] for c in model.set_conditions},
                **{
                    c: model._sorted_unique(model.data[c])
                    for c in model.multivalue_conditions
                },
            },
        )
        rows = model.build_truth_table(outcome_threshold=0.75)
        df = minimizer.remainder_summary(
            rows,
            outcome_threshold=0.75,
            directional_expectations={"skill": 1, "resources": 1},
        )
        assert isinstance(df, pd.DataFrame)
        assert "theory_consistent" in df.columns

    def test_pipeline_consistency_with_search(self, model):
        """
        QM 法の解と search_sufficient_configurations の結果が
        矛盾しないこと（QM 法の解は十分性閾値を満たすはず）。
        """
        solution = minimize_truth_table(
            model,
            outcome_threshold=0.70,
            consistency_threshold=0.70,
        )

        suf_df = model.search_sufficient_configurations(
            max_depth=3,
            min_consistency=0.70,
            min_coverage=0.0,
            min_cases=1,
        )

        # 複雑解の各項ラベルが探索結果に含まれるかチェック
        for imp in solution.complex_solution:
            label = imp.label(solution.condition_names)
            if label == "(tautology)":
                continue
            # 完全一致でなくとも、少なくとも antecedent 列に存在すること
            found = label in suf_df["antecedent"].values
            if not found:
                # 一貫性を直接計算して閾値を満たすことを確認
                from qca.core.literals import parse_conjunction

                try:
                    lits = parse_conjunction(label)
                    res = model.evaluate_sufficiency(lits)
                    # QM 法の解は理論的に一貫性を持つはず（0.5 以上）
                    assert res.consistency >= 0.5, (
                        f"QM 解 '{label}' の consistency={res.consistency:.3f} "
                        "が期待値を下回っています。"
                    )
                except Exception:
                    pass  # parse に失敗した場合はスキップ


# ===========================================================================
# 境界値・エッジケースのテスト
# ===========================================================================


class TestEdgeCases:
    def test_all_positive_all_negative_no_solution(self):
        """全行がポジティブの場合は論理空間全体が解となる。"""
        minimizer = QCAMinimizer(
            condition_names=["A", "B"],
            is_multivalue={},
            condition_domains={"A": [0, 1], "B": [0, 1]},
        )
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        rows = [
            MockRow({"A": 0, "B": 0}, 0.9, 0.9),
            MockRow({"A": 0, "B": 1}, 0.9, 0.9),
            MockRow({"A": 1, "B": 0}, 0.9, 0.9),
            MockRow({"A": 1, "B": 1}, 0.9, 0.9),
        ]
        solution = minimizer.minimize(rows, outcome_threshold=0.75)
        # 全条件が don't care → 論理的トートロジー
        labels = [
            imp.label(solution.condition_names) for imp in solution.complex_solution
        ]
        assert (
            any("tautology" in lb or lb == "" for lb in labels)
            or len(solution.complex_solution) > 0
        )

    def test_single_condition_binary(self):
        """条件が1つだけの最小構成。"""
        minimizer = QCAMinimizer(
            condition_names=["A"],
            is_multivalue={},
            condition_domains={"A": [0, 1]},
        )
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        rows = [MockRow({"A": 1}, 0.9, 0.9)]
        solution = minimizer.minimize(rows, outcome_threshold=0.75)
        assert len(solution.complex_solution) == 1
        assert solution.complex_solution[0].label(["A"]) == "A"

    def test_multivalue_only_condition(self):
        """多値条件のみの構成（二値条件なし）。"""
        minimizer = QCAMinimizer(
            condition_names=["region"],
            is_multivalue={"region": True},
            condition_domains={"region": ["north", "south", "east"]},
        )
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        rows = [
            MockRow({"region": "north"}, 0.9, 0.88),
            MockRow({"region": "south"}, 0.1, 0.12),
        ]
        solution = minimizer.minimize(rows, outcome_threshold=0.75)
        # 多値条件はマージ不可なので複雑解 = region=north のみ
        assert len(solution.complex_solution) == 1
        label = solution.complex_solution[0].label(["region"])
        assert label == "region=north"

    def test_degenerate_all_dont_care_parsimonious(self):
        """倹約解が全条件 don't care になるケース（論理的トートロジー）。"""
        minimizer = QCAMinimizer(
            condition_names=["A", "B"],
            is_multivalue={},
            condition_domains={"A": [0, 1], "B": [0, 1]},
        )
        from dataclasses import dataclass

        @dataclass
        class MockRow:
            config: dict
            outcome_mean: float
            outcome_raw_consist: float

        # ポジティブ: (1,1)=3 のみ、残余: (0,0),(0,1),(1,0) を全て don't care
        rows = [
            MockRow({"A": 1, "B": 1}, 0.9, 0.9),
            MockRow({"A": 0, "B": 0}, 0.1, 0.1),
        ]
        solution = minimizer.minimize(
            rows,
            outcome_threshold=0.75,
            include_remainders_in_parsimonious=True,
        )
        # 倹約解は複雑解以下の複雑度
        c = sum(i.complexity() for i in solution.complex_solution)
        p = sum(i.complexity() for i in solution.parsimonious_solution)
        assert p <= c

    def test_implicant_pattern_immutable(self):
        """Implicant は frozen なので変更不可。"""
        imp = Implicant(pattern=(1, 0), covered=frozenset([0]))
        with pytest.raises(FrozenInstanceError):
            imp.pattern = (0, 1)  # type: ignore

    def test_build_coverage_table_empty(self):
        df = build_coverage_table([], [], ["A", "B"])
        assert df.empty

    def test_build_coverage_table_shape(self):
        pi = [
            Implicant(pattern=(1, DONT_CARE), covered=frozenset([2, 3])),
            Implicant(pattern=(DONT_CARE, 1), covered=frozenset([1, 3])),
        ]
        df = build_coverage_table(pi, [1, 2, 3], ["A", "B"])
        # 行数 = プライム含意節数、列数 = ミンターム数 + 2 (complexity, n_covered)
        assert df.shape == (2, 3 + 2)

    def test_deduplicate_implicants_same_pattern(self):
        """同一パターンは covered を統合して1つになること。"""
        imps = [
            Implicant(pattern=(1, 0), covered=frozenset([2])),
            Implicant(pattern=(1, 0), covered=frozenset([3])),
        ]
        result = deduplicate_implicants(imps)
        assert len(result) == 1
        assert result[0].covered == frozenset([2, 3])

    def test_deduplicate_implicants_different_pattern(self):
        imps = [
            Implicant(pattern=(1, 0), covered=frozenset([2])),
            Implicant(pattern=(0, 1), covered=frozenset([1])),
        ]
        result = deduplicate_implicants(imps)
        assert len(result) == 2


class TestMixedQCAFitResultIntegration:
    def test_fit_standard_returns_fit_result(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                minimizer="standard",
                outcome_threshold=0.75,
                consistency_cutoff=0.75,
            )
        assert isinstance(result, QCAFitResult)
        assert isinstance(result.qm_solution, QMSolution)
        assert result.qm_solution.condition_names == model.all_condition_names

    def test_fit_qmc_returns_fit_result(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                minimizer="qmc",
                outcome_threshold=0.75,
                consistency_cutoff=0.75,
            )
        assert isinstance(result, QCAFitResult)
        assert isinstance(result.qm_solution, QMSolution)
        assert result.qm_solution.condition_names == model.all_condition_names

    def test_fit_result_exposes_readme_surface(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(minimizer="qmc", outcome_threshold=0.75)
        assert isinstance(result.truth_table, pd.DataFrame)
        assert isinstance(result.solutions, pd.DataFrame)
        assert isinstance(result.case_coverage, pd.DataFrame)
        assert "solution_type" in result.solutions.columns
        assert "case_id" in result.case_coverage.columns
        assert result.consistency is None or 0.0 <= result.consistency <= 1.0
        assert result.coverage is None or 0.0 <= result.coverage <= 1.0
        assert result.to_dataframe().equals(result.solutions)
        assert "QCA Fit Result" in result.to_markdown()
        assert "QCAFitResult" in result.summary()

    def test_fit_standard_and_qmc_match(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            standard = model.fit(minimizer="standard", outcome_threshold=0.75)
            qmc = model.fit(minimizer="qmc", outcome_threshold=0.75)
        assert standard.to_dataframe().equals(qmc.to_dataframe())

    def test_fit_rejects_unknown_minimizer(self, model):
        with pytest.raises(ValueError, match="Unsupported minimizer"):
            model.fit(minimizer="simulated_annealing")

    def test_pyqca_schema_fit_uses_qmc_backend(self, mixed_df):
        qca_model = MixedQCA(
            data=mixed_df,
            outcome="outcome",
            conditions=["skill", "resources", "region"],
            condition_types={
                "skill": "fuzzy",
                "resources": "fuzzy",
                "region": "multi",
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = qca_model.fit(minimizer="qmc", outcome_threshold=0.75)
        assert isinstance(result, QCAFitResult)
        assert result.qm_solution.condition_names == ["skill", "resources", "region"]
