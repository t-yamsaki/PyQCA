"""フェーズ3: core/qca.py の動作確認テスト"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qca.core.literals import MultiValueLiteral, SetLiteral
from qca.core.results import NecessityResult, SufficiencyResult, TruthTableRow
from qca.engines.mixed import MixedQCA

# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def base_df() -> pd.DataFrame:
    """Ragin (2008) の例題に近い構造のテスト用 DataFrame。"""
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
def model(base_df) -> MixedQCA:
    return MixedQCA(
        data=base_df,
        case_id="case",
        set_conditions=["skill", "resources"],
        multivalue_conditions=["region"],
        outcome="outcome",
    )


@pytest.fixture
def crisp_df() -> pd.DataFrame:
    """crisp set のみのテスト用 DataFrame。"""
    return pd.DataFrame(
        {
            "case": ["A", "B", "C", "D"],
            "A": [1.0, 1.0, 0.0, 0.0],
            "B": [1.0, 0.0, 1.0, 0.0],
            "outcome": [1.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def crisp_model(crisp_df) -> MixedQCA:
    return MixedQCA(
        data=crisp_df,
        case_id="case",
        set_conditions=["A", "B"],
        multivalue_conditions=[],
        outcome="outcome",
    )


# ===========================================================================
# __init__ / プロパティのテスト
# ===========================================================================


class TestMixedQCAInit:
    def test_n_cases(self, model):
        assert model.n_cases == 8

    def test_n_conditions(self, model):
        assert model.n_conditions == 3

    def test_all_condition_names(self, model):
        assert model.all_condition_names == ["skill", "resources", "region"]

    def test_outcome_series_float(self, model):
        assert model.outcome_series.dtype == float

    def test_repr(self, model):
        r = repr(model)
        assert "MixedQCA" in r
        assert "n_cases=8" in r

    def test_index_normalized(self, base_df):
        """reset_index(drop=True) によりインデックスが 0 始まりになること。"""
        df = base_df.copy()
        df.index = range(10, 18)  # 意図的にずらす
        m = MixedQCA(df, "case", ["skill", "resources"], ["region"], "outcome")
        assert list(m.data.index) == list(range(8))

    def test_data_is_copy(self, base_df):
        """元の DataFrame を変更しても model.data に影響しないこと。"""
        m = MixedQCA(base_df, "case", ["skill", "resources"], ["region"], "outcome")
        base_df.loc[0, "skill"] = 0.0
        assert m.data.loc[0, "skill"] == 0.9

    def test_pyqca_schema_maps_to_internal_conditions(self, base_df):
        """conditions / condition_types 形式を内部の set/multi 条件へ変換すること。"""
        m = MixedQCA(
            data=base_df,
            case_id="case",
            outcome="outcome",
            conditions=["skill", "resources", "region"],
            condition_types={
                "skill": "fuzzy",
                "resources": "crisp",
                "region": "multi",
            },
        )
        assert m.set_conditions == ["skill", "resources"]
        assert m.multivalue_conditions == ["region"]
        assert m.conditions == ["skill", "resources", "region"]
        assert m.condition_types == {
            "skill": "fuzzy",
            "resources": "crisp",
            "region": "multi",
        }

    def test_pyqca_schema_can_generate_case_id(self, base_df):
        """case_id 未指定時は内部ケース ID を生成すること。"""
        df = base_df.drop(columns=["case"])
        m = MixedQCA(
            data=df,
            outcome="outcome",
            conditions=["skill", "resources", "region"],
            condition_types={
                "skill": "fuzzy",
                "resources": "fuzzy",
                "region": "multi",
            },
        )
        assert m.generated_case_id is True
        assert m.case_id in m.data.columns
        assert m.data[m.case_id].is_unique
        assert sum(row.n_cases for row in m.build_truth_table()) == m.n_cases

    def test_legacy_schema_infers_condition_types(self, crisp_model):
        """legacy 形式でも condition_types を利用可能にすること。"""
        assert crisp_model.condition_types == {"A": "crisp", "B": "crisp"}
        assert crisp_model.conditions == ["A", "B"]


class TestMixedQCAValidation:
    def test_missing_column_raises(self, base_df):
        with pytest.raises(ValueError, match="必須カラム"):
            MixedQCA(base_df, "case", ["nonexistent"], [], "outcome")

    def test_set_condition_out_of_range_raises(self, base_df):
        base_df = base_df.copy()
        base_df["skill"] = base_df["skill"] * 2  # 1.0 を超える値
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MixedQCA(base_df, "case", ["skill", "resources"], ["region"], "outcome")

    def test_outcome_out_of_range_raises(self, base_df):
        base_df = base_df.copy()
        base_df["outcome"] = base_df["outcome"] - 0.5  # 負の値が生じる
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MixedQCA(base_df, "case", ["skill", "resources"], ["region"], "outcome")

    def test_duplicate_case_id_raises(self, base_df):
        base_df = base_df.copy()
        base_df.loc[0, "case"] = "B"  # B が重複
        with pytest.raises(ValueError, match="重複"):
            MixedQCA(base_df, "case", ["skill", "resources"], ["region"], "outcome")

    def test_condition_overlap_raises(self, base_df):
        with pytest.raises(ValueError, match="重複カラム"):
            MixedQCA(
                base_df,
                "case",
                ["skill", "resources"],
                ["skill"],  # set_conditions と重複
                "outcome",
            )

    def test_non_numeric_set_condition_raises(self, base_df):
        base_df = base_df.copy()
        base_df["skill"] = "high"
        with pytest.raises(TypeError, match="数値に変換"):
            MixedQCA(base_df, "case", ["skill", "resources"], ["region"], "outcome")

    def test_pyqca_schema_requires_condition_types(self, base_df):
        with pytest.raises(ValueError, match="conditions と condition_types"):
            MixedQCA(
                data=base_df,
                case_id="case",
                outcome="outcome",
                conditions=["skill", "resources"],
            )

    def test_pyqca_schema_rejects_unknown_condition_type(self, base_df):
        with pytest.raises(ValueError, match="未知の型"):
            MixedQCA(
                data=base_df,
                case_id="case",
                outcome="outcome",
                conditions=["skill"],
                condition_types={"skill": "ordinal"},
            )

    def test_pyqca_schema_rejects_legacy_mix(self, base_df):
        with pytest.raises(ValueError, match="同時指定"):
            MixedQCA(
                data=base_df,
                case_id="case",
                outcome="outcome",
                conditions=["skill", "region"],
                condition_types={"skill": "fuzzy", "region": "multi"},
                set_conditions=["skill"],
            )

    def test_pyqca_schema_rejects_extra_condition_type_key(self, base_df):
        with pytest.raises(ValueError, match="含まれない"):
            MixedQCA(
                data=base_df,
                case_id="case",
                outcome="outcome",
                conditions=["skill"],
                condition_types={"skill": "fuzzy", "region": "multi"},
            )


# ===========================================================================
# メンバーシップ計算のテスト
# ===========================================================================


class TestConjunctionMembership:
    def test_single_literal(self, model):
        result = model.conjunction_membership([SetLiteral("skill")])
        expected = model.data["skill"].astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_two_literals_min(self, model):
        """AND = min(skill, resources) であること。"""
        result = model.conjunction_membership(
            [SetLiteral("skill"), SetLiteral("resources")]
        )
        expected = model.data[["skill", "resources"]].min(axis=1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_multivalue_literal(self, model):
        result = model.conjunction_membership([MultiValueLiteral("region", "north")])
        expected = (model.data["region"] == "north").astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_mixed_literals(self, model):
        result = model.conjunction_membership(
            [SetLiteral("skill"), MultiValueLiteral("region", "north")]
        )
        mask = (model.data["region"] == "north").astype(float)
        expected = pd.concat([model.data["skill"].astype(float), mask], axis=1).min(
            axis=1
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_empty_raises(self, model):
        with pytest.raises(ValueError, match="空にできません"):
            model.conjunction_membership([])

    def test_contradictory_raises(self, model):
        with pytest.raises(ValueError, match="矛盾"):
            model.conjunction_membership(
                [SetLiteral("skill"), SetLiteral("skill", True)]
            )

    def test_index_matches_data(self, model):
        result = model.conjunction_membership([SetLiteral("skill")])
        assert list(result.index) == list(model.data.index)


class TestDisjunctionMembership:
    def test_two_terms_max(self, model):
        """OR = max(term1, term2) であること。"""
        t1 = [SetLiteral("skill")]
        t2 = [SetLiteral("resources")]
        result = model.disjunction_membership([t1, t2])
        expected = model.data[["skill", "resources"]].max(axis=1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_single_term(self, model):
        """1項のみの OR は conjunction と等価。"""
        t = [SetLiteral("skill")]
        result = model.disjunction_membership([t])
        expected = model.conjunction_membership(t)
        pd.testing.assert_series_equal(result, expected)

    def test_empty_raises(self, model):
        with pytest.raises(ValueError, match="空にできません"):
            model.disjunction_membership([])


# ===========================================================================
# 十分性評価のテスト
# ===========================================================================


class TestEvaluateSufficiency:
    def test_returns_sufficiency_result(self, model):
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert isinstance(result, SufficiencyResult)

    def test_consistency_range(self, model):
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert 0.0 <= result.consistency <= 1.0

    def test_raw_coverage_range(self, model):
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert 0.0 <= result.raw_coverage <= 1.0

    def test_n_cases_in_nonnegative(self, model):
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert result.n_cases_in >= 0

    def test_unique_coverage_none_without_solution(self, model):
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert result.unique_coverage is None

    def test_crisp_perfect_sufficiency(self, crisp_model):
        """A=1 かつ B=1 の唯一のケース（A）がアウトカム=1 → consistency=1.0。"""
        result = crisp_model.evaluate_sufficiency([SetLiteral("A"), SetLiteral("B")])
        assert result.consistency == pytest.approx(1.0)
        assert result.n_cases_in == 1

    def test_empty_set_returns_zero(self, model):
        """メンバーシップが全てゼロの条件は consistency=0, n_cases_in=0。"""
        result = model.evaluate_sufficiency([MultiValueLiteral("region", "west")])
        assert result.consistency == 0.0
        assert result.n_cases_in == 0

    def test_antecedent_label_conjunction(self, model):
        result = model.evaluate_sufficiency(
            [SetLiteral("skill"), SetLiteral("resources")]
        )
        assert result.antecedent == "skill * resources"

    def test_formula_consistency(self, model):
        """consistency = Σ min(Xi, Yi) / Σ Xi の手計算値と一致すること。"""
        x = model.conjunction_membership([SetLiteral("skill")]).to_numpy()
        y = model.outcome_series.to_numpy()
        expected = float(np.minimum(x, y).sum()) / float(x.sum())
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert result.consistency == pytest.approx(expected)

    def test_formula_raw_coverage(self, model):
        """raw_coverage = Σ min(Xi, Yi) / Σ Yi の手計算値と一致すること。"""
        x = model.conjunction_membership([SetLiteral("skill")]).to_numpy()
        y = model.outcome_series.to_numpy()
        expected = float(np.minimum(x, y).sum()) / float(y.sum())
        result = model.evaluate_sufficiency([SetLiteral("skill")])
        assert result.raw_coverage == pytest.approx(expected)


class TestEvaluateSolution:
    def test_returns_sufficiency_result(self, model):
        terms = [
            [SetLiteral("skill")],
            [SetLiteral("resources")],
        ]
        result = model.evaluate_solution(terms)
        assert isinstance(result, SufficiencyResult)

    def test_unique_coverage_is_none(self, model):
        """解全体の unique_coverage は None であること。"""
        terms = [[SetLiteral("skill")], [SetLiteral("resources")]]
        result = model.evaluate_solution(terms)
        assert result.unique_coverage is None

    def test_antecedent_format(self, model):
        terms = [
            [SetLiteral("skill"), MultiValueLiteral("region", "north")],
            [SetLiteral("resources")],
        ]
        result = model.evaluate_solution(terms)
        assert "(skill * region=north)" in result.antecedent
        assert "(resources)" in result.antecedent
        assert "+" in result.antecedent

    def test_consistency_geq_single_term(self, model):
        """解全体の consistency は各項の consistency 以下であること。"""
        t1 = [SetLiteral("skill")]
        t2 = [SetLiteral("resources")]
        sol = model.evaluate_solution([t1, t2])
        r1 = model.evaluate_sufficiency(t1)
        r2 = model.evaluate_sufficiency(t2)
        # OR の consistency は各項の最大以下
        assert sol.consistency <= max(r1.consistency, r2.consistency) + 1e-9


# ===========================================================================
# 必要性評価のテスト
# ===========================================================================


class TestEvaluateNecessity:
    def test_returns_necessity_result(self, model):
        result = model.evaluate_necessity(SetLiteral("skill"))
        assert isinstance(result, NecessityResult)

    def test_consistency_range(self, model):
        result = model.evaluate_necessity(SetLiteral("skill"))
        assert 0.0 <= result.consistency <= 1.0

    def test_coverage_range(self, model):
        result = model.evaluate_necessity(SetLiteral("skill"))
        assert 0.0 <= result.coverage <= 1.0

    def test_formula_consistency(self, model):
        """consistency = Σ min(Xi, Yi) / Σ Yi の手計算値と一致すること。"""
        x = SetLiteral("skill").membership(model.data).to_numpy()
        y = model.outcome_series.to_numpy()
        expected = float(np.minimum(x, y).sum()) / float(y.sum())
        result = model.evaluate_necessity(SetLiteral("skill"))
        assert result.consistency == pytest.approx(expected)

    def test_formula_coverage(self, model):
        """coverage = Σ min(Xi, Yi) / Σ Xi の手計算値と一致すること。"""
        x = SetLiteral("skill").membership(model.data).to_numpy()
        y = model.outcome_series.to_numpy()
        expected = float(np.minimum(x, y).sum()) / float(x.sum())
        result = model.evaluate_necessity(SetLiteral("skill"))
        assert result.coverage == pytest.approx(expected)

    def test_multivalue_literal(self, model):
        result = model.evaluate_necessity(MultiValueLiteral("region", "north"))
        assert isinstance(result, NecessityResult)
        assert result.n_cases_in == 3


# ===========================================================================
# unique coverage のテスト
# ===========================================================================


class TestComputeSolutionCoverages:
    def test_returns_list_of_sufficiency_results(self, model):
        terms = [[SetLiteral("skill")], [SetLiteral("resources")]]
        results = model.compute_solution_coverages(terms)
        assert len(results) == 2
        assert all(isinstance(r, SufficiencyResult) for r in results)

    def test_unique_coverage_not_none(self, model):
        terms = [[SetLiteral("skill")], [SetLiteral("resources")]]
        results = model.compute_solution_coverages(terms)
        for r in results:
            assert r.unique_coverage is not None

    def test_single_term_unique_equals_raw(self, model):
        """1項のみの解では unique_coverage = raw_coverage であること。"""
        terms = [[SetLiteral("skill")]]
        results = model.compute_solution_coverages(terms)
        assert results[0].unique_coverage == pytest.approx(
            results[0].raw_coverage, abs=1e-9
        )

    def test_unique_coverage_nonnegative(self, model):
        terms = [
            [SetLiteral("skill")],
            [SetLiteral("resources")],
            [MultiValueLiteral("region", "north")],
        ]
        results = model.compute_solution_coverages(terms)
        for r in results:
            assert r.unique_coverage >= 0.0

    def test_unique_coverage_leq_raw_coverage(self, model):
        """unique_coverage <= raw_coverage であること。"""
        terms = [[SetLiteral("skill")], [SetLiteral("resources")]]
        results = model.compute_solution_coverages(terms)
        for r in results:
            assert r.unique_coverage <= r.raw_coverage + 1e-9


# ===========================================================================
# 網羅探索のテスト
# ===========================================================================


class TestSearchSufficientConfigurations:
    def test_returns_dataframe(self, model):
        df = model.search_sufficient_configurations(max_depth=1)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, model):
        df = model.search_sufficient_configurations(max_depth=1)
        for col in ["antecedent", "consistency", "raw_coverage", "n_cases_in", "depth"]:
            assert col in df.columns

    def test_consistency_above_threshold(self, model):
        threshold = 0.75
        df = model.search_sufficient_configurations(
            max_depth=2, min_consistency=threshold
        )
        if not df.empty:
            assert (df["consistency"] >= threshold).all()

    def test_depth_column_values(self, model):
        df = model.search_sufficient_configurations(max_depth=2)
        if not df.empty:
            assert df["depth"].between(1, 2).all()

    def test_sorted_by_consistency_desc(self, model):
        df = model.search_sufficient_configurations(max_depth=2)
        if len(df) > 1:
            assert (df["consistency"].diff().dropna() <= 0).all()

    def test_no_contradictory_conjunctions(self, model):
        """結果に矛盾した conjunction が含まれないこと。"""
        df = model.search_sufficient_configurations(max_depth=2)
        for ant in df["antecedent"]:
            parts = ant.split(" * ")
            set_names = [p.lstrip("~") for p in parts if "=" not in p]
            # 同名の正・否定が共存しないこと
            assert len(set_names) == len(set(set_names)), f"矛盾した conjunction: {ant}"

    def test_min_cases_filter(self, model):
        df = model.search_sufficient_configurations(max_depth=1, min_cases=3)
        if not df.empty:
            assert (df["n_cases_in"] >= 3).all()

    def test_empty_result_on_strict_threshold(self, model):
        """非常に厳しい閾値では空の DataFrame が返ること。"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = model.search_sufficient_configurations(
                max_depth=1, min_consistency=0.999
            )
        assert df.empty

    def test_empty_df_has_correct_columns(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = model.search_sufficient_configurations(
                max_depth=1, min_consistency=0.999
            )
        for col in ["antecedent", "consistency", "raw_coverage", "depth"]:
            assert col in df.columns


class TestSearchNecessaryConditions:
    def test_returns_dataframe(self, model):
        df = model.search_necessary_conditions(min_consistency=0.5)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, model):
        df = model.search_necessary_conditions(min_consistency=0.5)
        if not df.empty:
            for col in ["condition", "consistency", "coverage", "n_cases_in"]:
                assert col in df.columns

    def test_consistency_above_threshold(self, model):
        threshold = 0.7
        df = model.search_necessary_conditions(min_consistency=threshold)
        if not df.empty:
            assert (df["consistency"] >= threshold).all()

    def test_sorted_by_consistency_desc(self, model):
        df = model.search_necessary_conditions(min_consistency=0.5)
        if len(df) > 1:
            assert (df["consistency"].diff().dropna() <= 0).all()


# ===========================================================================
# 真理表構築のテスト
# ===========================================================================


class TestBuildTruthTable:
    def test_returns_list_of_truth_table_rows(self, model):
        rows = model.build_truth_table()
        assert isinstance(rows, list)
        assert all(isinstance(r, TruthTableRow) for r in rows)

    def test_all_cases_covered(self, model):
        """全ケースがいずれかの行に属すること。"""
        rows = model.build_truth_table()
        total_cases = sum(r.n_cases for r in rows)
        assert total_cases == model.n_cases

    def test_include_flag_set(self, model):
        """include が None でないこと。"""
        rows = model.build_truth_table(outcome_threshold=0.5)
        assert all(r.include is not None for r in rows)

    def test_include_threshold(self, model):
        """include=True の行は outcome_mean >= threshold であること。"""
        threshold = 0.75
        rows = model.build_truth_table(outcome_threshold=threshold)
        for r in rows:
            if r.include:
                assert r.outcome_mean >= threshold
            else:
                assert r.outcome_mean < threshold

    def test_outcome_mean_range(self, model):
        rows = model.build_truth_table()
        for r in rows:
            assert 0.0 <= r.outcome_mean <= 1.0

    def test_row_consistency_range(self, model):
        rows = model.build_truth_table()
        for r in rows:
            assert 0.0 <= r.outcome_raw_consist <= 1.0

    def test_config_keys_match_conditions(self, model):
        """config のキーが全条件名と一致すること。"""
        rows = model.build_truth_table()
        expected_keys = set(model.all_condition_names)
        for r in rows:
            assert set(r.config.keys()) == expected_keys

    def test_sorted_by_outcome_mean_desc(self, model):
        """outcome_mean の降順でソートされていること。"""
        rows = model.build_truth_table()
        means = [r.outcome_mean for r in rows]
        assert means == sorted(means, reverse=True)

    def test_multivalue_config_values(self, model):
        """multi-value 条件の config 値が実際のカテゴリ値であること。"""
        rows = model.build_truth_table()
        valid_regions = {"north", "south", "east"}
        for r in rows:
            assert r.config["region"] in valid_regions

    def test_set_condition_config_values_binary(self, model):
        """set_conditions の config 値が 0 または 1 であること。"""
        rows = model.build_truth_table()
        for r in rows:
            assert r.config["skill"] in {0, 1}
            assert r.config["resources"] in {0, 1}

    def test_truth_table_to_df_returns_dataframe(self, model):
        df = model.truth_table_to_df()
        assert isinstance(df, pd.DataFrame)
        assert "skill" in df.columns
        assert "region" in df.columns
        assert "include" in df.columns

    def test_row_consistency_formula(self, model):
        """行一貫性スコア = Σ min(Xi, Yi) / Σ Xi を手計算で検証。"""
        rows = model.build_truth_table()
        for row in rows:
            # config → literals を取得して手計算
            lits = model._config_to_literals(row.config)
            # 該当ケースを特定
            mask = pd.Series(True, index=model.data.index)
            for lit in lits:
                lit.membership(model.data)
                if isinstance(lit, SetLiteral):
                    if lit.negated:
                        mask = mask & (model.data[lit.name] < 0.5)
                    else:
                        mask = mask & (model.data[lit.name] >= 0.5)
                else:
                    mask = mask & (model.data[lit.name] == lit.value)

            if mask.sum() == 0:
                continue

            group_idx = model.data.index[mask]
            x = model.conjunction_membership(lits).loc[group_idx].to_numpy()
            y = model.data.loc[group_idx, model.outcome].to_numpy()
            x_sum = x.sum()
            if x_sum > 0:
                expected = float(np.minimum(x, y).sum()) / x_sum
                assert row.outcome_raw_consist == pytest.approx(expected, abs=1e-9)


# ===========================================================================
# 内部ユーティリティのテスト
# ===========================================================================


class TestInternalUtilities:
    def test_sorted_unique_numeric(self, model):
        result = MixedQCA._sorted_unique(pd.Series([3, 1, 2]))
        assert result == [1, 2, 3]

    def test_sorted_unique_string(self, model):
        result = MixedQCA._sorted_unique(pd.Series(["north", "east", "south"]))
        assert result == ["east", "north", "south"]

    def test_sorted_unique_numeric_string(self):
        result = MixedQCA._sorted_unique(pd.Series([10, 2, 1]))
        assert result == [1, 2, 10]

    def test_is_contradictory_set_literal(self):
        lits = [SetLiteral("skill"), SetLiteral("skill", True)]
        assert MixedQCA._is_contradictory(lits) is True

    def test_is_contradictory_multivalue(self):
        lits = [
            MultiValueLiteral("region", "north"),
            MultiValueLiteral("region", "south"),
        ]
        assert MixedQCA._is_contradictory(lits) is True

    def test_not_contradictory(self):
        lits = [SetLiteral("skill"), MultiValueLiteral("region", "north")]
        assert MixedQCA._is_contradictory(lits) is False

    def test_config_to_literals_set(self, model):
        config = {"skill": 1, "resources": 0, "region": "north"}
        lits = model._config_to_literals(config)
        labels = {lit.label() for lit in lits}
        assert "skill" in labels
        assert "~resources" in labels
        assert "region=north" in labels

    def test_n_cases_in(self, model):
        n = model._n_cases_in(SetLiteral("skill"))
        # skill > 0.5: A(0.9), B(0.8), E(0.9), F(0.8) の4件
        assert n == 4
