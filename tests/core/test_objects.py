"""Core literal and result object tests."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from qca._types import TruthTableRowProtocol
from qca.core.literals import (
    MultiValueLiteral,
    SetLiteral,
    parse_conjunction,
    parse_literal,
)
from qca.core.results import (
    NecessityResult,
    SufficiencyResult,
    TruthTableRow,
    necessity_results_to_df,
    sufficiency_results_to_df,
    truth_table_to_df,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Regression coverage."""
    return pd.DataFrame(
        {
            "skill": [0.9, 0.8, 0.3, 0.2, 0.6],
            "resources": [0.8, 0.3, 0.8, 0.2, 0.7],
            "region": ["north", "north", "south", "south", "east"],
            "outcome": [0.9, 0.7, 0.8, 0.1, 0.6],
        }
    )


@pytest.fixture
def crossover_df() -> pd.DataFrame:
    """Regression coverage."""
    return pd.DataFrame(
        {
            "skill": [0.5, 0.5, 0.3],
            "outcome": [0.8, 0.4, 0.3],
        }
    )


# ===========================================================================
# ===========================================================================


class TestSetLiteralLabel:
    def test_positive_label(self):
        assert SetLiteral("skill").label() == "skill"

    def test_negated_label(self):
        assert SetLiteral("skill", True).label() == "~skill"

    def test_repr_positive(self):
        assert repr(SetLiteral("skill")) == "SetLiteral('skill')"

    def test_repr_negated(self):
        assert repr(SetLiteral("skill", True)) == "SetLiteral('~skill')"


class TestSetLiteralMembership:
    def test_positive_membership(self, simple_df):
        result = SetLiteral("skill").membership(simple_df)
        pd.testing.assert_series_equal(
            result,
            simple_df["skill"].astype(float),
            check_names=False,
        )

    def test_negated_membership(self, simple_df):
        result = SetLiteral("skill", True).membership(simple_df)
        expected = 1.0 - simple_df["skill"].astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_missing_column_raises_key_error(self, simple_df):
        with pytest.raises(KeyError, match="nonexistent"):
            SetLiteral("nonexistent").membership(simple_df)

    def test_out_of_range_raises_value_error(self):
        df = pd.DataFrame({"skill": [0.5, 1.5, -0.1]})
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            SetLiteral("skill").membership(df)

    def test_crossover_warns(self, crossover_df):
        with pytest.warns(UserWarning, match="crossover point"):
            SetLiteral("skill").membership(crossover_df)

    def test_no_crossover_no_warning(self, simple_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SetLiteral("skill").membership(simple_df)

    def test_returns_series(self, simple_df):
        result = SetLiteral("skill").membership(simple_df)
        assert isinstance(result, pd.Series)

    def test_index_preserved(self, simple_df):
        result = SetLiteral("skill").membership(simple_df)
        assert list(result.index) == list(simple_df.index)


class TestSetLiteralUtility:
    def test_negate(self):
        lit = SetLiteral("skill")
        neg = lit.negate()
        assert neg.name == "skill"
        assert neg.negated is True

    def test_negate_double(self):
        lit = SetLiteral("skill", True)
        assert lit.negate().negated is False

    def test_is_negation_of_true(self):
        a = SetLiteral("skill")
        b = SetLiteral("skill", True)
        assert a.is_negation_of(b)
        assert b.is_negation_of(a)

    def test_is_negation_of_same_direction(self):
        a = SetLiteral("skill")
        b = SetLiteral("skill")
        assert not a.is_negation_of(b)

    def test_is_negation_of_different_name(self):
        a = SetLiteral("skill")
        b = SetLiteral("resources", True)
        assert not a.is_negation_of(b)

    def test_is_negation_of_multivalue(self):
        a = SetLiteral("skill")
        b = MultiValueLiteral("skill", 1)
        assert not a.is_negation_of(b)

    def test_hashable(self):
        """Regression coverage."""
        s = {SetLiteral("skill"), SetLiteral("skill", True), SetLiteral("skill")}
        assert len(s) == 2


# ===========================================================================
# ===========================================================================


class TestMultiValueLiteralLabel:
    def test_string_value_label(self):
        assert MultiValueLiteral("region", "north").label() == "region=north"

    def test_int_value_label(self):
        assert MultiValueLiteral("type", 1).label() == "type=1"

    def test_repr(self):
        assert (
            repr(MultiValueLiteral("region", "north"))
            == "MultiValueLiteral('region', 'north')"
        )


class TestMultiValueLiteralMembership:
    def test_matching_category(self, simple_df):
        result = MultiValueLiteral("region", "north").membership(simple_df)
        expected = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_non_matching_category(self, simple_df):
        result = MultiValueLiteral("region", "south").membership(simple_df)
        expected = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_missing_column_raises_key_error(self, simple_df):
        with pytest.raises(KeyError, match="nonexistent"):
            MultiValueLiteral("nonexistent", "val").membership(simple_df)

    def test_missing_value_warns(self, simple_df):
        with pytest.warns(UserWarning, match="not present"):
            MultiValueLiteral("region", "west").membership(simple_df)

    def test_returns_float_series(self, simple_df):
        result = MultiValueLiteral("region", "north").membership(simple_df)
        assert result.dtype == float

    def test_binary_output(self, simple_df):
        result = MultiValueLiteral("region", "north").membership(simple_df)
        assert set(result.unique()).issubset({0.0, 1.0})


class TestMultiValueLiteralUtility:
    def test_conflicts_with_same_name_different_value(self):
        a = MultiValueLiteral("region", "north")
        b = MultiValueLiteral("region", "south")
        assert a.conflicts_with(b)

    def test_not_conflicts_same_value(self):
        a = MultiValueLiteral("region", "north")
        b = MultiValueLiteral("region", "north")
        assert not a.conflicts_with(b)

    def test_not_conflicts_different_name(self):
        a = MultiValueLiteral("region", "north")
        b = MultiValueLiteral("type", "north")
        assert not a.conflicts_with(b)

    def test_not_conflicts_with_set_literal(self):
        a = MultiValueLiteral("region", "north")
        b = SetLiteral("region")
        assert not a.conflicts_with(b)

    def test_hashable(self):
        s = {
            MultiValueLiteral("region", "north"),
            MultiValueLiteral("region", "south"),
            MultiValueLiteral("region", "north"),
        }
        assert len(s) == 2


# ===========================================================================
# ===========================================================================


class TestParseLiteral:
    def test_set_literal_positive(self):
        lit = parse_literal("skill")
        assert isinstance(lit, SetLiteral)
        assert lit.name == "skill"
        assert lit.negated is False

    def test_set_literal_negated(self):
        lit = parse_literal("~skill")
        assert isinstance(lit, SetLiteral)
        assert lit.name == "skill"
        assert lit.negated is True

    def test_multivalue_string(self):
        lit = parse_literal("region=north")
        assert isinstance(lit, MultiValueLiteral)
        assert lit.name == "region"
        assert lit.value == "north"

    def test_multivalue_int(self):
        lit = parse_literal("type=1")
        assert isinstance(lit, MultiValueLiteral)
        assert lit.value == 1
        assert isinstance(lit.value, int)

    def test_multivalue_float(self):
        lit = parse_literal("score=0.5")
        assert isinstance(lit, MultiValueLiteral)
        assert lit.value == pytest.approx(0.5)
        assert isinstance(lit.value, float)

    def test_whitespace_stripped(self):
        lit = parse_literal("  skill  ")
        assert lit.label() == "skill"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_literal("")

    def test_roundtrip_set_literal(self):
        """Regression coverage."""
        for label in ["skill", "~skill"]:
            assert parse_literal(label).label() == label

    def test_roundtrip_multivalue(self):
        for label in ["region=north", "type=1"]:
            assert parse_literal(label).label() == label


class TestParseConjunction:
    def test_single_literal(self):
        result = parse_conjunction("skill")
        assert len(result) == 1
        assert result[0].label() == "skill"

    def test_multiple_literals(self):
        result = parse_conjunction("skill * ~resources * region=north")
        assert len(result) == 3
        assert result[0].label() == "skill"
        assert result[1].label() == "~resources"
        assert result[2].label() == "region=north"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_conjunction("")


# ===========================================================================
# ===========================================================================


class TestSufficiencyResult:
    @pytest.fixture
    def strong_result(self):
        return SufficiencyResult(
            antecedent="skill * region=north",
            consistency=0.88,
            raw_coverage=0.45,
            unique_coverage=0.30,
            n_cases_in=5,
        )

    @pytest.fixture
    def empty_result(self):
        return SufficiencyResult(
            antecedent="skill",
            consistency=0.0,
            raw_coverage=0.0,
            unique_coverage=None,
            n_cases_in=0,
        )

    def test_is_sufficient_true(self, strong_result):
        assert strong_result.is_sufficient is True

    def test_is_sufficient_false_low_consistency(self):
        r = SufficiencyResult("A", 0.70, 0.40, None, 3)
        assert r.is_sufficient is False

    def test_is_sufficient_false_no_cases(self, empty_result):
        assert empty_result.is_sufficient is False

    def test_strength_strong(self, strong_result):
        assert strong_result.strength == "strong"

    def test_strength_moderate(self):
        r = SufficiencyResult("A", 0.80, 0.30, None, 3)
        assert r.strength == "moderate"

    def test_strength_weak(self):
        r = SufficiencyResult("A", 0.70, 0.30, None, 3)
        assert r.strength == "weak"

    def test_strength_empty(self, empty_result):
        assert empty_result.strength == "empty"

    def test_to_dict_keys(self, strong_result):
        d = strong_result.to_dict()
        assert "antecedent" in d
        assert "consistency" in d
        assert "raw_coverage" in d
        assert "unique_coverage" in d
        assert "n_cases_in" in d
        assert "is_sufficient" in d
        assert "strength" in d

    def test_to_series_type(self, strong_result):
        assert isinstance(strong_result.to_series(), pd.Series)

    def test_str_contains_antecedent(self, strong_result):
        assert "skill * region=north" in str(strong_result)

    def test_unique_coverage_none_in_str(self, empty_result):
        assert "N/A" in str(empty_result)

    def test_frozen(self, strong_result):
        with pytest.raises(FrozenInstanceError):
            strong_result.consistency = 0.99  # type: ignore


# ===========================================================================
# ===========================================================================


class TestNecessityResult:
    @pytest.fixture
    def necessary_result(self):
        return NecessityResult(
            condition="skill",
            consistency=0.92,
            coverage=0.65,
            n_cases_in=7,
        )

    @pytest.fixture
    def trivial_result(self):
        return NecessityResult(
            condition="~nonexistent",
            consistency=0.95,
            coverage=0.95,
            n_cases_in=10,
        )

    def test_is_necessary_true(self, necessary_result):
        assert necessary_result.is_necessary is True

    def test_is_necessary_false(self):
        r = NecessityResult("A", 0.85, 0.60, 5)
        assert r.is_necessary is False

    def test_is_trivial_true(self, trivial_result):
        assert trivial_result.is_trivial is True

    def test_is_trivial_false(self, necessary_result):
        assert necessary_result.is_trivial is False

    def test_strength_strong(self):
        r = NecessityResult("A", 0.96, 0.5, 5)
        assert r.strength == "strong"

    def test_strength_necessary(self, necessary_result):
        assert necessary_result.strength == "necessary"

    def test_strength_weak(self):
        r = NecessityResult("A", 0.80, 0.5, 5)
        assert r.strength == "weak"

    def test_strength_insufficient(self):
        r = NecessityResult("A", 0.60, 0.5, 5)
        assert r.strength == "insufficient"

    def test_to_dict_keys(self, necessary_result):
        d = necessary_result.to_dict()
        assert "condition" in d
        assert "consistency" in d
        assert "coverage" in d
        assert "n_cases_in" in d
        assert "is_necessary" in d
        assert "is_trivial" in d

    def test_str_trivial_note(self, trivial_result):
        assert "Potentially trivial" in str(trivial_result)

    def test_str_no_trivial_note(self, necessary_result):
        assert "Potentially trivial" not in str(necessary_result)


# ===========================================================================
# ===========================================================================


class TestTruthTableRow:
    @pytest.fixture
    def positive_row(self):
        return TruthTableRow(
            config={"skill": 1, "resources": 1, "region": "north"},
            n_cases=3,
            outcome_mean=0.85,
            outcome_raw_consist=0.82,
            case_ids=["A", "B", "C"],
            include=True,
        )

    @pytest.fixture
    def negative_row(self):
        return TruthTableRow(
            config={"skill": 0, "resources": 0, "region": "south"},
            n_cases=2,
            outcome_mean=0.15,
            outcome_raw_consist=0.18,
            case_ids=["D", "E"],
            include=False,
        )

    def test_protocol_compliance(self, positive_row):
        """Regression coverage."""
        assert isinstance(positive_row, TruthTableRowProtocol)

    def test_is_positive(self, positive_row):
        assert positive_row.is_positive is True
        assert positive_row.is_negative is False

    def test_is_negative(self, negative_row):
        assert negative_row.is_negative is True
        assert negative_row.is_positive is False

    def test_is_undecided(self):
        row = TruthTableRow(
            config={"skill": 1},
            n_cases=1,
            outcome_mean=0.6,
            outcome_raw_consist=0.6,
            case_ids=["X"],
            include=None,
        )
        assert row.is_undecided is True

    def test_invalid_outcome_mean_raises(self):
        with pytest.raises(ValueError, match="outcome_mean"):
            TruthTableRow(
                config={},
                n_cases=1,
                outcome_mean=1.5,
                outcome_raw_consist=0.5,
                case_ids=[],
            )

    def test_invalid_outcome_raw_consist_raises(self):
        with pytest.raises(ValueError, match="outcome_raw_consist"):
            TruthTableRow(
                config={},
                n_cases=1,
                outcome_mean=0.5,
                outcome_raw_consist=-0.1,
                case_ids=[],
            )

    def test_invalid_n_cases_raises(self):
        with pytest.raises(ValueError, match="n_cases"):
            TruthTableRow(
                config={},
                n_cases=-1,
                outcome_mean=0.5,
                outcome_raw_consist=0.5,
                case_ids=[],
            )

    def test_to_dict_flat(self, positive_row):
        d = positive_row.to_dict()
        assert "skill" in d
        assert "resources" in d
        assert "region" in d
        assert "n_cases" in d
        assert "outcome_mean" in d
        assert "outcome_raw_consist" in d
        assert "include" in d
        assert "cases" in d

    def test_to_dict_cases_joined(self, positive_row):
        d = positive_row.to_dict()
        assert d["cases"] == "A, B, C"

    def test_str_contains_config(self, positive_row):
        s = str(positive_row)
        assert "skill" in s
        assert "north" in s

    def test_str_include_checkmark(self, positive_row):
        assert "yes" in str(positive_row)

    def test_str_include_cross(self, negative_row):
        assert "no" in str(negative_row)

    def test_many_cases_preview(self):
        """Regression coverage."""
        row = TruthTableRow(
            config={},
            n_cases=6,
            outcome_mean=0.8,
            outcome_raw_consist=0.8,
            case_ids=["A", "B", "C", "D", "E", "F"],
            include=True,
        )
        assert "..." in str(row)


# ===========================================================================
# ===========================================================================


class TestConversionUtilities:
    def test_sufficiency_results_to_df(self):
        results = [
            SufficiencyResult("A", 0.88, 0.45, 0.30, 5),
            SufficiencyResult("B", 0.76, 0.30, None, 3),
        ]
        df = sufficiency_results_to_df(results, depth=2)
        assert len(df) == 2
        assert "depth" in df.columns
        assert "strength" in df.columns

    def test_sufficiency_results_to_df_empty(self):
        df = sufficiency_results_to_df([])
        assert df.empty
        assert "antecedent" in df.columns

    def test_necessity_results_to_df(self):
        results = [
            NecessityResult("skill", 0.92, 0.65, 7),
            NecessityResult("~resources", 0.85, 0.70, 6),
        ]
        df = necessity_results_to_df(results)
        assert len(df) == 2
        assert "is_necessary" in df.columns
        assert "is_trivial" in df.columns

    def test_necessity_results_to_df_empty(self):
        df = necessity_results_to_df([])
        assert df.empty

    def test_truth_table_to_df(self):
        rows = [
            TruthTableRow(
                {"skill": 1, "region": "north"},
                3,
                0.85,
                0.82,
                ["A", "B", "C"],
                True,
            ),
            TruthTableRow(
                {"skill": 0, "region": "south"},
                2,
                0.15,
                0.18,
                ["D", "E"],
                False,
            ),
        ]
        df = truth_table_to_df(rows)
        assert len(df) == 2
        assert "skill" in df.columns
        assert "region" in df.columns
        assert "include" in df.columns

    def test_truth_table_to_df_empty(self):
        df = truth_table_to_df([])
        assert df.empty
