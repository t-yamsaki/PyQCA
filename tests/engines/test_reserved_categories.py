"""Reserved wildcard category regressions (#27)."""

import pandas as pd
import pytest

from qca import GSQCA, MVQCA, ConditionSpec, MultiValueLiteral
from qca.core.literals import parse_literal
from qca.minimizers.algorithms import quine_mccluskey
from qca.minimizers.remainder import generate_all_minterms


@pytest.mark.parametrize("engine", [MVQCA, GSQCA])
def test_observed_wildcard_is_rejected_before_fit(engine):
    kwargs = {"condition_types": {"A": "multi"}} if engine is GSQCA else {}
    with pytest.raises(ValueError, match="reserved"):
        engine(
            pd.DataFrame({"A": ["*", "other"], "Y": [1, 0]}),
            outcome="Y", conditions=["A"], **kwargs,
        )


@pytest.mark.parametrize("kwargs", [
    {"domain": ("*", "other")},
    {"domain": ("other",), "value_columns": {"*": "membership"}},
    {"value_columns": {"*": "membership"}},
    {"domain": (frozenset({"*", "other"}),)},
])
def test_schema_rejects_reserved_categories(kwargs):
    with pytest.raises(ValueError, match="reserved"):
        ConditionSpec("A", "multi", **kwargs)


@pytest.mark.parametrize("value", ["*", ("*", "other"), frozenset({"*", "other"})])
def test_literal_rejects_reserved_category_and_unions(value):
    with pytest.raises(ValueError, match="reserved"):
        MultiValueLiteral("A", value)


@pytest.mark.parametrize("text", ["A=*", "A={*,other}"])
def test_parser_rejects_reserved_categories(text):
    with pytest.raises(ValueError, match="reserved"):
        parse_literal(text)


def test_low_level_concrete_domains_reject_wildcard():
    with pytest.raises(ValueError, match="reserved"):
        generate_all_minterms(["A"], {"A": ["*", "other"]})
    with pytest.raises(ValueError, match="reserved"):
        quine_mccluskey([0], [("*",), ("other",)], [True])


@pytest.mark.parametrize("engine", [MVQCA, GSQCA])
def test_recoded_category_and_embedded_asterisk_keep_membership(engine):
    kwargs = {"condition_types": {"A": "multi"}} if engine is GSQCA else {}
    model = engine(
        pd.DataFrame({"A": ["A*", "other"], "Y": [1, 0]}),
        outcome="Y", conditions=["A"], **kwargs,
    )
    literal = MultiValueLiteral("A", "A*")
    assert model.conjunction_membership([literal]).tolist() == [1, 0]
    assert model.evaluate_sufficiency([literal]).consistency == pytest.approx(1)
    result = model.fit()
    assert result.qm_solution.complex_solution
    assert all(
        implicant.pattern != ("*",)
        for implicant in result.qm_solution.complex_solution
    )


def test_internal_wildcard_still_represents_a_full_domain():
    primes = quine_mccluskey([0, 1], [("A*",), ("other",)], [True])
    assert len(primes) == 1
    assert primes[0].pattern == ("*",)
