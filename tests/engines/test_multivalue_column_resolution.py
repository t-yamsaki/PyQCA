"""Regression tests for schema-aware multi-value column resolution (#17)."""

import pandas as pd
import pytest

from qca import GSQCA, MVQCA, ConditionSpec, MultiValueLiteral


@pytest.mark.parametrize("engine", [MVQCA, GSQCA])
@pytest.mark.parametrize("template", ["A={}", "A[{}]", "A__{}"])
@pytest.mark.parametrize("auxiliary", ["reversed", "invalid", "partial"])
def test_categorical_results_ignore_auxiliary_columns(engine, template, auxiliary):
    data = pd.DataFrame({"A": [0, 1, 2], "Y": [0, 1, 1]})
    kwargs = {"outcome": "Y", "conditions": ["A"]}
    if engine is GSQCA:
        kwargs["condition_types"] = {"A": "multi"}
    baseline = engine(data, **kwargs)
    shadow = data.copy()
    values = [0] if auxiliary == "partial" else [0, 1, 2]
    for value in values:
        shadow[template.format(value)] = (
            ["unused"] * 3 if auxiliary == "invalid" else [1, 0, 0]
        )
    model = engine(shadow, **kwargs)
    assert model.build_truth_table() == baseline.build_truth_table()
    for value in [0, 1, 2, frozenset({1, 2})]:
        literal = MultiValueLiteral("A", value)
        pd.testing.assert_series_equal(
            model.conjunction_membership([literal]),
            baseline.conjunction_membership([literal]),
        )
        assert model.evaluate_necessity(literal) == baseline.evaluate_necessity(literal)
        assert model.evaluate_sufficiency([literal]) == baseline.evaluate_sufficiency(
            [literal]
        )
    expected = baseline.fit()
    actual = model.fit()
    assert actual.qm_solution.complex_solution == expected.qm_solution.complex_solution
    assert actual.solution_metrics == expected.solution_metrics
    pd.testing.assert_frame_equal(actual.case_coverage, expected.case_coverage)
    pd.testing.assert_frame_equal(
        model.search_sufficient_configurations(),
        baseline.search_sufficient_configurations(),
    )


@pytest.mark.parametrize("mode", ["explicit", "calibrated", "columns_only"])
@pytest.mark.parametrize("template", ["A={}", "A[{}]", "A__{}"])
def test_gsqca_preserves_value_memberships(mode, template):
    data = pd.DataFrame({"Y": [0.8, 0.2]})
    columns = {0: template.format(0), 1: template.format(1)}
    data[columns[0]] = [0.1, 0.9]
    data[columns[1]] = [0.8, 0.2]
    if mode != "columns_only":
        data["A"] = [0, 1]  # Opposite to the calibrated corner assignment.
    spec = ConditionSpec(
        "A", "multi", domain=(0, 1),
        calibrated=True if mode == "calibrated" else None,
        value_columns=columns if mode == "explicit" else None,
    )
    model = GSQCA(data, outcome="Y", condition_specs=[spec])
    literal = MultiValueLiteral("A", 1)
    assert model.conjunction_membership([literal]).tolist() == [0.8, 0.2]
    assert model.evaluate_necessity(literal).consistency == pytest.approx(1)
    assert model.evaluate_sufficiency([literal]).consistency == pytest.approx(1)
    union = MultiValueLiteral("A", frozenset({0, 1}))
    assert model.conjunction_membership([union]).tolist() == [0.8, 0.9]
    rows = {row.config["A"]: row for row in model.build_truth_table()}
    assert rows[1].case_ids == ["case_1"]
    assert rows[0].case_ids == ["case_2"]


def test_standalone_literal_retains_value_column_precedence():
    data = pd.DataFrame({"A": [0, 1], "A=1": [0.8, 0.2]})
    assert MultiValueLiteral("A", 1).membership(data).tolist() == [0.8, 0.2]


def test_explicit_mapping_overrides_competing_columns():
    data = pd.DataFrame({
        "A": [0, 1], "Y": [0.8, 0.2],
        "low": [0.1, 0.9], "high": [0.8, 0.2],
        "A=0": [1, 0], "A=1": [0, 1],
    })
    original = data.copy(deep=True)
    model = GSQCA(data, outcome="Y", condition_specs=[
        ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
    ])
    assert model.conjunction_membership([MultiValueLiteral("A", 1)]).tolist() == [
        0.8, 0.2
    ]
    pd.testing.assert_frame_equal(data, original)


def test_calibrated_condition_does_not_fall_back_to_category():
    model = GSQCA(
        pd.DataFrame({"A": [0, 1], "Y": [0, 1], "A=0": [0.8, 0.2]}),
        outcome="Y",
        condition_specs=[ConditionSpec("A", "multi", domain=(0, 1), calibrated=True)],
    )
    with pytest.raises(KeyError, match="MultiValueLiteral"):
        model.evaluate_necessity(MultiValueLiteral("A", 1))


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_alias_cycle_reads_original_columns(reverse):
    data = pd.DataFrame({"A=0": [0.1, 0.9], "A=1": [0.8, 0.2], "Y": [0.8, 0.2]})
    original = data.copy(deep=True)
    items = [(0, "A=1"), (1, "A=0")]
    mapping = dict(reversed(items) if reverse else items)
    model = GSQCA(data, outcome="Y", condition_specs=[
        ConditionSpec("A", "multi", value_columns=mapping)
    ])
    assert model.conjunction_membership(
        [MultiValueLiteral("A", 0)]
    ).tolist() == [0.8, 0.2]
    assert model.conjunction_membership(
        [MultiValueLiteral("A", 1)]
    ).tolist() == [0.1, 0.9]
    assert model.outcome_series.tolist() == [0.8, 0.2]
    assert model.evaluate_sufficiency([MultiValueLiteral("A", 0)]).consistency == 1
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("reverse", [False, True])
def test_cross_condition_sources_read_original_columns(reverse):
    data = pd.DataFrame({
        "A=0": [0.1, 0.9], "A=1": [0.8, 0.2],
        "B=0": [0.3, 0.7], "B=1": [0.6, 0.4], "Y": [1, 0],
    })
    original = data.copy(deep=True)
    specs = [
        ConditionSpec("A", "multi", value_columns={0: "B=0", 1: "B=1"}),
        ConditionSpec("B", "multi", value_columns={0: "A=0", 1: "A=1"}),
    ]
    model = GSQCA(data, outcome="Y", condition_specs=specs[::-1] if reverse else specs)
    for condition, source in [("A", "B"), ("B", "A")]:
        for value in [0, 1]:
            assert model.conjunction_membership(
                [MultiValueLiteral(condition, value)]
            ).tolist() == original[f"{source}={value}"].tolist()
    assert model.outcome_series.tolist() == [1, 0]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("role", ["outcome", "case_id", "crisp", "multi"])
def test_alias_targets_cannot_overwrite_analysis_columns(role):
    data = pd.DataFrame({
        "low": [0.9, 0.1], "high": [0.1, 0.9], "A=0": [0, 1], "Y": [1, 0],
    })
    original = data.copy(deep=True)
    specs = [ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})]
    kwargs = {"outcome": "Y"}
    if role in {"outcome", "case_id"}:
        kwargs[role] = "A=0"
    else:
        specs.append(ConditionSpec("A=0", role))
    with pytest.raises(
        ValueError, match="conflicts with.*outcome, case ID, or condition"
    ):
        GSQCA(data, condition_specs=specs, **kwargs)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_ambiguous_value_column_targets_are_rejected(reverse, explicit):
    data = pd.DataFrame({
        "low": [0.9, 0.1], "high": [0.1, 0.9],
        "A=x=0": [0.2, 0.8], "A=x=1": [0.7, 0.3], "Y": [1, 0],
    })
    specs = [
        ConditionSpec("A", "multi", value_columns={"x=0": "low", "x=1": "high"}),
        ConditionSpec("A=x", "multi", domain=(0, 1),
                      value_columns={0: "high", 1: "low"} if explicit else None),
    ]
    with pytest.raises(ValueError, match="another value-membership column"):
        GSQCA(data, outcome="Y", condition_specs=specs[::-1] if reverse else specs)


def test_shared_source_and_identity_aliases_are_allowed():
    data = pd.DataFrame({"A=0": [0.1, 0.9], "high": [0.8, 0.2], "Y": [1, 0]})
    model = GSQCA(data, outcome="Y", condition_specs=[
        ConditionSpec("A", "multi", value_columns={0: "A=0", 1: "high"}),
        ConditionSpec("B", "multi", value_columns={0: "A=0", 1: "high"}),
    ])
    for condition in ["A", "B"]:
        assert model.conjunction_membership(
            [MultiValueLiteral(condition, 0)]
        ).tolist() == [0.1, 0.9]


def test_alias_cannot_read_a_column_created_by_another_alias():
    data = pd.DataFrame({"low": [0.9, 0.1], "high": [0.1, 0.9], "Y": [1, 0]})
    with pytest.raises(ValueError, match="missing column"):
        GSQCA(data, outcome="Y", condition_specs=[
            ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"}),
            ConditionSpec("B", "multi", value_columns={0: "A=0", 1: "high"}),
        ])
