"""Truth-table grouping must not depend on internal-looking input names (#29)."""

from dataclasses import replace
from itertools import product

import pandas as pd
import pytest

from qca import CSQCA, FSQCA, GSQCA, ConditionSpec


def assert_equivalent(baseline, model, rename):
    original = model.data.copy(deep=True)
    inverse = {value: key for key, value in rename.items()}
    expected = sorted(baseline.build_truth_table(), key=lambda row: row.case_ids)
    actual = [
        replace(row, config={inverse.get(k, k): v for k, v in row.config.items()})
        for row in model.build_truth_table()
    ]
    assert sorted(actual, key=lambda row: row.case_ids) == expected
    # Compare every solution's logical patterns in the same condition order,
    # rather than comparing display strings containing the renamed conditions.
    expected_fit = baseline.fit()
    actual_fit = model.fit()
    for kind in ["complex_solution", "parsimonious_solution", "intermediate_solution"]:
        assert {imp.pattern for imp in getattr(actual_fit.qm_solution, kind)} == {
            imp.pattern for imp in getattr(expected_fit.qm_solution, kind)
        }
    assert expected_fit.qm_solution.complex_solution
    pd.testing.assert_frame_equal(model.data, original)


@pytest.mark.parametrize("path", ["crisp", "corner"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_conditions_with_multiple_collision_candidates(path, reverse):
    data = pd.DataFrame(list(product([0, 1], repeat=4)), columns=["A", "M", "N", "P"])
    data = pd.concat([data, data.iloc[[3, 7]]], ignore_index=True)
    data["Y"] = data["A"] * (1 - data["M"])
    data["case"] = [f"case_{i}" for i in data.index]
    data["F"] = data["P"] * 0.8 + 0.1
    if path == "corner":
        data["low"] = 1 - data["A"]
        data["high"] = data["A"]
        a_spec = ConditionSpec("A", "multi", value_columns={0: "low", 1: "high"})
        prefix = "__gsqca_corner_A"
    else:
        a_spec = ConditionSpec("A", "crisp")
        prefix = "__crisp_A"
    specs = [
        a_spec,
        ConditionSpec("M", "multi"),
        ConditionSpec("N", "multi"),
        ConditionSpec("P", "multi"),
        ConditionSpec("F", "fuzzy"),
    ]
    if reverse:
        specs.reverse()
    rename = {"M": prefix, "N": prefix + "_1", "P": prefix + "_2"}
    renamed = data.rename(columns=rename)
    original = renamed.copy(deep=True)
    baseline = GSQCA(data, outcome="Y", case_id="case", condition_specs=specs)
    model = GSQCA(
        renamed,
        outcome="Y",
        case_id="case",
        condition_specs=[
            replace(spec, name=rename.get(spec.name, spec.name)) for spec in specs
        ],
    )
    assert_equivalent(baseline, model, rename)
    pd.testing.assert_frame_equal(renamed, original)


@pytest.mark.parametrize("engine", [CSQCA, FSQCA, GSQCA])
@pytest.mark.parametrize("reverse", [False, True])
def test_set_condition_chain_does_not_overwrite_later_inputs(engine, reverse):
    data = pd.DataFrame(
        {
            "A": [0, 0, 1, 1],
            "B": [0, 1, 0, 1],
            "C": [1, 0, 1, 0],
            "Y": [0, 0, 1, 0],
            "case": ["a", "b", "c", "d"],
        }
    )
    rename = {"B": "__crisp_A", "C": "__crisp___crisp_A"}
    conditions = ["A", "B", "C"]
    if reverse:
        conditions.reverse()

    def create(frame, names):
        kwargs = (
            {"condition_types": dict.fromkeys(names, "crisp")}
            if engine is GSQCA
            else {}
        )
        return engine(frame, outcome="Y", case_id="case", conditions=names, **kwargs)

    assert_equivalent(
        create(data, conditions),
        create(data.rename(columns=rename), [rename.get(c, c) for c in conditions]),
        rename,
    )
