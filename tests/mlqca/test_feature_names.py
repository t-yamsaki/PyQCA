"""Regression coverage for named and positional XGBoost features."""

import numpy as np
import pandas as pd
import pytest

from qca import extract_split_cutoffs, extract_xgboost_importance
from qca.mlqca.cutoffs import xgboost_split_nodes


class Booster:
    def __init__(self, names, keys):
        self.feature_names = names
        self.keys = keys

    def get_score(self, importance_type):
        return dict(zip(self.keys, range(1, len(self.keys) + 1), strict=True))

    def trees_to_dataframe(self):
        return pd.DataFrame(
            {
                "Feature": self.keys,
                "Split": [10.0] * len(self.keys),
                "Gain": [3.0] * len(self.keys),
                "Cover": [5.0] * len(self.keys),
            }
        )


@pytest.mark.parametrize(
    "names", [["A", "f0"], ["f0", "A"], ["f1", "f0", "f2"], ["A", "B"]]
)
@pytest.mark.parametrize("named", [True, False])
def test_metadata_controls_both_extractors(names, named):
    keys = names if named else [f"f{i}" for i in range(len(names))]
    booster = Booster(names if named else None, keys)
    nodes = xgboost_split_nodes(booster, names)
    assert nodes.feature.tolist() == names
    importance = extract_xgboost_importance(
        booster, names, split_nodes=nodes
    ).set_index("feature")
    for i, name in enumerate(names):
        assert importance.loc[name, "weight"] == i + 1
        assert importance.loc[name, "split_count"] == 1
    assert set(extract_split_cutoffs(booster, names).feature) == set(names)


@pytest.mark.parametrize("extract", [xgboost_split_nodes, extract_xgboost_importance])
@pytest.mark.parametrize(
    "metadata,keys,names,message",
    [
        (["A", "f0"], ["f0"], ["f0", "A"], "model order"),
        (["A", "B"], ["f0"], ["A", "B"], "Unknown"),
        (None, ["A"], ["A", "B"], "Unknown"),
        (None, ["f9"], ["A", "B"], "Unknown"),
        (["A", "B"], ["missing"], ["A", "B"], "Unknown"),
        (None, [], ["A", "A"], "duplicates"),
        (None, [], ["A", ""], "empty"),
        (None, [], [], "at least one"),
    ],
)
def test_invalid_mapping_is_rejected(extract, metadata, keys, names, message):
    with pytest.raises(ValueError, match=message):
        extract(Booster(metadata, keys), names)


@pytest.mark.parametrize("extract", [xgboost_split_nodes, extract_xgboost_importance])
def test_metadata_free_adapter_rejects_ambiguous_alias(extract):
    booster = Booster(None, ["f0"])
    del booster.feature_names
    with pytest.raises(ValueError, match="Ambiguous"):
        extract(booster, ["A", "f0"])


def test_metadata_free_importance_rejects_duplicate_aliases():
    booster = Booster(None, ["A", "f0"])
    del booster.feature_names
    with pytest.raises(ValueError, match="Multiple"):
        extract_xgboost_importance(booster, ["A", "B"])


@pytest.mark.parametrize("names", [["A", "f0"], ["f0", "A"], ["f1", "f0"]])
@pytest.mark.parametrize("named", [True, False])
def test_real_xgboost_preserves_feature_attribution(names, named):
    xgb = pytest.importorskip("xgboost")
    # Only f0 predicts Y; the other column is constant, so attribution is known.
    frame = pd.DataFrame(
        {name: np.arange(40) if name == "f0" else np.zeros(40) for name in names}
    )
    target = (frame["f0"] >= 20).astype(int)
    matrix = xgb.DMatrix(frame if named else frame.to_numpy(), label=target)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1, "nthread": 1, "seed": 7},
        matrix,
        num_boost_round=4,
    )
    nodes = xgboost_split_nodes(booster, names)
    assert not nodes.empty
    assert set(nodes.feature) == {"f0"}
    cutoffs = extract_split_cutoffs(booster, names)
    assert set(cutoffs.feature) == {"f0"}
    assert cutoffs.cutoff.tolist() == [20.0]
    importance = extract_xgboost_importance(
        booster,
        names,
        contributions=booster.predict(matrix, pred_contribs=True),
        feature_values=frame,
        split_nodes=nodes,
    ).set_index("feature")
    assert importance.loc["f0", "weight"] == len(nodes)
    assert importance.loc["f0", "shap_mean_abs"] > 0
    for name in names:
        if name != "f0":
            assert importance.loc[name, "weight"] == 0
            assert importance.loc[name, "shap_mean_abs"] == 0
            assert not importance.loc[name, "used"]


@pytest.mark.parametrize("extract", [xgboost_split_nodes, extract_xgboost_importance])
def test_unnamed_model_requires_matching_feature_count(extract):
    booster = Booster(None, ["f0"])
    booster.num_features = lambda: 3
    with pytest.raises(ValueError, match="feature count"):
        extract(booster, ["A", "B"])


def test_backend_keeps_named_feature_selection_and_cutoffs():
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from qca import MLQCAConfig, fit_xgboost_predictor, validate_mlqca_input

    data = pd.DataFrame(
        {"A": np.tile([0, 1], 20), "f0": np.arange(40), "Y": [0] * 20 + [1] * 20}
    )
    validated = validate_mlqca_input(data, "Y")
    result = fit_xgboost_predictor(
        validated,
        MLQCAConfig(n_jobs=1, model_params={"n_estimators": 4, "max_depth": 1}),
    )
    assert result.used_features == ("f0",)
    assert set(result.cutoff_candidates.feature) == {"f0"}
    assert result.cutoff_candidates.cutoff.tolist() == [20.0]
