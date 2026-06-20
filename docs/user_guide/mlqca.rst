Machine-learning-enhanced QCA
=============================

The optional :mod:`qca.mlqca` package is a Python-native workflow informed by
the published mlQCA protocol. XGBoost supplies predictive evidence for
condition ranking and split-threshold proposals. PyQCA performs calibration,
QCA evaluation, model search, Pareto ranking, reporting, and stability
analysis.

Machine learning is evidence, not authority
-------------------------------------------

Predictive importance does not replace theory, causal interpretation, or QCA
assumptions. ``radical`` mode explores combinations from ranked conditions;
``conservative`` mode requires theory-driven constraints or explicit models.

Fit the predictor
-----------------

.. code-block:: python

   from qca import (
       MLQCAConfig,
       fit_xgboost_predictor,
       validate_mlqca_input,
   )

   validated = validate_mlqca_input(
       raw_data,
       outcome="Y",
       case_id="case",
   )
   config = MLQCAConfig(
       mode="radical",
       top_k=10,
       model_size=4,
       random_state=201,
       model_params={
           "n_estimators": 150,
           "max_depth": 4,
           "learning_rate": 0.3,
       },
   )
   predictor = fit_xgboost_predictor(validated, config)

   print(predictor.feature_importance)
   print(predictor.cutoff_candidates)

Search csQCA models
-------------------

.. code-block:: python

   from qca import search_csqca_models

   result = search_csqca_models(validated, predictor, config)

   print(result.candidate_models)
   print(result.pareto_models)
   print(result.best_qca_result)

Use ``required_conditions``, ``excluded_conditions``, or
``conservative_models`` for theory-constrained searches.

Mixed and fuzzy workflows
-------------------------

Use :func:`qca.fit_mixedqca_from_predictor` to assign crisp, fuzzy, and
multi-value calibration strategies before fitting :class:`qca.MixedQCA`.
Use :func:`qca.run_anchor_sensitivity` to evaluate proposed fuzzy anchors.

Cross-validation and bootstrap
------------------------------

.. code-block:: python

   from qca import bootstrap_mlqca, cross_validate_mlqca

   cv_result = cross_validate_mlqca(
       validated,
       config,
       n_splits=5,
       top_k=10,
   )
   bootstrap_result = bootstrap_mlqca(
       validated,
       config,
       n_bootstrap=100,
       top_k=10,
   )

   print(cv_result.run_summary)
   print(cv_result.feature_stability)
   print(bootstrap_result.cutoff_stability)

Cross-validation evaluates held-out folds. Bootstrap uses out-of-bag cases
when both outcome classes are present. The aggregate tables report condition
selection rates, feature-use rates, rank and contribution variation, and
rounded cutoff-selection rates.

Published reproduction fixture
------------------------------

The test suite includes a mechanically converted copy of the public
``voteData`` fixture from the mlQCA repository. The fixed-seed test reproduces
the tutorial's highlighted conditions and model-combination count. See
``tests/data/README.md`` and ``THIRD_PARTY_NOTICES.md`` for provenance.
