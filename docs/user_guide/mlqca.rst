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

GSQCA and fuzzy workflows
-------------------------

Use :func:`qca.fit_gsqca_from_predictor` to assign crisp, fuzzy, and
multi-value calibration strategies before fitting :class:`qca.GSQCA`.
Use :func:`qca.run_anchor_sensitivity` to evaluate proposed fuzzy anchors.
Its ``dir_exp`` mapping accepts original feature names or the corresponding
``calibration.condition_map`` names, including partial specifications for
conditions other than the swept feature. Both names refer to the same
calibrated set: ``1`` expects membership and ``0`` expects non-membership.
For a low-direction feature such as ``burden``, ``{"burden": 1}`` and
``{"burden__fuzzy": 1}`` both expect membership in the low-burden set;
``0`` expects its absence. Internal negation of raw values and reversal of
the anchor grid do not invert these direction values.

Equal duplicate aliases are combined; conflicting duplicates or ambiguous
aliases raise ``ValueError`` before the sweep. Unknown names remain subject
to the engine's validation and produce invalid runs with an error message.
The input mapping is not modified. Scalar and sequence specifications retain
the existing sweep behavior; sequence order is the swept condition followed
by the other calibrated conditions in ``condition_map`` order.

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
at least five of the tutorial's six highlighted conditions within the top ten,
plus the published model-combination count. Exact top-ten ordering can vary
across XGBoost and Python runtime versions. See ``tests/data/README.md`` and
``THIRD_PARTY_NOTICES.md`` for provenance.


XGBoost feature names
---------------------

Threshold and importance extraction share the same feature-name rules.
For a named Booster, pass ``feature_names`` exactly in model order; actual
names such as ``f0`` and ``f1`` remain literal names. Reordering or renaming
this list without changing the trained model raises ``ValueError``, also
preventing mislabeling of positional SHAP contribution columns.

For an unnamed Booster (``booster.feature_names is None``), pass the original
training-column order: output keys ``f0``, ``f1``, etc. denote positions.
Lightweight adapters without a ``feature_names`` attribute may use literal
names or positional aliases only when the correspondence is unambiguous.
Unknown or ambiguous keys raise ``ValueError`` instead of silently assigning
or dropping evidence. Unused features still have zero native importance.


SHAP direction inference and measurement units
-------------------------------------------------

The inferred direction uses Pearson correlation between feature values and
XGBoost contributions over pairs where both values are finite. Positive
correlation indicates ``high`` and negative correlation indicates ``low``.
The calculation shifts and rescales each vector before centering and
normalizing, avoiding an absolute standard-deviation tolerance and overflow
or underflow from squaring values in their original units. Positive unit
changes preserve the direction, subject to floating-point representability
and the fitted model producing equivalent contributions.

A truly constant vector, fewer than two finite pairs, or unavailable raw
feature values leave direction unidentified. Correlation within the existing
dimensionless zero tolerance (absolute ``1e-8``) also has no direction.
This is a heuristic association, not a test of monotonicity or causality.

Both crisp and fuzzy proposals use the priority: ``direction_overrides``,
then ``MLConditionSpec.direction``, then an identified ``shap_direction``.
When inference is unavailable, proposals raise ``ValueError`` asking for an
explicit direction. The sign of ``shap_mean`` is no longer a fallback:
mean contribution relative to a model baseline does not identify whether
increasing a feature increases or decreases the prediction. Custom predictor
tables previously relying on that fallback must provide an identified
direction or callers must specify one explicitly. ``shap_mean`` remains
available as a descriptive output.


Direction names in calibration proposals
----------------------------------------

The public ``calibrate_crisp_conditions`` and ``calibrate_fuzzy_conditions``
APIs accept ``high``, ``positive``, ``ge``, and ``>=`` for high membership,
and ``low``, ``negative``, ``le``, and ``<=`` for low membership. Names are
case-insensitive and surrounding whitespace is ignored. Unknown directions
are rejected before calibration.

Validation and calibration use the same normalized proposal copy. Returned
``proposals.direction`` contains ``high`` or ``low``; a supplied crisp
``operator`` column is synchronized to ``>=`` or ``<=``. Other proposal
metadata and indices are preserved, and the caller's proposal table is not
modified. Crisp threshold equality remains included for both directions;
fuzzy membership at the crossover remains 0.5 for both directions.
