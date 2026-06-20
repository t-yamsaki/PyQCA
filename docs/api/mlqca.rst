mlQCA API
=========

Configuration and validation
----------------------------

.. autoclass:: qca.MLQCAConfig
   :members:

.. autoclass:: qca.MLConditionSpec
   :members:

.. autoclass:: qca.ValidatedMLQCAInput
   :members:

.. autofunction:: qca.validate_mlqca_input

Predictor
---------

.. autoclass:: qca.XGBoostBackend
   :members:

.. autoclass:: qca.PredictorFitResult
   :members:

.. autofunction:: qca.fit_xgboost_predictor

Calibration and engine adapters
-------------------------------

.. autofunction:: qca.propose_crisp_calibrations

.. autofunction:: qca.calibrate_crisp_conditions

.. autofunction:: qca.fit_csqca_from_predictor

.. autofunction:: qca.propose_fuzzy_anchors

.. autofunction:: qca.calibrate_fuzzy_conditions

.. autofunction:: qca.fit_fsqca_from_predictor

.. autofunction:: qca.calibrate_mixed_conditions

.. autofunction:: qca.fit_mixedqca_from_predictor

Search and stability
--------------------

.. autofunction:: qca.generate_condition_combinations

.. autofunction:: qca.search_csqca_models

.. autofunction:: qca.pareto_frontier

.. autoclass:: qca.MLQCAResult
   :members:

.. autofunction:: qca.cross_validate_mlqca

.. autofunction:: qca.bootstrap_mlqca

.. autoclass:: qca.MLQCAStabilityResult
   :members:
