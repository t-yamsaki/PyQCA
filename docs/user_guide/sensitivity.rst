Sensitivity analysis
====================

PyQCA treats thresholds and fuzzy calibration anchors as analytical choices
that should be examined rather than hidden in preprocessing.

ThresholdSweep
--------------

:class:`qca.ThresholdSweep` supports outcome, single-condition,
multi-condition grid, and dual outcome-condition sweeps.

.. code-block:: python

   from qca import ThresholdSweep

   sweep = ThresholdSweep(
       raw_data,
       outcome="Y",
       conditions=["A", "B", "C"],
       case_id="case",
   )

   result = sweep.outcome(
       thresholds=[5, 6, 7, 8],
       condition_thresholds={"A": 6, "B": 6, "C": 6},
       incl_cut=0.8,
       n_cut=1,
   )

   print(result.summary_df)
   print(result.stability)

The name ``__tsqca_Y__`` is reserved for the prepared outcome. Rename a
condition or case ID using that name before running threshold analysis.
``prepare_threshold_data`` raises ``ValueError`` before binarization for such
collisions; all four sweep modes record an invalid run with an explanatory
``error`` and no fitted solution. This also applies to ``pre_calibrated``
conditions and negated outcomes. A source outcome named ``__tsqca_Y__`` is
allowed when it is not also a condition or case ID; unused input columns with
that name do not affect analysis. Prepared data retain the existing outcome
column name and return format.

AnchorSensitivity
-----------------

:class:`qca.AnchorSensitivity` varies full non-membership, crossover, full
membership, or a multi-anchor grid.

.. code-block:: python

   from qca import AnchorSensitivity

   analysis = AnchorSensitivity(
       raw_data,
       outcome="Y",
       conditions=["capacity", "support"],
       case_id="case",
   )

   result = analysis.crossover(
       "capacity",
       values=[4, 5, 6],
       full_out=1,
       full_in=9,
       incl_cut=0.8,
   )

The calibrated output defaults to ``<condition>_fuzzy``; use ``out_col`` to
choose another name. An output name that matches the outcome, case ID, or
another model condition raises ``ValueError`` before any sweep runs. This also
applies to automatically generated names and all three single-anchor methods.
The swept condition itself may be replaced by setting ``out_col=condition``.
Existing auxiliary output columns may also be recalibrated, as in the mlQCA
adapter. Calibration takes place on copies; the input data remain unchanged.

Matplotlib is optional. Install ``pyqca[viz]`` before calling heatmap or
trajectory plotting methods.
