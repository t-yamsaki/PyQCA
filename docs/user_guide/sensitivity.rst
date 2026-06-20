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

Matplotlib is optional. Install ``pyqca[viz]`` before calling heatmap or
trajectory plotting methods.
