Quickstart
==========

This example constructs a crisp-set model, builds its truth table, minimizes
the sufficient configurations, and inspects the unified result object.

.. code-block:: python

   import pandas as pd

   from qca import CSQCA

   data = pd.DataFrame(
       {
           "case": [f"c{i}" for i in range(1, 9)],
           "A": [1, 1, 1, 1, 0, 0, 0, 0],
           "B": [1, 1, 0, 0, 1, 1, 0, 0],
           "C": [1, 0, 1, 0, 1, 0, 1, 0],
           "Y": [1, 1, 1, 0, 1, 0, 0, 0],
       }
   )

   model = CSQCA(
       data=data,
       outcome="Y",
       conditions=["A", "B", "C"],
       case_id="case",
   )

   result = model.fit(
       consistency_cutoff=0.8,
       coverage_cutoff=0.1,
       minimizer="standard",
   )

   print(result.truth_table)
   print(result.solutions)
   print(result.formula)

The :class:`qca.QCAFitResult` keeps the truth table, all solution types,
selected formula, consistency, coverage, and case-level coverage together.
It can also export Markdown, LaTeX tables, and formula text.

Next steps
----------

* Use :doc:`../user_guide/mixed_qca` for crisp, fuzzy, and multi-value
  conditions in one model.
* Use :doc:`../user_guide/calibration` when raw variables need set-membership
  calibration.
* Use :doc:`../user_guide/sensitivity` to assess threshold or anchor choices.
* Use :doc:`../user_guide/mlqca` for machine-learning-enhanced condition and
  cutoff proposals.
