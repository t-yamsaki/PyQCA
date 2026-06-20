Reporting and reproducibility
=============================

Unified result objects can be inspected in notebooks and exported for research
records.

Markdown and LaTeX
------------------

.. code-block:: python

   from pathlib import Path

   from qca import generate_markdown_report, to_latex_table

   Path("reports").mkdir(exist_ok=True)
   generate_markdown_report(result, path="reports/analysis.md")
   to_latex_table(
       result,
       table="solutions",
       path="reports/solutions.tex",
   )

Reproducibility metadata
------------------------

.. code-block:: python

   from qca import (
       ExperimentLogger,
       collect_reproducibility_metadata,
   )

   metadata = collect_reproducibility_metadata(
       result,
       extra={"dataset": "study-1"},
   )
   ExperimentLogger("reports/experiments.jsonl").log(
       result,
       name="baseline",
   )

Jupyter summaries
-----------------

.. code-block:: python

   summary = result.to_jupyter_summary()
   display(summary)

The notebook representation intentionally provides a concise summary. Use the
DataFrame properties and export functions for complete analytical records.
