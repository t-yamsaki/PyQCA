PyQCA
=====

PyQCA is a Python-native toolkit for crisp-set, multi-value, fuzzy-set,
generalized-set, threshold-sweep, and machine-learning-enhanced Qualitative
Comparative Analysis.

The library keeps the analytical pipeline inspectable: calibrated data, truth
tables, minimized solutions, consistency and coverage, calibration choices,
candidate models, Pareto frontiers, and stability results remain available as
Python objects.

.. note::

   PyQCA is experimental and under active development. For publication or
   high-stakes analysis, validate important results against established QCA
   software and methodological expectations.

Start here
----------

* :doc:`getting_started/installation` explains the base and optional installs.
* :doc:`getting_started/quickstart` runs a complete crisp-set analysis.
* :doc:`user_guide/mlqca` covers XGBoost-assisted QCA and stability analysis.
* :doc:`api/index` documents the public Python API from the source docstrings.
* The `Japanese README <https://github.com/t-yamsaki/PyQCA/blob/main/README.ja.md>`_
  provides a Japanese project overview.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/core_qca
   user_guide/gsqca
   user_guide/calibration
   user_guide/minimization
   user_guide/sensitivity
   user_guide/reporting
   user_guide/mlqca

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   methodology
   development
   release_notes
   releasing
